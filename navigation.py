"""Navigation stack: global A* planning + local sampling-based avoidance.

* :class:`OccupancyGrid` rasterises the box obstacles (inflated by the robot
  radius plus a safety margin) so A* can plan in configuration space.
* :func:`astar` returns a 8-connected path; :func:`smooth_path` removes
  redundant waypoints with line-of-sight checks.
* :func:`choose_velocity` is the local planner. It samples candidate velocities
  and scores them by (a) deviation from the preferred velocity, (b) predicted
  time-to-collision with other robots (velocity-obstacle style) and
  (c) clearance to static obstacles over a short horizon.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .world import Layout

SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


# --------------------------------------------------------------------------
# Occupancy grid
# --------------------------------------------------------------------------
class OccupancyGrid:
    def __init__(self, layout: Layout, res: float, inflate: float):
        self.res = res
        self.half = layout.arena_half
        self.n = int(round(2 * self.half / res))
        xs = (np.arange(self.n) + 0.5) * res - self.half
        X, Y = np.meshgrid(xs, xs, indexing="ij")          # X[i, j] -> x of cell (i, j)
        occ = np.zeros((self.n, self.n), dtype=bool)
        for cx, cy, hx, hy in layout.obstacles:
            occ |= (np.abs(X - cx) <= hx + inflate) & (np.abs(Y - cy) <= hy + inflate)
        # arena boundary
        occ |= (np.abs(X) > self.half - inflate) | (np.abs(Y) > self.half - inflate)
        self.occ = occ

    # -- coordinate helpers --------------------------------------------------
    def to_cell(self, p) -> tuple[int, int]:
        i = int((p[0] + self.half) / self.res)
        j = int((p[1] + self.half) / self.res)
        return (min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1))

    def to_world(self, c) -> np.ndarray:
        return np.array([(c[0] + 0.5) * self.res - self.half,
                         (c[1] + 0.5) * self.res - self.half])

    def nearest_free(self, c: tuple[int, int], max_r: int = 10) -> tuple[int, int] | None:
        if not self.occ[c]:
            return c
        for r in range(1, max_r + 1):
            best, best_d = None, 1e9
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    i, j = c[0] + di, c[1] + dj
                    if 0 <= i < self.n and 0 <= j < self.n and not self.occ[i, j]:
                        d = di * di + dj * dj
                        if d < best_d:
                            best, best_d = (i, j), d
            if best is not None:
                return best
        return None

    def line_of_sight(self, a: np.ndarray, b: np.ndarray) -> bool:
        d = np.linalg.norm(b - a)
        steps = max(int(d / (self.res * 0.5)), 1)
        for t in np.linspace(0.0, 1.0, steps + 1):
            if self.occ[self.to_cell(a + (b - a) * t)]:
                return False
        return True


# --------------------------------------------------------------------------
# A*
# --------------------------------------------------------------------------
def astar(grid: OccupancyGrid, start_xy, goal_xy) -> list[np.ndarray] | None:
    """8-connected A*. Returns world-space waypoints (start..goal) or None."""
    s = grid.nearest_free(grid.to_cell(start_xy))
    g = grid.nearest_free(grid.to_cell(goal_xy))
    if s is None or g is None:
        return None

    def h(c):
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)      # octile distance

    open_heap = [(h(s), 0.0, s)]
    came: dict = {}
    best_g = {s: 0.0}
    closed = set()
    n = grid.n
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return [grid.to_world(c) for c in path]
        closed.add(cur)
        for di, dj, cost in _NEIGHBOURS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < n and 0 <= nj < n) or grid.occ[ni, nj]:
                continue
            if di and dj and (grid.occ[cur[0] + di, cur[1]] or grid.occ[cur[0], cur[1] + dj]):
                continue                                     # no corner cutting
            nxt = (ni, nj)
            ng = gc + cost
            if ng < best_g.get(nxt, 1e18):
                best_g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def distance_field(grid: OccupancyGrid, start_xy) -> np.ndarray:
    """Dijkstra flood-fill: shortest 8-connected path length (metres) from
    ``start_xy`` to every grid cell (``inf`` where unreachable).

    One flood per robot gives path costs to *all* targets at once, which is
    much cheaper than running A* for every robot/target pair.
    """
    n = grid.n
    dist = np.full((n, n), np.inf)
    s = grid.nearest_free(grid.to_cell(start_xy))
    if s is None:
        return dist
    dist[s] = 0.0
    heap = [(0.0, s)]
    occ = grid.occ
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for di, dj, cost in _NEIGHBOURS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < n and 0 <= nj < n) or occ[ni, nj]:
                continue
            if di and dj and (occ[cur[0] + di, cur[1]] or occ[cur[0], cur[1] + dj]):
                continue
            nd = d + cost * grid.res
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                heapq.heappush(heap, (nd, (ni, nj)))
    return dist


def smooth_path(grid: OccupancyGrid, path: list[np.ndarray]) -> list[np.ndarray]:
    """Greedy line-of-sight shortcutting ("string pulling")."""
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not grid.line_of_sight(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_length(path: list[np.ndarray] | None) -> float:
    if path is None:
        return float("inf")
    return float(sum(np.linalg.norm(path[k + 1] - path[k]) for k in range(len(path) - 1)))


# --------------------------------------------------------------------------
# Local avoidance (sampling-based velocity obstacles)
# --------------------------------------------------------------------------
def _point_box_dist(P: np.ndarray, box) -> np.ndarray:
    """Distance from points P (..., 2) to an axis-aligned box (cx, cy, hx, hy)."""
    cx, cy, hx, hy = box
    dx = np.maximum(np.abs(P[..., 0] - cx) - hx, 0.0)
    dy = np.maximum(np.abs(P[..., 1] - cy) - hy, 0.0)
    return np.hypot(dx, dy)


def _time_to_collision(p_rel: np.ndarray, v_rel: np.ndarray, R: float) -> np.ndarray:
    """Earliest t >= 0 with |p_rel + v_rel * t| = R, for many v_rel at once.

    p_rel: (2,) neighbour position relative to us. v_rel: (K, 2) our candidate
    velocity minus neighbour velocity. Returns (K,) array, inf if no collision.
    """
    # we move relative to the neighbour: position(t) = -p_rel + v_rel * t  (|.| = R)
    a = np.einsum("ki,ki->k", v_rel, v_rel)
    b = -2.0 * (v_rel @ p_rel)
    c = float(p_rel @ p_rel) - R * R
    ttc = np.full(len(v_rel), np.inf)
    if c <= 0.0:                                  # already overlapping
        ttc[:] = 0.0
        return ttc
    ok = (a > 1e-9) & (b < 0.0)
    disc = b * b - 4.0 * a * c
    ok &= disc >= 0.0
    ttc[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2.0 * a[ok])
    return ttc


def choose_velocity(
    pos: np.ndarray,
    vel: np.ndarray,
    pref_vel: np.ndarray,
    neighbours: list[tuple[np.ndarray, np.ndarray]],
    obstacles: list[tuple[float, float, float, float]],
    radius: float,
    max_speed: float,
    horizon: float,
    n_samples: int,
    margin: float = 0.08,
) -> np.ndarray:
    """Pick a collision-free velocity close to ``pref_vel``."""
    # Candidate set: preferred velocity, scaled copies, and a polar sampling.
    cands = [pref_vel, pref_vel * 0.5, np.zeros(2)]
    n_ang = max(n_samples // 3, 8)
    for speed_frac in (1.0, 0.6, 0.3):
        for ang in np.linspace(0, 2 * math.pi, n_ang, endpoint=False):
            cands.append(speed_frac * max_speed * np.array([math.cos(ang), math.sin(ang)]))
    C = np.asarray(cands)                                             # (K, 2)
    cost = np.linalg.norm(C - pref_vel, axis=1)
    # small penalty for changing velocity abruptly keeps motion smooth
    cost += 0.15 * np.linalg.norm(C - vel, axis=1)

    # ---- other robots ----
    R = 2.0 * radius + margin
    for n_pos, n_vel in neighbours:
        p_rel = n_pos - pos
        if np.linalg.norm(p_rel) > R + max_speed * 2 * horizon:
            continue
        # Both robots run the same rule, so avoidance effort is shared implicitly.
        v_rel = C - n_vel
        ttc = _time_to_collision(p_rel, v_rel, R)
        hit = ttc < horizon
        cost[hit] += 3.0 * (horizon - ttc[hit]) / horizon + 0.5

    # ---- static obstacles ----
    ts = np.linspace(0.25, horizon * 0.75, 4)
    pts = pos[None, None, :] + C[:, None, :] * ts[None, :, None]       # (K, T, 2)
    for box in obstacles:
        d = _point_box_dist(pts, box)                                   # (K, T)
        pen = np.clip((radius + margin - d) / (radius + margin), 0.0, None)
        cost += 4.0 * pen.max(axis=1)

    return C[int(np.argmin(cost))]
