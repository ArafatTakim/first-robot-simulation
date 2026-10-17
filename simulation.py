"""Simulation loop tying together MuJoCo physics, allocation and navigation."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from .allocation import get_strategy
from .config import Config
from .navigation import (OccupancyGrid, astar, choose_velocity, distance_field,
                         smooth_path)
from .world import build_mjcf, generate_layout, robot_color

DONE_COLOR = np.array([0.2, 0.8, 0.35, 1.0])
UNCLAIMED_COLOR = np.array([0.95, 0.8, 0.1, 1.0])
HIDDEN_Z = -1.0


@dataclass
class Metrics:
    strategy: str
    seed: int
    completed: bool
    makespan: float            # simulated seconds until the last target was reached
    total_distance: float      # summed over all robots (m)
    robot_collisions: int      # distinct robot-robot contact events
    obstacle_collisions: int   # distinct robot-obstacle/wall contact events
    targets_done: int
    targets_total: int

    def as_dict(self) -> dict:
        return asdict(self)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Simulation:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.layout = generate_layout(c)
        self.n = len(self.layout.robot_starts)
        self.m = len(self.layout.targets)

        self.model = mujoco.MjModel.from_xml_string(build_mjcf(c, self.layout))
        self.data = mujoco.MjData(self.model)
        self._cache_ids()

        # The planner inflates obstacles by the robot radius + a safety margin.
        self.grid = OccupancyGrid(self.layout, c.grid_res, c.robot_radius + c.safety_margin)
        self.assign_fn = get_strategy(c.strategy)
        self.rng = np.random.default_rng(c.seed + 12345)
        self.reset()

    # ------------------------------------------------------------------ ids
    def _cache_ids(self) -> None:
        m = self.model
        n = self.n
        self.body_id = [m.body(f"robot{i}").id for i in range(n)]
        self.qx = [m.joint(f"rx{i}").qposadr[0] for i in range(n)]
        self.qz = [m.joint(f"rz{i}").qposadr[0] for i in range(n)]
        self.dx = [m.joint(f"rx{i}").dofadr[0] for i in range(n)]
        self.dy = [m.joint(f"ry{i}").dofadr[0] for i in range(n)]
        self.act = [(m.actuator(f"vx{i}").id, m.actuator(f"vy{i}").id,
                     m.actuator(f"wz{i}").id) for i in range(n)]
        self.target_mocap = [m.body_mocapid[m.body(f"target{j}").id] for j in range(self.m)]
        self.target_geom = [m.geom(f"target{j}_geom").id for j in range(self.m)]
        self.geom_robot = {m.geom(f"robot{i}_body").id: i for i in range(n)}

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for i, (_, _, yaw) in enumerate(self.layout.robot_starts):
            self.data.qpos[self.qz[i]] = yaw
        for j in range(self.m):
            self.data.mocap_pos[self.target_mocap[j]] = [*self.layout.targets[j], 0.03]
            self.model.geom_rgba[self.target_geom[j]] = UNCLAIMED_COLOR
        mujoco.mj_forward(self.model, self.data)

        self.t = 0.0
        self.task = [-1] * self.n                       # target index per robot
        self.path: list[list[np.ndarray]] = [[] for _ in range(self.n)]
        self.wp = [0] * self.n
        self.done = [False] * self.m
        self.n_done = 0
        self.distance = [0.0] * self.n
        self._last_pos = [self.pos(i).copy() for i in range(self.n)]
        self._active_contacts: set = set()
        self._last_contact: dict = {}                   # contact key -> last time seen
        self.robot_collisions = 0
        self.obstacle_collisions = 0
        self._stuck_ref = [self.pos(i).copy() for i in range(self.n)]
        self._stuck_since = [0.0] * self.n
        self._nudge = [(0.0, np.zeros(2)) for _ in range(self.n)]   # (until_t, velocity)
        self._next_alloc = 0.0
        self._parked: set = set()                       # formation: robots sitting in a slot
        self.finish_time: float | None = None

    # -------------------------------------------------------------- sensing
    def pos(self, i: int) -> np.ndarray:
        return self.data.xpos[self.body_id[i]][:2].copy()

    def yaw(self, i: int) -> float:
        return float(self.data.qpos[self.qz[i]])

    def vel(self, i: int) -> np.ndarray:
        return np.array([self.data.qvel[self.dx[i]], self.data.qvel[self.dy[i]]])

    @property
    def all_done(self) -> bool:
        return self.n_done >= self.m

    # ----------------------------------------------------------- allocation
    def _cost_rows(self, robot: int, targets: list[int]) -> np.ndarray:
        """Cost from ``robot`` to each of ``targets`` (inf when unreachable)."""
        if self.cfg.cost_metric == "euclid":
            return np.array([np.linalg.norm(self.pos(robot) - self.layout.targets[j])
                             for j in targets])
        field = distance_field(self.grid, self.pos(robot))
        return np.array([field[self.grid.to_cell(self.layout.targets[j])] for j in targets])

    def _allocate(self) -> None:
        idle = [i for i in range(self.n) if self.task[i] < 0]
        if not idle:
            return
        undone = [j for j in range(self.m) if not self.done[j]]
        claimed = {self.task[i] for i in range(self.n) if self.task[i] >= 0}
        # "greedy" has no shared knowledge of claims -> it may duplicate effort.
        free = undone if self.cfg.strategy == "greedy" else [j for j in undone if j not in claimed]
        if self.cfg.scenario == "formation":
            # a parked robot already sits in its slot; only unparked robots bid
            idle = [i for i in idle if i not in self._parked]
        if not idle or not free:
            return

        cost = np.full((self.n, self.m), np.inf)
        for i in idle:
            cost[i, free] = self._cost_rows(i, free)
        for i, j in self.assign_fn(cost, idle, free).items():
            self._set_task(i, j)

    def _set_task(self, i: int, j: int) -> None:
        self.task[i] = j
        self._plan(i)
        if self.cfg.strategy != "greedy":
            self.model.geom_rgba[self.target_geom[j]] = [*robot_color(i, self.n), 1.0]

    def _plan(self, i: int) -> None:
        j = self.task[i]
        goal = self.layout.targets[j]
        p = astar(self.grid, self.pos(i), goal)
        if p is None:                       # unreachable: give the task up
            self.task[i] = -1
            self.path[i] = []
            return
        p = smooth_path(self.grid, p)
        p[-1] = goal.copy()                 # finish exactly on the target
        self.path[i] = p
        self.wp[i] = 1 if len(p) > 1 else 0

    # -------------------------------------------------------- bookkeeping
    def _check_progress(self) -> None:
        c = self.cfg
        for i in range(self.n):
            j = self.task[i]
            if j < 0:
                continue
            if self.done[j]:                                  # someone else got it
                self.task[i] = -1
                self.path[i] = []
                continue
            if np.linalg.norm(self.pos(i) - self.layout.targets[j]) < c.target_radius:
                self.done[j] = True
                self.n_done += 1
                if c.scenario == "collect":
                    self.data.mocap_pos[self.target_mocap[j]][2] = HIDDEN_Z
                else:
                    self.model.geom_rgba[self.target_geom[j]] = DONE_COLOR
                    self._parked.add(i)
                self.task[i] = -1
                self.path[i] = []
                if self.all_done and self.finish_time is None:
                    self.finish_time = self.t

    # ------------------------------------------------------------- control
    def _preferred_velocity(self, i: int) -> np.ndarray:
        c = self.cfg
        if self.task[i] < 0 or not self.path[i]:
            return np.zeros(2)
        p = self.pos(i)
        path = self.path[i]
        while self.wp[i] < len(path) - 1 and np.linalg.norm(path[self.wp[i]] - p) < 0.25:
            self.wp[i] += 1
        d = path[self.wp[i]] - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return np.zeros(2)
        last = self.wp[i] == len(path) - 1
        speed = c.max_speed * (min(1.0, max(dist / 0.6, 0.35)) if last else 1.0)
        return d / dist * speed

    def _stuck_check(self, i: int) -> None:
        """If a robot with a task barely moves for 3 s: replan and nudge it."""
        if self.task[i] < 0:
            self._stuck_ref[i] = self.pos(i)
            self._stuck_since[i] = self.t
            return
        if np.linalg.norm(self.pos(i) - self._stuck_ref[i]) > 0.15:
            self._stuck_ref[i] = self.pos(i)
            self._stuck_since[i] = self.t
        elif self.t - self._stuck_since[i] > 3.0:
            self._plan(i)
            ang = self.rng.uniform(0, 2 * math.pi)
            self._nudge[i] = (self.t + 1.0, 0.6 * self.cfg.max_speed *
                              np.array([math.cos(ang), math.sin(ang)]))
            self._stuck_ref[i] = self.pos(i)
            self._stuck_since[i] = self.t

    def _control(self) -> None:
        c = self.cfg
        states = [(self.pos(i), self.vel(i)) for i in range(self.n)]
        for i in range(self.n):
            self._stuck_check(i)
            pref = self._preferred_velocity(i)
            if self.t < self._nudge[i][0]:
                pref = self._nudge[i][1]
            others = [states[k] for k in range(self.n) if k != i]
            v = choose_velocity(states[i][0], states[i][1], pref, others,
                                self.layout.obstacles, c.robot_radius, c.max_speed,
                                c.avoid_horizon, c.n_vel_samples, c.safety_margin)
            self._drive_unicycle(i, v)

    def _drive_unicycle(self, i: int, v: np.ndarray) -> None:
        """Map a desired world velocity to (forward speed, turn rate)."""
        c = self.cfg
        speed = float(np.linalg.norm(v))
        yaw = self.yaw(i)
        if speed < 1e-3:
            fwd, w = 0.0, 0.0
        else:
            err = wrap_angle(math.atan2(v[1], v[0]) - yaw)
            w = float(np.clip(4.0 * err, -c.max_turn_rate, c.max_turn_rate))
            fwd = speed * max(0.0, math.cos(err)) ** 2
        ax, ay, aw = self.act[i]
        self.data.ctrl[ax] = fwd * math.cos(yaw)
        self.data.ctrl[ay] = fwd * math.sin(yaw)
        self.data.ctrl[aw] = w

    # ------------------------------------------------------------ contacts
    def _scan_contacts(self) -> None:
        now = set()
        d = self.data
        for k in range(d.ncon):
            con = d.contact[k]
            r1 = self.geom_robot.get(con.geom1)
            r2 = self.geom_robot.get(con.geom2)
            if r1 is None and r2 is None:
                continue
            if r1 is not None and r2 is not None:
                now.add(("rr", min(r1, r2), max(r1, r2)))
            else:
                now.add(("ro", r1 if r1 is not None else r2,
                         con.geom2 if r1 is not None else con.geom1))
        # Debounce: a contact only counts as a *new* collision if that pair had
        # been apart for at least 0.5 s (otherwise resting contact would flicker).
        for key in now - self._active_contacts:
            if self.data.time - self._last_contact.get(key, -1e9) > 0.5:
                if key[0] == "rr":
                    self.robot_collisions += 1
                else:
                    self.obstacle_collisions += 1
        for key in now:
            self._last_contact[key] = self.data.time
        self._active_contacts = now

    # ---------------------------------------------------------------- step
    def step(self) -> None:
        """Advance one control period (``cfg.control_dt`` seconds)."""
        c = self.cfg
        self._check_progress()
        if self.t >= self._next_alloc:
            self._allocate()
            self._next_alloc = self.t + 0.25
        self._control()
        for _ in range(max(1, round(c.control_dt / c.timestep))):
            mujoco.mj_step(self.model, self.data)
            self._scan_contacts()
        self.t += c.control_dt
        for i in range(self.n):
            p = self.pos(i)
            self.distance[i] += float(np.linalg.norm(p - self._last_pos[i]))
            self._last_pos[i] = p

    # ------------------------------------------------------------- results
    def metrics(self) -> Metrics:
        return Metrics(
            strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            completed=self.all_done,
            makespan=self.finish_time if self.finish_time is not None else self.t,
            total_distance=float(sum(self.distance)),
            robot_collisions=self.robot_collisions,
            obstacle_collisions=self.obstacle_collisions,
            targets_done=self.n_done,
            targets_total=self.m,
        )

    # ------------------------------------------------------------- drawing
    def draw_paths(self, scn) -> None:
        """Overlay each robot's remaining planned path on an mjvScene."""
        for i in range(self.n):
            if self.task[i] < 0 or not self.path[i]:
                continue
            rgba = np.array([*robot_color(i, self.n), 0.7], dtype=np.float32)
            pts = [self.pos(i)] + list(self.path[i][self.wp[i]:])
            for a, b in zip(pts[:-1], pts[1:]):
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), np.zeros(9), rgba)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012,
                                     np.array([a[0], a[1], 0.02]),
                                     np.array([b[0], b[1], 0.02]))
                scn.ngeom += 1

    # ----------------------------------------------------------------- run
    def run(self, viewer: bool = False, record: str | None = None,
            fps: int = 40, size: tuple[int, int] = (720, 720),
            realtime: bool = True, verbose: bool = False) -> Metrics:
        """Run until every target is reached or ``max_time`` expires.

        viewer:  open the interactive MuJoCo viewer (needs a display)
        record:  path to an .mp4/.gif to render offscreen (top-down camera)
        """
        c = self.cfg
        handle = None
        writer = renderer = None

        if viewer:
            from mujoco import viewer as mj_viewer
            handle = mj_viewer.launch_passive(self.model, self.data)
            handle.cam.lookat[:] = [0, 0, 0]
            handle.cam.distance = c.arena_size * 1.4
            handle.cam.elevation = -90
            handle.cam.azimuth = 90
        if record:
            import imageio.v2 as imageio
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(self.model, height=size[0], width=size[1])
            writer = imageio.get_writer(record, fps=fps)

        try:
            wall0 = time.time()
            while self.t < c.max_time and not self.all_done:
                self.step()
                if verbose and int(self.t / c.control_dt) % 100 == 0:
                    print(f"t={self.t:6.1f}s  done {self.n_done}/{self.m}")
                if handle is not None:
                    if not handle.is_running():
                        break
                    handle.user_scn.ngeom = 0
                    self.draw_paths(handle.user_scn)
                    handle.sync()
                    if realtime:
                        lag = self.t - (time.time() - wall0)
                        if lag > 0:
                            time.sleep(lag)
                if writer is not None:
                    renderer.update_scene(self.data, camera="top")
                    self.draw_paths(renderer.scene)
                    writer.append_data(renderer.render())
            if writer is not None:              # hold the final frame for a second
                for _ in range(fps):
                    writer.append_data(renderer.render())
        finally:
            if writer is not None:
                writer.close()
            if renderer is not None:
                renderer.close()
            if handle is not None:
                handle.close()
        return self.metrics()
