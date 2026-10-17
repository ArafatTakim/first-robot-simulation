"""Procedural scene generation.

The arena is described by a plain-Python :class:`Layout` (robot starts, target
positions, axis-aligned box obstacles).  The same layout feeds both the MJCF
XML that MuJoCo loads and the occupancy grid used by the path planner, so the
planner's view of the world always matches the physics.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from .config import Config

WALL_THICKNESS = 0.1
ROBOT_HALF_HEIGHT = 0.06


@dataclass
class Layout:
    arena_half: float
    robot_starts: np.ndarray                 # (N, 3)  x, y, yaw
    targets: np.ndarray                      # (M, 2)
    obstacles: list[tuple[float, float, float, float]] = field(default_factory=list)
    # each obstacle: (cx, cy, half_x, half_y)


def robot_color(i: int, n: int) -> tuple[float, float, float]:
    """Evenly spaced, saturated hues -> one distinct colour per robot."""
    return colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.95)


# --------------------------------------------------------------------------
# layout generation
# --------------------------------------------------------------------------
def formation_slots(cfg: Config) -> np.ndarray:
    n = cfg.n_robots
    if cfg.shape == "circle":
        r = max(1.0, 0.28 * n)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[r * np.cos(ang), r * np.sin(ang)]
    if cfg.shape == "line":
        xs = np.linspace(-0.45 * cfg.arena_size, 0.45 * cfg.arena_size, n)
        return np.c_[xs, np.full(n, 0.5)]
    if cfg.shape == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        pts = [((k % cols) - (cols - 1) / 2, (k // cols) - 0.5 * ((n - 1) // cols))
               for k in range(n)]
        return np.array(pts, dtype=float) * 0.9
    raise ValueError(f"unknown formation shape: {cfg.shape}")


def _point_box_dist(p: np.ndarray, box: tuple[float, float, float, float]) -> float:
    cx, cy, hx, hy = box
    dx = max(abs(p[0] - cx) - hx, 0.0)
    dy = max(abs(p[1] - cy) - hy, 0.0)
    return float(np.hypot(dx, dy))


def generate_layout(cfg: Config) -> Layout:
    rng = np.random.default_rng(cfg.seed)
    half = cfg.arena_size / 2

    # Robots start in a row along the bottom edge, facing "up".
    xs = np.linspace(-half * 0.7, half * 0.7, cfg.n_robots) if cfg.n_robots > 1 else [0.0]
    starts = np.array([[x, -half + 0.6, np.pi / 2] for x in xs])

    fixed_pts = starts[:, :2].copy()
    slots = None
    if cfg.scenario == "formation":
        slots = formation_slots(cfg)
        fixed_pts = np.vstack([fixed_pts, slots])

    # Obstacles: random boxes that keep clear of starts / formation slots.
    obstacles: list[tuple[float, float, float, float]] = []
    tries = 0
    while len(obstacles) < cfg.n_obstacles and tries < 2000:
        tries += 1
        hx, hy = rng.uniform(0.2, 0.55, size=2)
        if rng.random() < 0.5:                     # make some of them elongated
            hx, hy = (hx * 1.8, hy) if rng.random() < 0.5 else (hx, hy * 1.8)
        c = rng.uniform(-half + 1.2, half - 1.2, size=2)
        box = (float(c[0]), float(c[1]), float(hx), float(hy))
        if any(_point_box_dist(p, box) < 1.0 for p in fixed_pts):
            continue
        # keep corridors open between obstacles
        if any(abs(c[0] - o[0]) < hx + o[2] + 0.9 and abs(c[1] - o[1]) < hy + o[3] + 0.9
               for o in obstacles):
            continue
        obstacles.append(box)

    # Targets
    if slots is not None:
        targets = slots
    else:
        pts: list[np.ndarray] = []
        tries = 0
        while len(pts) < cfg.n_targets and tries < 5000:
            tries += 1
            p = rng.uniform(-half + 0.6, half - 0.6, size=2)
            if any(_point_box_dist(p, b) < 0.5 for b in obstacles):
                continue
            if any(np.linalg.norm(p - q) < 0.6 for q in pts):
                continue
            if np.min(np.linalg.norm(starts[:, :2] - p, axis=1)) < 0.8:
                continue
            pts.append(p)
        targets = np.array(pts)

    return Layout(half, starts, targets, obstacles)


# --------------------------------------------------------------------------
# MJCF
# --------------------------------------------------------------------------
def build_mjcf(cfg: Config, layout: Layout) -> str:
    n = len(layout.robot_starts)
    half = layout.arena_half
    cam_h = cfg.arena_size * 1.25
    r = cfg.robot_radius

    parts: list[str] = []
    a = parts.append
    a(f"""<mujoco model="multi_robot_arena">
  <compiler angle="radian"/>
  <option timestep="{cfg.timestep}" integrator="implicitfast" gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.15 0.15 0.15" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.80 0.83 0.88"
             rgb2="0.72 0.76 0.82" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="{cfg.arena_size:.0f} {cfg.arena_size:.0f}" reflectance="0.0"/>
    <material name="wall" rgba="0.25 0.27 0.32 1"/>
    <material name="obstacle" rgba="0.45 0.47 0.55 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 {cam_h:.1f}" dir="0 0 -1" directional="true" diffuse="0.35 0.35 0.35" specular="0 0 0" castshadow="false"/>
    <camera name="top" pos="0 0 {cam_h:.2f}" mode="fixed"/>
    <geom name="floor" type="plane" size="{half:.2f} {half:.2f} 0.1" material="floor"
          contype="0" conaffinity="0"/>""")

    # Walls
    t = WALL_THICKNESS
    for name, (px, py, sx, sy) in {
        "wall_n": (0, half + t, half + 2 * t, t),
        "wall_s": (0, -half - t, half + 2 * t, t),
        "wall_e": (half + t, 0, t, half + 2 * t),
        "wall_w": (-half - t, 0, t, half + 2 * t),
    }.items():
        a(f'    <geom name="{name}" type="box" pos="{px:.3f} {py:.3f} 0.1" '
          f'size="{sx:.3f} {sy:.3f} 0.1" material="wall"/>')

    # Obstacles
    for k, (cx, cy, hx, hy) in enumerate(layout.obstacles):
        a(f'    <geom name="obstacle{k}" type="box" pos="{cx:.3f} {cy:.3f} 0.1" '
          f'size="{hx:.3f} {hy:.3f} 0.1" material="obstacle"/>')

    # Targets (mocap bodies so we can move/hide them at runtime)
    for j, (tx, ty) in enumerate(layout.targets):
        a(f'    <body name="target{j}" mocap="true" pos="{tx:.3f} {ty:.3f} 0.03">'
          f'<geom name="target{j}_geom" type="cylinder" size="0.09 0.015" '
          f'rgba="0.95 0.8 0.1 1" contype="0" conaffinity="0"/></body>')

    # Robots: planar (x, y, yaw) joints. Slides come first so their axes stay
    # world-aligned; the initial yaw is written into qpos by the simulation.
    for i, (x, y, yaw) in enumerate(layout.robot_starts):
        cr, cg, cb = robot_color(i, n)
        a(f"""    <body name="robot{i}" pos="{x:.3f} {y:.3f} {ROBOT_HALF_HEIGHT}">
      <joint name="rx{i}" type="slide" axis="1 0 0" damping="0.0"/>
      <joint name="ry{i}" type="slide" axis="0 1 0" damping="0.0"/>
      <joint name="rz{i}" type="hinge" axis="0 0 1" damping="0.0"/>
      <geom name="robot{i}_body" type="cylinder" size="{r} {ROBOT_HALF_HEIGHT}"
            rgba="{cr:.3f} {cg:.3f} {cb:.3f} 1" mass="1.0" friction="0.1 0.005 0.0001"/>
      <geom name="robot{i}_nose" type="box" pos="{r * 0.7:.3f} 0 {ROBOT_HALF_HEIGHT}"
            size="{r * 0.28:.3f} {r * 0.14:.3f} 0.012" rgba="1 1 1 1"
            contype="0" conaffinity="0" mass="0"/>
    </body>""")
    a("  </worldbody>\n  <actuator>")
    for i in range(n):
        a(f'    <velocity name="vx{i}" joint="rx{i}" kv="40" ctrlrange="-3 3"/>')
        a(f'    <velocity name="vy{i}" joint="ry{i}" kv="40" ctrlrange="-3 3"/>')
        a(f'    <velocity name="wz{i}" joint="rz{i}" kv="2.0" ctrlrange="-8 8"/>')
    a("  </actuator>\n</mujoco>")
    return "\n".join(parts)
