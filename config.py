"""Central configuration for a simulation run."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- scenario ---------------------------------------------------------
    scenario: str = "collect"        # "collect" | "formation"
    shape: str = "circle"            # formation shape: "circle" | "line" | "grid"
    n_robots: int = 5
    n_targets: int = 12
    n_obstacles: int = 6
    arena_size: float = 8.0          # square arena, side length in metres
    seed: int = 0

    # --- coordination -----------------------------------------------------
    strategy: str = "hungarian"      # "greedy" | "auction" | "hungarian"
    cost_metric: str = "path"        # "path" (A* length) | "euclid"

    # --- robot model ------------------------------------------------------
    robot_radius: float = 0.15
    max_speed: float = 0.9           # m/s
    max_turn_rate: float = 3.0       # rad/s
    target_radius: float = 0.12      # a target is reached within this distance

    # --- navigation -------------------------------------------------------
    grid_res: float = 0.1            # occupancy grid resolution (m)
    safety_margin: float = 0.08      # extra obstacle inflation (m)
    avoid_horizon: float = 2.0       # seconds look-ahead for collision checks
    n_vel_samples: int = 48          # candidate velocities per robot

    # --- timing -----------------------------------------------------------
    timestep: float = 0.005          # physics dt
    control_dt: float = 0.05         # controller / planner period
    max_time: float = 120.0          # simulated-seconds timeout
