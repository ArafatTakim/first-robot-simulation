#!/usr/bin/env python3
"""Command-line entry point.

Examples
--------
    python run.py --viewer                       # interactive window
    python run.py --strategy auction --robots 6
    python run.py --scenario formation --shape circle --record videos/circle.mp4
"""
from __future__ import annotations

import argparse
import os

from multirobot import Config, Simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-robot coordination in MuJoCo")
    p.add_argument("--scenario", choices=["collect", "formation"], default="collect")
    p.add_argument("--shape", choices=["circle", "line", "grid"], default="circle",
                   help="formation shape (formation scenario only)")
    p.add_argument("--strategy", choices=["greedy", "auction", "hungarian"], default="hungarian")
    p.add_argument("--cost", choices=["path", "euclid"], default="path", dest="cost_metric")
    p.add_argument("--robots", type=int, default=5)
    p.add_argument("--targets", type=int, default=12)
    p.add_argument("--obstacles", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-time", type=float, default=120.0)
    p.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    p.add_argument("--record", metavar="FILE", help="render a top-down video (.mp4 or .gif)")
    p.add_argument("--gl", choices=["egl", "osmesa", "glfw"], default=None,
                   help="MuJoCo OpenGL backend for headless recording")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.gl:
        os.environ["MUJOCO_GL"] = a.gl
    cfg = Config(scenario=a.scenario, shape=a.shape, strategy=a.strategy,
                 cost_metric=a.cost_metric, n_robots=a.robots, n_targets=a.targets,
                 n_obstacles=a.obstacles, seed=a.seed, max_time=a.max_time)
    sim = Simulation(cfg)
    m = sim.run(viewer=a.viewer, record=a.record, verbose=True)

    print("\n=== Result ===")
    print(f"strategy            : {m.strategy}")
    print(f"completed           : {m.completed} ({m.targets_done}/{m.targets_total} targets)")
    print(f"makespan            : {m.makespan:.1f} s (simulated)")
    print(f"total distance      : {m.total_distance:.1f} m")
    print(f"robot collisions    : {m.robot_collisions}")
    print(f"obstacle collisions : {m.obstacle_collisions}")


if __name__ == "__main__":
    main()
