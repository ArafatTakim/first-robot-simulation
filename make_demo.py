#!/usr/bin/env python3
"""Render the demo GIFs used in the README (needs MUJOCO_GL=egl on headless boxes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multirobot import Config, Simulation  # noqa: E402

DEMOS = {
    "docs/demo_collect.gif": Config(strategy="hungarian", seed=2, n_robots=6,
                                    n_targets=16, n_obstacles=8),
    "docs/demo_formation.gif": Config(scenario="formation", shape="circle",
                                      n_robots=8, n_obstacles=4, seed=1),
}

if __name__ == "__main__":
    for path, cfg in DEMOS.items():
        m = Simulation(cfg).run(record=path, fps=20, size=(480, 480), realtime=False)
        print(f"{path}: makespan {m.makespan:.1f}s, collisions {m.robot_collisions}")
