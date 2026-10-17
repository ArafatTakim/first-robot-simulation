#!/usr/bin/env python3
"""Compare allocation strategies over many random worlds.

    python scripts/benchmark.py --seeds 20 --robots 6 --targets 16

Writes results/benchmark.csv and results/benchmark.png and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multirobot import Config, Simulation  # noqa: E402

STRATEGIES = ["greedy", "auction", "hungarian"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--robots", type=int, default=6)
    ap.add_argument("--targets", type=int, default=16)
    ap.add_argument("--obstacles", type=int, default=8)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        for strat in STRATEGIES:
            cfg = Config(strategy=strat, seed=seed, n_robots=args.robots,
                         n_targets=args.targets, n_obstacles=args.obstacles)
            m = Simulation(cfg).run(realtime=False)
            rows.append(m.as_dict())
            print(f"seed {seed:2d}  {strat:9s}  T={m.makespan:5.1f}s  "
                  f"dist={m.total_distance:6.1f}m  coll={m.robot_collisions}")

    with open(out / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ---- summary table
    print("\n| strategy | success | makespan (s) | distance (m) | robot collisions |")
    print("|---|---|---|---|---|")
    summary = {}
    for s in STRATEGIES:
        r = [x for x in rows if x["strategy"] == s]
        mk = np.array([x["makespan"] for x in r])
        di = np.array([x["total_distance"] for x in r])
        co = np.array([x["robot_collisions"] for x in r])
        ok = np.mean([x["completed"] for x in r])
        summary[s] = (mk, di, co)
        print(f"| {s} | {ok:.0%} | {mk.mean():.1f} ± {mk.std():.1f} | "
              f"{di.mean():.1f} ± {di.std():.1f} | {co.mean():.2f} |")

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    titles = ["Makespan (s)", "Total distance (m)", "Robot collisions"]
    colors = ["#e07a5f", "#3d8bb8", "#5a9e6f"]
    for k, ax in enumerate(axes):
        data = [summary[s][k] for s in STRATEGIES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55)
        ax.set_xticks(range(1, len(STRATEGIES) + 1), STRATEGIES)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_title(titles[k], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{args.robots} robots, {args.targets} targets, {args.obstacles} obstacles "
                 f"({args.seeds} random worlds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=150)
    print(f"\nSaved {out/'benchmark.csv'} and {out/'benchmark.png'}")


if __name__ == "__main__":
    main()
