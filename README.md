# Multi-Robot Coordination in MuJoCo

A compact, hackable simulation of a robot team that has to **share work, plan paths and avoid each other**
inside a cluttered arena, built on the [MuJoCo](https://mujoco.org) physics engine.

<p align="center">
  <img src="docs/demo_collect.gif" width="380" alt="Target collection with Hungarian allocation">
  <img src="docs/demo_formation.gif" width="380" alt="Circle formation">
</p>

*Left: 6 robots collect 16 targets (each target takes its assigned robot's colour, lines show planned paths).
Right: 8 robots swap into a circle formation.*

## What it demonstrates

| Layer | Technique | File |
|---|---|---|
| Physics | Planar robots (x, y, yaw joints + velocity actuators) with real contact against walls, obstacles and each other | `world.py` |
| Task allocation | **Greedy** (no communication), **Auction** (decentralised), **Hungarian** (centralised optimal) | `allocation.py` |
| Global planning | Occupancy grid, Dijkstra distance fields for path-cost matrices, A\*, line-of-sight path smoothing | `navigation.py` |
| Local avoidance | Sampling-based velocity obstacles (time-to-collision) + static obstacle clearance | `navigation.py` |
| Control | Unicycle (differential-drive) mapping from desired velocity to forward speed and turn rate | `simulation.py` |
| Robustness | Stuck detection, replanning and random nudges | `simulation.py` |
| Evaluation | Makespan, total distance, debounced collision counting, multi-seed benchmark | `scripts/benchmark.py` |

Two scenarios are included: `collect` (visit all targets as fast as possible) and `formation`
(move into a circle / line / grid; slots are assigned with the same allocation code).

## Quick start

```bash
git clone <your-repo-url> && cd multi-robot-mujoco
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py --viewer                                   # interactive MuJoCo window
python run.py --strategy auction --robots 6 --targets 16 --obstacles 8
python run.py --scenario formation --shape circle --robots 8
python run.py --record videos/run.mp4 --gl egl           # headless video (or .gif)
```

On macOS, launching the interactive viewer needs `mjpython run.py --viewer`.
For headless video on Linux use `--gl egl` (GPU) or `--gl osmesa` (software).

### Useful flags

`--strategy {greedy,auction,hungarian}` · `--cost {path,euclid}` · `--robots N` · `--targets N` ·
`--obstacles N` · `--seed S` · `--scenario {collect,formation}` · `--shape {circle,line,grid}`

### Use it as a library

```python
from multirobot import Config, Simulation

sim = Simulation(Config(strategy="auction", n_robots=8, n_targets=20, seed=3))
m = sim.run(realtime=False)
print(m.makespan, m.total_distance, m.robot_collisions)
```

## Results

20 random worlds, 6 robots, 16 targets, 8 obstacles (`python scripts/benchmark.py --seeds 20`):

| strategy | success | makespan (s) | distance (m) | robot collisions |
|---|---|---|---|---|
| greedy | 100% | 23.1 ± 8.9 | 63.9 ± 17.6 | 1.15 |
| auction | 100% | 13.8 ± 6.5 | 39.6 ± 3.8 | 0.10 |
| hungarian | 100% | 12.3 ± 2.9 | 38.3 ± 3.7 | 0.10 |

![benchmark](results/benchmark.png)

Take-aways:

* Coordination matters: even a simple auction cuts travel distance by ~38% over uncoordinated greedy,
  because greedy robots keep racing for the same target and crowd each other.
* The decentralised auction lands close to the centralised optimum, with higher variance.
* Hungarian minimises *total cost*, not makespan, so on some seeds a single long assignment can still dominate
  the finish time. Minimising makespan directly (a min-max / bottleneck assignment) is a nice extension.

## How it works

```
          +-------------------+      cost matrix       +-------------------+
 world -> | Dijkstra distance | ---------------------> | allocation        |
 layout   | fields (per robot)|                        | greedy/auction/   |
          +-------------------+                        | hungarian         |
                                                       +---------+---------+
                                                                 | robot -> target
                                                       +---------v---------+
                                                       | A* + smoothing    |
                                                       +---------+---------+
                                                                 | waypoints
 +--------------+  desired velocity   +----------------+   preferred velocity
 | unicycle     | <------------------ | local avoidance| <---------+
 | controller   |                     | (VO sampling)  |
 +------+-------+                     +----------------+
        | vx, vy, wz velocity commands
 +------v-------+
 |   MuJoCo     |  contacts -> collision metrics
 +--------------+
```

Allocation only touches *idle* robots and unclaimed targets, so robots already on their way are never
reshuffled. The control loop runs at 20 Hz while physics steps at 200 Hz.

## Project layout

```
multirobot/
  config.py       Config dataclass (all tunables)
  world.py        procedural layout + MJCF generation
  allocation.py   greedy / auction / hungarian
  navigation.py   occupancy grid, A*, distance field, velocity sampling
  simulation.py   physics loop, metrics, viewer + video recording
scripts/          benchmark.py, make_demo.py
tests/            pytest suite (allocation, navigation, end-to-end)
run.py            command-line entry point
```

## Tests

```bash
python -m pytest -q
```

## Ideas to extend

* Battery / charging stations, or a depot that robots must deliver targets to
* Dynamic targets that spawn over time (re-auction on arrival)
* Bundle auctions (several tasks per robot) or min-max bottleneck assignment
* Real wheeled robots with MuJoCo wheel actuators instead of the velocity-driven planar model
* Communication limits (robots only see neighbours within a radius)
* Heterogeneous teams (different speeds or capabilities)

## License

MIT
