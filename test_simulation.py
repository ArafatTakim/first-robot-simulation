import pytest

from multirobot import Config, Simulation
from multirobot.world import generate_layout


@pytest.mark.parametrize("strategy", ["greedy", "auction", "hungarian"])
def test_collect_completes(strategy):
    sim = Simulation(Config(strategy=strategy, seed=2, n_robots=4, n_targets=8))
    m = sim.run(realtime=False)
    assert m.completed
    assert m.targets_done == 8


def test_formation_completes():
    sim = Simulation(Config(scenario="formation", shape="circle", n_robots=5))
    assert sim.run(realtime=False).completed


def test_layout_is_deterministic():
    a = generate_layout(Config(seed=7))
    b = generate_layout(Config(seed=7))
    assert (a.targets == b.targets).all() and a.obstacles == b.obstacles


def test_targets_not_inside_obstacles():
    lay = generate_layout(Config(seed=3, n_obstacles=8))
    for tx, ty in lay.targets:
        for cx, cy, hx, hy in lay.obstacles:
            assert not (abs(tx - cx) < hx and abs(ty - cy) < hy)
