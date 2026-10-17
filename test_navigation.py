import numpy as np

from multirobot import Config
from multirobot.navigation import (OccupancyGrid, astar, choose_velocity,
                                   distance_field, path_length, smooth_path)
from multirobot.world import Layout


def wall_layout():
    # a wall across the middle with a gap on the right
    obstacles = [(-1.0, 0.0, 3.0, 0.2)]
    return Layout(4.0, np.zeros((1, 3)), np.zeros((0, 2)), obstacles)


def test_astar_goes_around_wall():
    grid = OccupancyGrid(wall_layout(), 0.1, 0.2)
    path = astar(grid, np.array([-2.0, -2.0]), np.array([-2.0, 2.0]))
    assert path is not None
    assert path_length(path) > 4.0 + 2.0          # must detour around the wall
    for p in path:
        assert not grid.occ[grid.to_cell(p)]


def test_smoothing_shortens_path():
    grid = OccupancyGrid(wall_layout(), 0.1, 0.2)
    raw = astar(grid, np.array([-2.0, -2.0]), np.array([-2.0, 2.0]))
    smooth = smooth_path(grid, raw)
    assert len(smooth) <= len(raw)
    assert path_length(smooth) <= path_length(raw) + 1e-6


def test_distance_field_matches_astar():
    grid = OccupancyGrid(wall_layout(), 0.1, 0.2)
    s, g = np.array([-2.0, -2.0]), np.array([2.5, 2.0])
    field = distance_field(grid, s)
    assert abs(field[grid.to_cell(g)] - path_length(astar(grid, s, g))) < 0.3


def test_unreachable_goal():
    layout = Layout(4.0, np.zeros((1, 3)), np.zeros((0, 2)), [(0.0, 0.0, 4.0, 0.2)])
    grid = OccupancyGrid(layout, 0.1, 0.2)
    assert astar(grid, np.array([0.0, -2.0]), np.array([0.0, 2.0])) is None


def test_head_on_robots_dodge():
    cfg = Config()
    a = np.array([-1.0, 0.0]); b = np.array([1.0, 0.0])
    va = np.array([0.9, 0.0]); vb = np.array([-0.9, 0.0])
    v = choose_velocity(a, va, va, [(b, vb)], [], cfg.robot_radius, cfg.max_speed,
                        cfg.avoid_horizon, cfg.n_vel_samples)
    # straight ahead would collide -> expect a lateral component or slowdown
    assert abs(v[1]) > 0.05 or v[0] < 0.8


def test_free_space_keeps_preferred_velocity():
    cfg = Config()
    pref = np.array([0.5, 0.2])
    v = choose_velocity(np.zeros(2), pref, pref, [], [], cfg.robot_radius,
                        cfg.max_speed, cfg.avoid_horizon, cfg.n_vel_samples)
    assert np.allclose(v, pref)
