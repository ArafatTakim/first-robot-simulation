import numpy as np

from multirobot.allocation import auction, greedy, hungarian


def total(cost, assign):
    return sum(cost[i, j] for i, j in assign.items())


def test_hungarian_is_optimal():
    # greedy picks (0->0) first and is forced into a worse total
    cost = np.array([[1.0, 2.0],
                     [1.1, 9.0]])
    a = hungarian(cost, [0, 1], [0, 1])
    assert a == {0: 1, 1: 0}
    assert total(cost, a) < total(cost, {0: 0, 1: 1})


def test_auction_gives_unique_targets():
    rng = np.random.default_rng(0)
    cost = rng.uniform(1, 10, size=(5, 8))
    a = auction(cost, list(range(5)), list(range(8)))
    assert len(a) == 5
    assert len(set(a.values())) == 5


def test_greedy_can_duplicate_targets():
    cost = np.array([[1.0, 5.0], [1.0, 5.0]])
    a = greedy(cost, [0, 1], [0, 1])
    assert a[0] == a[1] == 0


def test_unreachable_targets_are_skipped():
    cost = np.array([[np.inf, 3.0], [np.inf, 4.0]])
    for fn in (greedy, auction, hungarian):
        a = fn(cost, [0, 1], [0, 1])
        assert 0 not in a.values()
    assert hungarian(cost, [0, 1], [0, 1]) in ({0: 1}, {1: 1})


def test_more_robots_than_targets():
    cost = np.array([[1.0], [2.0], [3.0]])
    assert hungarian(cost, [0, 1, 2], [0]) == {0: 0}
    assert auction(cost, [0, 1, 2], [0]) == {0: 0}
