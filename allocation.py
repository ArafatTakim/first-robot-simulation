"""Task-allocation strategies.

All strategies share one interface::

    assign(cost, idle_robots, free_targets) -> {robot_idx: target_idx}

``cost[i, j]`` is the cost for robot ``i`` to reach target ``j`` (``inf`` if
unreachable). Only *idle* robots are (re)assigned, and only to targets that
nobody has claimed, so robots already en route are never disturbed.

* ``greedy``    - every robot independently grabs its nearest target. There is
                  no communication, so several robots may pick the same one
                  (the baseline the other strategies are compared against).
* ``auction``   - decentralised parallel single-item auction. Each idle robot
                  bids on its cheapest target; the lowest bid per target wins;
                  losers re-bid in the next round.
* ``hungarian`` - centralised optimal assignment minimising total cost.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6

Assignment = dict[int, int]
Strategy = Callable[[np.ndarray, Sequence[int], Sequence[int]], Assignment]


def greedy(cost: np.ndarray, idle: Sequence[int], free: Sequence[int]) -> Assignment:
    out: Assignment = {}
    if not len(free):
        return out
    for i in idle:
        j = min(free, key=lambda t: cost[i, t])
        if np.isfinite(cost[i, j]):
            out[i] = j
    return out


def auction(cost: np.ndarray, idle: Sequence[int], free: Sequence[int]) -> Assignment:
    out: Assignment = {}
    bidders = list(idle)
    items = list(free)
    while bidders and items:
        bids: dict[int, tuple[float, int]] = {}      # target -> (best bid, robot)
        for i in bidders:
            j = min(items, key=lambda t: cost[i, t])
            if not np.isfinite(cost[i, j]):
                continue
            if j not in bids or cost[i, j] < bids[j][0]:
                bids[j] = (cost[i, j], i)
        if not bids:
            break
        for j, (_, i) in bids.items():
            out[i] = j
            bidders.remove(i)
            items.remove(j)
    return out


def hungarian(cost: np.ndarray, idle: Sequence[int], free: Sequence[int]) -> Assignment:
    if not len(idle) or not len(free):
        return {}
    sub = cost[np.ix_(list(idle), list(free))]
    sub = np.where(np.isfinite(sub), sub, BIG)
    rows, cols = linear_sum_assignment(sub)
    return {idle[r]: free[c] for r, c in zip(rows, cols) if sub[r, c] < BIG}


STRATEGIES: dict[str, Strategy] = {
    "greedy": greedy,
    "auction": auction,
    "hungarian": hungarian,
}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy '{name}', choose from {list(STRATEGIES)}") from None
