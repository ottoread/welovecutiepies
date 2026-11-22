"""Exact optimal worker bin-packing for Blast Radius scoring."""
from __future__ import annotations

import math

WORKER_COST_MS = 2000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def _lpt_pack(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    loads = [0] * k
    bins: list[list[int]] = [[] for _ in range(k)]
    for dur, num in items:
        idx = loads.index(min(loads))
        bins[idx].append(num)
        loads[idx] += dur
    return max(loads), bins


def _exact_pack(items: list[tuple[int, int]], k: int, ub: int, lb: int, max_nodes: int = 8000) -> tuple[int, list[list[int]]] | None:
    """Branch-and-bound makespan minimization for k identical workers."""
    if ub <= lb:
        return None
    n = len(items)
    durs = [d for d, _ in items]
    loads = [0] * k
    assign = [0] * n
    best_max = ub
    best_assign: list[int] | None = None
    nodes = 0

    def dfs(idx: int, cur_max: int) -> None:
        nonlocal best_max, best_assign, nodes
        if nodes >= max_nodes:
            return
        if cur_max >= best_max:
            return
        if idx == n:
            best_max = cur_max
            best_assign = assign.copy()
            return
        nodes += 1
        d = durs[idx]
        seen_loads: set[int] = set()
        for b in range(k):
            lb_b = loads[b]
            if lb_b in seen_loads:
                continue
            new_load = lb_b + d
            if new_load >= best_max:
                continue
            seen_loads.add(lb_b)
            loads[b] = new_load
            assign[idx] = b
            dfs(idx + 1, new_load if new_load > cur_max else cur_max)
            loads[b] = lb_b
            if best_max <= lb:
                return

    dfs(0, 0)
    if best_assign is None:
        return None
    bins: list[list[int]] = [[] for _ in range(k)]
    for (_, num), b in zip(items, best_assign):
        bins[b].append(num)
    return best_max, bins


def pack_workers(test_nums: list[int] | set[int], dur_ms_map: dict[int, int]) -> list[list[int]]:
    """Return optimal worker assignment minimizing max(worker_load) + 2000 * k."""
    return _pack(test_nums, dur_ms_map)[0]


def pack_cost_ms(test_nums: list[int] | set[int], dur_ms_map: dict[int, int]) -> int:
    """Return the simulated cost of the assignment produced by pack_workers."""
    return _pack(test_nums, dur_ms_map)[1]


def _pack(test_nums: list[int] | set[int], dur_ms_map: dict[int, int]) -> tuple[list[list[int]], int]:
    if not test_nums:
        return [], 0
    items = sorted(((dur_ms_map[n], n) for n in test_nums), reverse=True)
    total = sum(d for d, _ in items)
    max_d = items[0][0]

    # k = 1 baseline
    best_cost = total + WORKER_COST_MS
    best_bins: list[list[int]] = [[n for _, n in items]]

    max_k = min(MAX_WORKERS, len(items))
    for k in range(2, max_k + 1):
        lb_makespan = max(max_d, (total + k - 1) // k)
        if lb_makespan + WORKER_COST_MS * k >= best_cost:
            continue
        lpt_makespan, lpt_bins = _lpt_pack(items, k)
        if lpt_makespan + WORKER_COST_MS * k < best_cost:
            best_cost = lpt_makespan + WORKER_COST_MS * k
            best_bins = lpt_bins
        # Try exact branch-and-bound if gap remains
        target_ub = min(lpt_makespan, best_cost - WORKER_COST_MS * k)
        if target_ub > lb_makespan and len(items) <= 45:
            exact = _exact_pack(items, k, target_ub, lb_makespan)
            if exact is not None:
                em, ebins = exact
                if em + WORKER_COST_MS * k < best_cost:
                    best_cost = em + WORKER_COST_MS * k
                    best_bins = ebins

    nonempty_bins = [b for b in best_bins if b]
    return nonempty_bins, best_cost - WORKER_COST_MS * (len(best_bins) - len(nonempty_bins))
