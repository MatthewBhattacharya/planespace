"""Greedy AI: picks the single move with the highest immediate score.

Strategy:
  1. For each unclaimed gem, try to build shapes that include that gem as a
     vertex — gem bonuses (+12 each) are guaranteed points worth targeting.
  2. Fill remaining candidate slots with random sampling across all shape sizes.
  3. Score every candidate with the same score_shape() function the game uses,
     and return the highest-scoring valid move.

This is simpler and faster than MCTS but plays sensibly: it always takes gems
when available, always prefers higher-value shape types, and never passes when
a legal move exists.
"""

import random
from typing import Optional

from planespace_rules import (
    GEM_SET, classify, score_shape, validate,
    COLS, ROWS,
)
from planespace_env import State

ALL_POINTS = [(x, y) for x in range(COLS) for y in range(ROWS)]


def _immediate_score(pts: list, claimed_gems: frozenset) -> int:
    cls = classify(pts)
    return score_shape(pts, cls, claimed_gems)['total']


def _find_gem_candidates(state: State, n_per_group: int = 30) -> list:
    """Try shapes that include 1, 2, or 3 unclaimed gems simultaneously.

    Single-gem sampling misses multi-gem hexagons because the chance of
    accidentally landing on 2 other specific gems in 5 random draws is ~0.3%.
    By explicitly anchoring on every pair and triple of gems, we guarantee
    those combinations are explored.
    """
    from itertools import combinations
    unclaimed = [g for g in GEM_SET if g not in state.claimed_gems]
    non_gem_pts = [p for p in ALL_POINTS if p not in GEM_SET]
    cands = []
    seen = set()

    def _try_group(anchors):
        n_anchor = len(anchors)
        for n_verts in range(max(3, n_anchor), 7):  # 3-6 verts
            n_fill = n_verts - n_anchor
            hits = 0
            for _ in range(n_per_group * 3):
                if hits >= n_per_group:
                    break
                fill = random.sample(non_gem_pts, n_fill)
                pts = list(anchors) + fill
                key = tuple(sorted(pts))
                if key in seen:
                    continue
                seen.add(key)
                ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
                if ok:
                    cands.append(pts)
                    hits += 1

    # Singles, pairs, triples
    for r in [1, 2, 3]:
        for group in combinations(unclaimed, r):
            _try_group(group)

    return cands


def _find_random_candidates(state: State, n_attempts: int = 1500) -> list:
    """Random sampling across all shape sizes with equal per-size budget."""
    sizes = [3, 4, 5, 6]
    per_size = n_attempts // len(sizes)
    seen = set()
    cands = []

    for n_verts in sizes:
        for _ in range(per_size):
            pts = random.sample(ALL_POINTS, n_verts)
            key = tuple(sorted(pts))
            if key in seen:
                continue
            seen.add(key)
            ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
            if ok:
                cands.append(pts)

    return cands


class GreedyAI:
    """Always plays the move with the highest immediate score."""

    def __init__(self, n_attempts: int = 1500, n_per_gem: int = 20):
        self.n_attempts = n_attempts
        self.n_per_gem = n_per_gem

    def choose(self, state: State) -> Optional[list]:
        """Return the highest-scoring move, or None if no valid move exists."""
        gem_cands = _find_gem_candidates(state, self.n_per_gem)
        rand_cands = _find_random_candidates(state, self.n_attempts)

        all_cands = gem_cands + rand_cands
        if not all_cands:
            return None

        return max(all_cands, key=lambda pts: _immediate_score(pts, state.claimed_gems))
