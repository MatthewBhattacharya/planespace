"""Greedy AI: picks the single move with the highest immediate score.

Strategy:
  1. Gem combos: explicitly anchor on every pair/triple of unclaimed gems so
     multi-gem hexagons are reliably found.
  2. Systematic rectangles: enumerate every axis-aligned rectangle exhaustively
     (only ~2000 checks) — guarantees high-bonus quads are never missed.
  3. Skewed random search: hexagons and pentagons get 4× more random attempts
     than triangles because they are proportionally harder to place; triangles
     are capped so they can't crowd out larger shapes in the candidate pool.
  4. Emergency fallback: if all of the above find nothing (very late game),
     run a high-budget triangle search before conceding a pass.
"""

import random
from itertools import combinations
from typing import Optional

from planespace_rules import (
    GEM_SET, MIN_AREA, MAX_AREA, COLS, ROWS,
    classify, score_shape, validate,
)
from planespace_env import State

ALL_POINTS = [(x, y) for x in range(COLS) for y in range(ROWS)]
NON_GEM_PTS = [p for p in ALL_POINTS if p not in GEM_SET]


def _score(pts: list, claimed_gems: frozenset) -> int:
    return score_shape(pts, classify(pts), claimed_gems)['total']


# ---------------------------------------------------------------------------
# 1. Gem-combo search
# ---------------------------------------------------------------------------

def _find_gem_candidates(state: State, n_per_group: int = 25) -> list:
    """Try every combination of 1, 2, 3 unclaimed gems as anchor vertices."""
    unclaimed = [g for g in GEM_SET if g not in state.claimed_gems]
    cands = []
    seen = set()

    def _try_group(anchors):
        n_anchor = len(anchors)
        for n_verts in range(max(3, n_anchor), 7):
            n_fill = n_verts - n_anchor
            hits = 0
            for _ in range(n_per_group * 4):
                if hits >= n_per_group:
                    break
                fill = random.sample(NON_GEM_PTS, n_fill)
                pts = list(anchors) + fill
                key = tuple(sorted(pts))
                if key in seen:
                    continue
                seen.add(key)
                ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
                if ok:
                    cands.append(pts)
                    hits += 1

    for r in [1, 2, 3]:
        for group in combinations(unclaimed, r):
            _try_group(group)

    return cands


# ---------------------------------------------------------------------------
# 2. Systematic rectangle search
# ---------------------------------------------------------------------------

def _find_rectangles(state: State) -> list:
    """Enumerate every axis-aligned rectangle exhaustively (~2000 checks)."""
    cands = []
    for x1 in range(COLS - 1):
        for x2 in range(x1 + 1, COLS):
            for y1 in range(ROWS - 1):
                for y2 in range(y1 + 1, ROWS):
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_AREA or area > MAX_AREA:
                        continue
                    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
                    if ok:
                        cands.append(pts)
    return cands


# ---------------------------------------------------------------------------
# 3. Skewed random search
# ---------------------------------------------------------------------------

def _find_random_candidates(state: State) -> list:
    """Random sampling with skewed budgets: hexagons get 4x more attempts,
    and triangles are capped so they can't dominate the candidate pool."""
    # (n_verts, attempt_budget, max_collected)
    configs = [
        (3,  400,  5),   # triangles easy to find — cap them
        (4,  600, 10),
        (5,  800, 10),
        (6, 1200, 10),   # hexagons hardest — most attempts
    ]
    seen = set()
    cands = []

    for n_verts, budget, cap in configs:
        hits = 0
        for _ in range(budget):
            if hits >= cap:
                break
            pts = random.sample(ALL_POINTS, n_verts)
            key = tuple(sorted(pts))
            if key in seen:
                continue
            seen.add(key)
            ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
            if ok:
                cands.append(pts)
                hits += 1

    return cands


# ---------------------------------------------------------------------------
# 4. Emergency fallback
# ---------------------------------------------------------------------------

def _emergency_triangle(state: State, budget: int = 15000) -> Optional[list]:
    """Brute-force triangle search. With C(100,3)=161k triangles and even
    5% validity on a crowded board, 15k attempts finds one almost certainly."""
    seen = set()
    for _ in range(budget):
        pts = random.sample(ALL_POINTS, 3)
        key = tuple(sorted(pts))
        if key in seen:
            continue
        seen.add(key)
        ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
        if ok:
            return pts
    return None


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class GreedyAI:
    """Always plays the legal move with the highest immediate score."""

    def choose(self, state: State) -> Optional[list]:
        all_cands = []
        all_cands.extend(_find_gem_candidates(state))
        all_cands.extend(_find_rectangles(state))
        all_cands.extend(_find_random_candidates(state))

        if not all_cands:
            fallback = _emergency_triangle(state)
            if fallback:
                return fallback
            return None  # genuinely no legal moves

        return max(all_cands,
                   key=lambda pts: _score(pts, state.claimed_gems))
