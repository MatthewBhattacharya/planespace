"""Micro-action MDP wrapper around planespace_rules.

Design decision: a human "turn" (click 3-8 dots, then Place) is decomposed
into a sequence of single-point picks, terminated by an explicit PLACE or
PASS action. This keeps the action space fixed and small (100 grid points +
PLACE + PASS = 102) regardless of how complex a polygon is, at the cost of
turns no longer being single plies -- most plies keep the same player to
move. See mcts.py for how tree search backup is adapted for that.
"""

from dataclasses import dataclass, field
from typing import Tuple

from planespace_rules import (
    COLS, ROWS, MAX_VERTS, MAX_AREA, GEM_PTS, GEM_SET,
    classify, score_shape, validate, is_simple, shoelace,
)

ACTION_SIZE = COLS * ROWS + 2  # 100 point-picks + PLACE + PASS
PLACE_ACTION = COLS * ROWS
PASS_ACTION = COLS * ROWS + 1
MAX_PLIES = 400  # safety cutoff; real games end far sooner via the two-pass rule
VALUE_NORM = 60.0  # squashes raw point totals into a comparable [-1,1] range


def idx_to_point(idx):
    return (idx % COLS, idx // COLS)


@dataclass(frozen=True)
class State:
    shapes: Tuple[Tuple[Tuple[Tuple[int, int], ...], int], ...] = ()  # ((verts...), player)
    claimed_gems: frozenset = frozenset()
    scores: Tuple[int, int] = (0, 0)
    player: int = 0
    sel: Tuple[Tuple[int, int], ...] = ()
    passes: int = 0
    done: bool = False
    plies: int = 0


def reset():
    return State()


def legal_actions_mask(state):
    """Returns a length-ACTION_SIZE list of bools.

    Vertex picks are filtered so that every legal pick keeps the running
    selection as a valid simple polygon (no self-intersections, area within
    cap). This means PLACE is always a legal option once 3+ vertices are
    selected, eliminating the failure mode where the agent accumulates
    vertices into an unplaceable configuration and is forced to PASS.

    The remaining validation at PLACE time (overlap with existing shapes,
    edge-sharing with opponent) still runs via validate(); these can't be
    cheaply checked during individual vertex picks.
    """
    mask = [False] * ACTION_SIZE
    sel_set = set(state.sel)
    n_sel = len(state.sel)

    if n_sel < MAX_VERTS:
        for idx in range(COLS * ROWS):
            p = idx_to_point(idx)
            if p in sel_set:
                continue
            if n_sel >= 2:
                candidate = list(state.sel) + [p]
                area = shoelace(candidate)
                if area == 0 or area > MAX_AREA:
                    continue
                # For 4+ vertex polygons, also check the closed polygon is simple.
                # (3-vertex case is always simple; shoelace > 0 already rules out
                # degenerate collinear triangles above.)
                if n_sel >= 3 and not is_simple(candidate):
                    continue
            mask[idx] = True

    if n_sel >= 3:
        ok, _ = validate(list(state.sel), state.shapes, state.claimed_gems, state.player)
        if ok:
            mask[PLACE_ACTION] = True
    mask[PASS_ACTION] = True
    return mask


def step(state, action):
    """Returns (next_state, reward_for_acting_player)."""
    if action == PASS_ACTION:
        passes = state.passes + 1
        done = passes >= 2
        nxt = State(
            shapes=state.shapes, claimed_gems=state.claimed_gems, scores=state.scores,
            player=state.player if done else 1 - state.player,
            sel=(), passes=passes, done=done, plies=state.plies + 1,
        )
        return nxt, 0.0

    if action == PLACE_ACTION:
        pts = list(state.sel)
        cls = classify(pts)
        sc = score_shape(pts, cls, state.claimed_gems)
        claimed = set(state.claimed_gems)
        for v in pts:
            if v in GEM_SET:
                claimed.add(v)
        scores = list(state.scores)
        scores[state.player] += sc['total']
        nxt = State(
            shapes=state.shapes + ((tuple(pts), state.player),),
            claimed_gems=frozenset(claimed),
            scores=tuple(scores),
            player=1 - state.player,
            sel=(), passes=0, done=False, plies=state.plies + 1,
        )
        return nxt, float(sc['total'])

    # pick a grid point
    p = idx_to_point(action)
    nxt = State(
        shapes=state.shapes, claimed_gems=state.claimed_gems, scores=state.scores,
        player=state.player, sel=state.sel + (p,), passes=state.passes,
        done=False, plies=state.plies + 1,
    )
    return nxt, 0.0


def is_terminal(state):
    return state.done or state.plies >= MAX_PLIES


def outcome_for(state, player, norm=VALUE_NORM):
    """tanh-squashed final score differential from `player`'s perspective."""
    diff = state.scores[player] - state.scores[1 - player]
    import math
    return math.tanh(diff / norm)
