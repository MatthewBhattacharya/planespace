"""Fast macro-level rollout for Planespace MCTS leaf evaluation.

The micro-action MDP decomposes each polygon placement into 3-8 vertex picks
followed by PLACE. This makes MCTS search trees very deep: a single placement
is 4-9 steps, so with 24-64 simulations the reward from PLACE actions rarely
propagates back to the root, leaving MCTS without a useful signal.

This module bypasses the micro-action steps entirely during rollout: given a
state, we directly attempt random valid polygon placements (bypassing the
vertex-pick MDP) to get a quick estimate of the game outcome. This rollout
value blends with the network's value head in _expand_node, giving MCTS a
real non-trivial signal from iteration 0 -- before the network has learned
anything -- breaking the otherwise stable uniform fixed point.
"""

import random
import math

from planespace_env import (
    State, outcome_for, is_terminal, VALUE_NORM,
    COLS, ROWS,
)
from planespace_rules import (
    validate, classify, score_shape, GEM_SET,
)

ALL_POINTS = [(x, y) for x in range(COLS) for y in range(ROWS)]


def _apply_placement(state, pts):
    """Directly apply a validated polygon placement, bypassing micro-actions."""
    cls = classify(pts)
    sc = score_shape(pts, cls, state.claimed_gems)
    claimed = set(state.claimed_gems)
    for v in pts:
        if v in GEM_SET:
            claimed.add(v)
    scores = list(state.scores)
    scores[state.player] += sc['total']
    return State(
        shapes=state.shapes + ((tuple(pts), state.player),),
        claimed_gems=frozenset(claimed),
        scores=tuple(scores),
        player=1 - state.player,
        sel=(),
        passes=0,
        done=False,
        plies=state.plies + 1,
    )


def _try_triangle(state, n_attempts=12):
    """Try to place a random triangle.

    Triangles are always simple polygons (no need for is_simple check inside
    validate), so only the overlap and edge-sharing checks run -- much cheaper
    than trying arbitrary n-gons.
    """
    for _ in range(n_attempts):
        pts = random.sample(ALL_POINTS, 3)
        ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
        if ok:
            return _apply_placement(state, pts)
    return None


def _pass(state):
    passes = state.passes + 1
    done = passes >= 2
    return State(
        shapes=state.shapes,
        claimed_gems=state.claimed_gems,
        scores=state.scores,
        player=state.player if done else 1 - state.player,
        sel=(),
        passes=passes,
        done=done,
        plies=state.plies + 1,
    )


def fast_rollout(state, n_turns=6):
    """Estimate game value from state via fast macro-rollout.

    Clears any partial vertex selection, then plays n_turns*2 random triangle
    placements (alternating players) and returns the tanh-squashed score
    differential from the original player's perspective.

    Triangles are used exclusively (vs general n-gons) because they are always
    simple, shaving the is_simple() check out of validate() and roughly halving
    the cost of each placement attempt.

    Args:
        state: current MDP state (may have non-empty sel)
        n_turns: total placement budget (both players combined)
    Returns:
        float in [-1, 1]
    """
    entry_player = state.player

    # Clear any partial polygon selection -- abandoned for rollout purposes.
    if state.sel:
        state = State(
            shapes=state.shapes,
            claimed_gems=state.claimed_gems,
            scores=state.scores,
            player=state.player,
            sel=(),
            passes=state.passes,
            done=state.done,
            plies=state.plies,
        )

    placements = 0
    consecutive_passes = 0

    while not is_terminal(state) and placements < n_turns:
        new_state = _try_triangle(state)
        if new_state is not None:
            state = new_state
            placements += 1
            consecutive_passes = 0
        else:
            state = _pass(state)
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break

    return outcome_for(state, entry_player)
