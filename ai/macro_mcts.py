"""Macro-action UCT for Planespace.

The micro-action MDP (pick 3-8 vertices one at a time, then PLACE) makes
AlphaZero-style search trees extremely deep: a single polygon placement
requires 4-9 steps. With typical simulation budgets (24-200 sims) and ~100
legal actions at the first vertex-pick, MCTS never goes deep enough to
encounter placement rewards, so the value signal never propagates back to the
root.

This module sidesteps the problem entirely by operating at the macro level:
each node in the search tree represents a full game state (polygons placed),
and each edge represents a complete polygon placement (or PASS). The action
set at each node is a small fixed-size set of candidate placements generated
by random sampling + scoring.

Because nodes correspond to complete turns, rewards are immediate and dense,
UCT works well with even 100-200 simulations, and the agent is competent
from iteration 0 with no training required.

No network, no learning needed -- this is classical UCT. It produces a
reasonable opponent for the game immediately. Later, a learned policy prior
can be layered on top to guide candidate generation.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from planespace_env import (
    State, outcome_for, is_terminal, VALUE_NORM,
)
from planespace_rules import (
    validate, classify, score_shape, GEM_SET, shoelace,
)

ALL_POINTS = [(x, y) for x in range(10) for y in range(10)]

# ---------------------------------------------------------------------------
# Candidate polygon generation
# ---------------------------------------------------------------------------

def _apply_placement(state: State, pts: list) -> State:
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
        sel=(), passes=0, done=False, plies=state.plies + 1,
    )


def _apply_pass(state: State) -> State:
    passes = state.passes + 1
    done = passes >= 2
    return State(
        shapes=state.shapes, claimed_gems=state.claimed_gems, scores=state.scores,
        player=state.player if done else 1 - state.player,
        sel=(), passes=passes, done=done, plies=state.plies + 1,
    )


def generate_candidates(state: State, n_cands: int = 25, n_attempts: int = 300) -> List[list]:
    """Generate up to n_cands valid polygon placements for state.player.

    Samples shapes across all sizes (triangle through hexagon) with equal
    attempt budget per size, then sorts by immediate score descending so high-
    value placements appear early.  Equal per-size budget ensures the AI
    considers quads, pentagons, and hexagons even though they're harder to
    find valid instances of.
    """
    seen = set()
    cands = []
    shapes = state.shapes
    claimed_gems = state.claimed_gems
    player = state.player

    # Equal budget per size (3-6 vertices)
    sizes = [3, 4, 5, 6]
    per_size = n_attempts // len(sizes)

    for n_verts in sizes:
        count_this_size = 0
        for _ in range(per_size):
            if len(cands) >= n_cands:
                break
            pts = random.sample(ALL_POINTS, n_verts)
            key = tuple(sorted(pts))
            if key in seen:
                continue
            seen.add(key)
            ok, _ = validate(pts, shapes, claimed_gems, player)
            if ok:
                cands.append(pts)
                count_this_size += 1

    if not cands:
        return cands

    def _score(pts):
        cls = classify(pts)
        sc = score_shape(pts, cls, claimed_gems)
        return sc['total']

    cands.sort(key=_score, reverse=True)
    return cands


# ---------------------------------------------------------------------------
# Fast rollout (for terminal value estimation)
# ---------------------------------------------------------------------------

def fast_rollout(state: State, n_turns: int = 6) -> float:
    """Quick random-triangle rollout. Returns value for state.player."""
    entry = state.player
    consecutive_passes = 0
    for _ in range(n_turns):
        if is_terminal(state):
            break
        placed = False
        for _ in range(15):
            pts = random.sample(ALL_POINTS, 3)
            ok, _ = validate(pts, state.shapes, state.claimed_gems, state.player)
            if ok:
                state = _apply_placement(state, pts)
                placed = True
                consecutive_passes = 0
                break
        if not placed:
            state = _apply_pass(state)
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
    return outcome_for(state, entry)


# ---------------------------------------------------------------------------
# UCT node + search
# ---------------------------------------------------------------------------

class MacroNode:
    """UCT node for macro-action search."""
    __slots__ = ('state', 'player', 'children', 'N', 'W', 'actions', 'unexplored')

    def __init__(self, state: State, actions: list):
        self.state = state
        self.player = state.player
        self.actions = actions       # list of pts (or None for PASS)
        self.children: dict = {}     # action_idx -> MacroNode
        self.N: dict = {}            # action_idx -> visit count
        self.W: dict = {}            # action_idx -> sum of values (from this node's player's POV)
        self.unexplored: list = list(range(len(actions)))
        random.shuffle(self.unexplored)

    def uct_select(self, c: float) -> int:
        """Select the action with highest UCB1 score among explored children."""
        total_n = sum(self.N.values()) or 1
        log_n = math.log(total_n)
        best_i, best_score = -1, -1e18
        for i in self.children:
            n_i = self.N.get(i, 1)
            q_i = self.W.get(i, 0.0) / n_i
            u_i = c * math.sqrt(log_n / n_i)
            if (score := q_i + u_i) > best_score:
                best_score, best_i = score, i
        return best_i


class MacroMCTS:
    """UCT search over polygon-placement macro-actions.

    Args:
        n_sims: number of simulations (tree walks) per move
        c_uct: UCB exploration constant
        n_cands: candidate placements to generate per state
        value_net: optional trained PlanespaceNet; its value head replaces
            random rollouts when provided (faster + position-aware evaluation)
        value_blend: weight for value_net vs fast_rollout (1.0 = net only,
            0.0 = rollout only). Ignored when value_net is None.
    """
    def __init__(self, n_sims: int = 200, c_uct: float = 1.5, n_cands: int = 20,
                 value_net=None, value_blend: float = 1.0):
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.n_cands = n_cands
        self.value_net = value_net
        self.value_blend = value_blend

    def _eval(self, state: State) -> float:
        """Evaluate a state from state.player's perspective.

        When value_net is provided, blends its prediction with a fast rollout
        (value_blend controls the mix). Without value_net, falls back to pure
        fast_rollout.
        """
        if self.value_net is not None and self.value_blend > 0.0:
            import torch
            from network import encode
            with torch.no_grad():
                x = encode(state).unsqueeze(0)
                _, v = self.value_net(x)
                net_val = float(v.item())
            if self.value_blend >= 1.0:
                return net_val
            rv = fast_rollout(state)
            return self.value_blend * net_val + (1.0 - self.value_blend) * rv
        return fast_rollout(state)

    def _build_actions(self, state: State) -> list:
        """Generate action list: [None (PASS)] + [candidate placements]."""
        cands = generate_candidates(state, n_cands=self.n_cands)
        return [None] + cands  # index 0 = PASS

    def choose(self, state: State, temperature: float = 0.0) -> Optional[list]:
        """Return the best polygon to place (list of (x,y)), or None to PASS."""
        if is_terminal(state):
            return None

        actions = self._build_actions(state)
        if len(actions) <= 1:  # only PASS available
            return None

        root = MacroNode(state, actions)
        for _ in range(self.n_sims):
            self._simulate(root)

        if not root.N:
            return None

        if temperature <= 1e-3:
            best_i = max(root.N, key=root.N.get)
        else:
            ks = list(root.N.keys())
            counts = [root.N[i] ** (1.0 / temperature) for i in ks]
            total = sum(counts) or 1.0
            probs = [c / total for c in counts]
            best_i = random.choices(ks, weights=probs)[0]

        return actions[best_i]  # None = PASS, list of pts = placement

    def _simulate(self, root: MacroNode):
        """Run one UCT simulation: select → expand → rollout → backup."""
        node = root
        path: List[Tuple[MacroNode, int]] = []

        # --- Selection ---
        while not node.unexplored and node.children and not is_terminal(node.state):
            best_i = node.uct_select(self.c_uct)
            if best_i < 0 or best_i not in node.children:
                break
            path.append((node, best_i))
            node = node.children[best_i]

        # --- Leaf evaluation ---
        if is_terminal(node.state):
            raw = outcome_for(node.state, node.player)
            # value is from node.player's POV; backup will re-orient per node
            value_for_node_player = raw
        elif node.unexplored:
            # Expand one untried action
            i = node.unexplored.pop()
            action = node.actions[i]
            next_state = (_apply_pass(node.state) if action is None
                          else _apply_placement(node.state, action))
            path.append((node, i))

            if is_terminal(next_state):
                # immediate terminal: outcome for next_state's mover
                raw = outcome_for(next_state, next_state.player)
                # flip to node.player's perspective if needed
                value_for_node_player = raw if next_state.player == node.player else -raw
            else:
                next_actions = self._build_actions(next_state)
                child = MacroNode(next_state, next_actions)
                node.children[i] = child
                rv = self._eval(next_state)
                value_for_node_player = rv if next_state.player == node.player else -rv
        else:
            # Fully expanded non-terminal with no children — shouldn't happen, treat as rollout
            raw = self._eval(node.state)
            value_for_node_player = raw

        # --- Backup ---
        # Walk path in reverse; each (parent_node, action_idx) gets updated.
        # value_for_node_player is in node.player's perspective at the leaf.
        # At each step up the path, the "current value perspective" may need to flip.
        v = value_for_node_player
        for parent, action_i in reversed(path):
            # If parent.player differs from node (leaf) player, flip perspective
            if parent.player != node.player:
                v = -v
            parent.N[action_i] = parent.N.get(action_i, 0) + 1
            parent.W[action_i] = parent.W.get(action_i, 0.0) + v
            # As we step further up, player alternates; update 'node' reference
            node = parent


# ---------------------------------------------------------------------------
# Convenience: play a full game, returning moves and scores
# ---------------------------------------------------------------------------

def play_macro_game(n_sims: int = 200, n_cands: int = 20, seed: int = 0,
                    temperature: float = 0.0):
    """Play a full Planespace game with MacroMCTS for both players.

    Returns (moves, final_state) where moves is a list of polygon vertex lists.
    """
    random.seed(seed)
    from planespace_env import reset
    state = reset()
    mcts = MacroMCTS(n_sims=n_sims, n_cands=n_cands)
    moves = []
    while not is_terminal(state):
        pts = mcts.choose(state, temperature=temperature)
        if pts is None:
            state = _apply_pass(state)
        else:
            moves.append({'player': state.player, 'verts': pts})
            state = _apply_placement(state, pts)
    return moves, state
