"""Generate one self-play game's worth of training examples."""

import numpy as np

from planespace_env import reset, step, is_terminal, ACTION_SIZE, legal_actions_mask, VALUE_NORM, PLACE_ACTION
from network import encode
from mcts import MCTS


def sample_action(pi, temperature):
    actions = list(pi.keys())
    if temperature <= 1e-3:
        return max(actions, key=lambda a: pi[a])
    probs = np.array([pi[a] ** (1.0 / temperature) for a in actions], dtype=np.float64)
    probs /= probs.sum()
    return actions[np.random.choice(len(actions), p=probs)]


def play_game(net, n_sims=24, c_puct=1.5, temp_plies=12, rollout_blend=0.0):
    """Self-play one full game. Returns (examples, final_scores).

    examples: list of (encoding[6,10,10], mask[102], pi[102], player) before
    the outcome is known -- the caller fills in z once the game is finished.
    """
    mcts = MCTS(net, c_puct=c_puct, n_sims=n_sims, rollout_blend=rollout_blend)
    state = reset()
    trajectory = []
    ply = 0
    while not is_terminal(state):
        pi_dict, root = mcts.run(state)
        if not pi_dict:  # terminal reached inside run (shouldn't happen given the while guard)
            break
        pi_full = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a, p in pi_dict.items():
            pi_full[a] = p
        mask = legal_actions_mask(state)
        trajectory.append((encode(state).numpy(), np.array(mask, dtype=bool), pi_full, state.player))

        # Near-greedy when PLACE is available: high temperature during vertex
        # picking (exploration is fine) but decisive at the commit decision
        # (high temp here causes the agent to overshoot into invalid territory).
        temperature = 0.1 if mask[PLACE_ACTION] else (1.0 if ply < temp_plies else 0.2)
        action = sample_action(pi_dict, temperature)
        state, _ = step(state, action)
        ply += 1

    final_scores = state.scores
    examples = []
    for enc, mask, pi_full, player in trajectory:
        diff = final_scores[player] - final_scores[1 - player]
        z = float(np.tanh(diff / VALUE_NORM))
        examples.append((enc, mask, pi_full, z))
    return examples, final_scores, ply
