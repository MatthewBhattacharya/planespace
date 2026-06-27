"""Play one MacroMCTS-guided game and export a JSON replay.

Usage:
    python3 export_game.py [output_path] [seed] [n_sims]

Defaults: ~/geometry-game/replay.json, seed=0, n_sims=200
"""

import json
import os
import sys

from planespace_env import reset, is_terminal
from planespace_rules import classify, score_shape, GEM_PTS, GEM_SET
from macro_mcts import MacroMCTS, _apply_placement, _apply_pass, generate_candidates

DEFAULT_OUT = os.path.expanduser('~/geometry-game/replay.json')


def play_and_export(out_path: str, seed: int = 0, n_sims: int = 200, n_cands: int = 20):
    import random
    random.seed(seed)

    state = reset()
    mcts = MacroMCTS(n_sims=n_sims, n_cands=n_cands)
    moves = []

    while not is_terminal(state):
        pts = mcts.choose(state, temperature=0.0)
        if pts:
            cls = classify(pts)
            sc = score_shape(pts, cls, state.claimed_gems)
            claimed_this_move = [list(v) for v in pts if v in GEM_SET and v not in state.claimed_gems]
            moves.append({
                'player': state.player,
                'verts': [list(v) for v in pts],
                'shape_names': cls['names'],
                'area': sc['area'],
                'base': sc['base'],
                'bonus': sc['bonus'],
                'gem_bonus': sc['gem_bonus'],
                'total': sc['total'],
                'claimed_gems': claimed_this_move,
            })
            state = _apply_placement(state, pts)
        else:
            state = _apply_pass(state)

    replay = {
        'grid': {'cols': 10, 'rows': 10},
        'gems': [list(g) for g in GEM_PTS],
        'moves': moves,
        'final_scores': list(state.scores),
    }

    out_path = os.path.expanduser(out_path)
    with open(out_path, 'w') as f:
        json.dump(replay, f, indent=2)

    return out_path, moves, list(state.scores)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_sims = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    path, moves, final_scores = play_and_export(out_path, seed=seed, n_sims=n_sims)
    print(f'wrote {len(moves)} moves → {path}')
    print(f'final scores: Blue={final_scores[0]}  Red={final_scores[1]}')
    winner = 'Blue' if final_scores[0] > final_scores[1] else 'Red' if final_scores[1] > final_scores[0] else 'Tie'
    print(f'winner: {winner}')
    if moves:
        m = moves[0]
        print(f'move 1: {" + ".join(m["shape_names"])} by {"Blue" if m["player"]==0 else "Red"}'
              f', verts={m["verts"]}, total={m["total"]}')


if __name__ == '__main__':
    main()
