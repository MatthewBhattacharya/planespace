"""Macro-action value training for Planespace.

Unlike the micro-action AlphaZero approach (which failed because ~100 legal
vertex picks per state means MCTS never differentiates them at any reasonable
simulation budget), this pipeline trains at the MACRO level:

  - Each state is a full-turn position: which polygons have been placed.
  - Each training example: (state_encoding, z) where z = tanh-squashed score
    differential at game end from state.player's perspective.
  - Network: same PlanespaceNet, but only the VALUE head is trained.
  - Use in MacroMCTS: replace rollouts with net.value(state), cutting ~80% of
    the time per move while providing a position-aware value signal.

Why macro states are learnable (where micro states weren't):
  - Macro state space has ~10^15 positions but in practice most games visit
    a tiny slice; the network generalises from visual features.
  - Each macro state uniquely represents what shapes have been placed; two
    positions where Blue placed a large square near gem (4,4) look very
    different from one where they placed a tiny triangle elsewhere.
  - Rollouts from macro states already give a strong learning signal;
    bootstrapping from these is standard AlphaZero practice.

Training loop (self-play → value learning → repeat):
  1. Run N self-play games using MacroMCTS (+ optional value net boosting).
  2. Collect (state, z) pairs for every full-turn state in each game.
  3. Train value head for K gradient steps on the accumulated buffer.
  4. Save checkpoint.  Repeat for num_iters iterations.

Usage:
  python3 macro_value_train.py                    # default: 10 iters × 50 games
  python3 macro_value_train.py --num-iters 20 --games-per-iter 100
"""

import argparse
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from planespace_env import (
    State, reset, is_terminal, outcome_for, VALUE_NORM,
)
from network import PlanespaceNet, encode
from macro_mcts import MacroMCTS, _apply_placement, _apply_pass, generate_candidates


# ---------------------------------------------------------------------------
# Self-play data generation
# ---------------------------------------------------------------------------

def collect_game(mcts: MacroMCTS):
    """Play one game with MacroMCTS and return (examples, final_scores)."""
    state = reset()
    trajectory = []

    while not is_terminal(state):
        trajectory.append((state, state.player))
        pts = mcts.choose(state, temperature=0.3)
        if pts is None:
            state = _apply_pass(state)
        else:
            state = _apply_placement(state, pts)

    final_scores = state.scores
    examples = []
    for s, player in trajectory:
        diff = final_scores[player] - final_scores[1 - player]
        z = float(np.tanh(diff / VALUE_NORM))
        examples.append((s, z))
    return examples, final_scores


def _worker_collect(args):
    """Top-level function (picklable) for ProcessPoolExecutor."""
    n_games, n_sims, n_cands, seed, ckpt_path, value_blend = args
    import torch as _torch
    _torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)

    value_net = None
    if ckpt_path and os.path.exists(ckpt_path):
        from network import PlanespaceNet as _Net
        value_net = _Net()
        value_net.load_state_dict(_torch.load(ckpt_path, map_location='cpu'))
        value_net.eval()

    mcts = MacroMCTS(n_sims=n_sims, n_cands=n_cands,
                     value_net=value_net, value_blend=value_blend if value_net else 0.0)

    all_examples = []
    all_scores = []
    for _ in range(n_games):
        ex, scores = collect_game(mcts)
        all_examples.extend(ex)
        all_scores.append(scores)
    return all_examples, all_scores


# ---------------------------------------------------------------------------
# Value training step
# ---------------------------------------------------------------------------

def value_train_step(net, optimizer, batch):
    """One gradient step on value loss only (policy head untouched)."""
    states, zs = zip(*batch)
    x = torch.stack([encode(s) for s in states])
    z = torch.tensor(np.array(zs), dtype=torch.float32)

    net.train()
    _, value = net(x)
    loss = F.mse_loss(value, z)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(num_iters, games_per_iter, n_sims, n_cands, workers,
         batch_size, train_steps, buffer_size, lr, weight_decay, out_dir, seed, value_blend):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    net = PlanespaceNet()
    ckpt_path = os.path.join(out_dir, 'value_net.pt')
    if os.path.exists(ckpt_path):
        net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f'resumed from {ckpt_path}')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    log_path = os.path.join(out_dir, 'loss_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['iter', 'games', 'mean_margin', 'value_loss', 'buffer_size', 'elapsed_s'])

    buffer = []
    t_start = time.time()
    games_done = 0
    best_loss = float('inf')

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for it in range(1, num_iters + 1):
            # Distribute games_per_iter games across workers
            per_worker = max(1, games_per_iter // workers)
            base_seed = seed * 1_000_000 + it * workers
            tasks = [(per_worker, n_sims, n_cands, base_seed + w,
                      ckpt_path if os.path.exists(ckpt_path) else None, value_blend)
                     for w in range(workers)]
            results = list(pool.map(_worker_collect, tasks))

            iter_margins = []
            for examples, scores_list in results:
                buffer.extend(examples)
                for sc in scores_list:
                    iter_margins.append(abs(sc[0] - sc[1]))
            games_done += sum(len(r[1]) for r in results)

            if len(buffer) > buffer_size:
                buffer = buffer[-buffer_size:]

            # Training (in main process, on accumulated buffer)
            step_losses = []
            if len(buffer) >= batch_size:
                for _ in range(train_steps):
                    idx = np.random.randint(0, len(buffer), size=batch_size)
                    batch = [buffer[i] for i in idx]
                    step_losses.append(value_train_step(net, optimizer, batch))

            vloss = float(np.mean(step_losses)) if step_losses else float('nan')
            mean_margin = float(np.mean(iter_margins)) if iter_margins else float('nan')
            elapsed = time.time() - t_start

            torch.save(net.state_dict(), ckpt_path)
            if it % 5 == 0:
                torch.save(net.state_dict(), os.path.join(out_dir, f'value_net_iter{it}.pt'))
            if not np.isnan(vloss) and vloss < best_loss:
                best_loss = vloss
                torch.save(net.state_dict(), os.path.join(out_dir, 'value_net_best.pt'))

            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([it, games_done, round(mean_margin, 2),
                                         round(vloss, 5), len(buffer), round(elapsed, 1)])

            print(f'iter {it:3d}/{num_iters}  games={games_done}  '
                  f'margin={mean_margin:.1f}  value_loss={vloss:.5f}  '
                  f'buffer={len(buffer)}  elapsed={elapsed:.1f}s', flush=True)

            if it % 5 == 0:
                mcts_sample = MacroMCTS(n_sims=n_sims, n_cands=n_cands)
                _print_sample_game(mcts_sample, seed=it)

    print(f'\ndone. value_net saved to {ckpt_path}')


def _print_sample_game(mcts, seed=0):
    random.seed(seed)
    state = reset()
    move_n = 0
    from planespace_rules import classify, score_shape
    print(f'  sample game (seed={seed}):')
    while not is_terminal(state) and move_n < 20:
        pts = mcts.choose(state, temperature=0.0)
        pname = 'Blue' if state.player == 0 else 'Red'
        if pts:
            cls = classify(pts)
            sc = score_shape(pts, cls, state.claimed_gems)
            print(f'    {pname}: {" + ".join(cls["names"])} area={sc["area"]} +{sc["total"]}')
            state = _apply_placement(state, pts)
        else:
            print(f'    {pname}: PASS')
            state = _apply_pass(state)
        move_n += 1
    print(f'  final: Blue={state.scores[0]}  Red={state.scores[1]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-iters', type=int, default=10)
    p.add_argument('--games-per-iter', type=int, default=64)
    p.add_argument('--n-sims', type=int, default=50)
    p.add_argument('--n-cands', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--train-steps', type=int, default=200)
    p.add_argument('--buffer-size', type=int, default=30000)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--out-dir', type=str, default='runs/macro_value')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--value-blend', type=float, default=1.0)
    args = p.parse_args()
    main(args.num_iters, args.games_per_iter, args.n_sims, args.n_cands,
         args.workers, args.batch_size, args.train_steps, args.buffer_size,
         args.lr, args.weight_decay, args.out_dir, args.seed, args.value_blend)
