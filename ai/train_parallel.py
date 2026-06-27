"""Parallel self-play training loop.

Same algorithm as train.py (same loss, same MCTS, same env) -- the only
change is *where self-play games are generated*. train.py generates them
one at a time in the main process; this generates `games_per_round` games
per round spread across a persistent pool of worker processes (one per
core), since self-play dominated wall-clock time and was previously using
only one of the machine's 8 cores.

Round structure (synchronous, like the standard small-scale AlphaZero
recipe -- not a fully async actor/learner pipeline, which would need a
shared queue and continuous weight broadcasting for not much extra benefit
at this scale):

  1. snapshot current network weights to disk (current.pt)
  2. fan out games_per_round games across `workers` processes, each loading
     that snapshot fresh
  3. pool all returned examples into the replay buffer
  4. take a few gradient steps on the updated buffer
  5. overwrite current.pt with the new weights, go to 1
"""

import argparse
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from network import PlanespaceNet
from train import train_step
from selfplay_worker import run_worker


def main(num_games, n_sims, workers, games_per_round, batch_size,
         train_steps_per_round, buffer_size, lr, weight_decay, entropy_coef,
         time_budget_s, out_dir, seed, init_checkpoint=None, rollout_blend=0.0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    net = PlanespaceNet()
    if init_checkpoint:
        net.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print(f'resumed from {init_checkpoint}')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    buffer = []

    ckpt_path = os.path.join(out_dir, 'current.pt')
    torch.save(net.state_dict(), ckpt_path)

    log_path = os.path.join(out_dir, 'loss_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['round', 'games_so_far', 'cum_plies', 'mean_margin',
                                 'loss', 'policy_loss', 'value_loss', 'entropy', 'buffer_size', 'elapsed_s'])

    t_start = time.time()
    games_done = 0
    round_idx = 0
    cum_plies = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while games_done < num_games:
            if time.time() - t_start > time_budget_s:
                print(f'time budget ({time_budget_s}s) reached at {games_done} games')
                break
            round_idx += 1
            remaining = num_games - games_done
            this_round_games = min(games_per_round, remaining)
            per_worker = max(1, this_round_games // workers)

            base_seed = seed * 1_000_000 + round_idx * workers
            tasks = [(ckpt_path, per_worker, n_sims, base_seed + w, rollout_blend) for w in range(workers)]

            results = list(pool.map(run_worker, tasks))

            round_examples = []
            round_stats = []
            for examples, stats in results:
                round_examples.extend(examples)
                round_stats.extend(stats)

            buffer.extend(round_examples)
            if len(buffer) > buffer_size:
                buffer = buffer[-buffer_size:]
            games_done += len(round_stats)
            cum_plies += sum(s[1] for s in round_stats)
            margins = [abs(s[0][0] - s[0][1]) for s in round_stats]
            mean_margin = float(np.mean(margins)) if margins else float('nan')

            step_losses = []
            if len(buffer) >= batch_size:
                for _ in range(train_steps_per_round):
                    idx = np.random.randint(0, len(buffer), size=batch_size)
                    batch = [buffer[i] for i in idx]
                    step_losses.append(train_step(net, optimizer, batch, entropy_coef=entropy_coef))
            if step_losses:
                loss, ploss, vloss, ent = np.mean(step_losses, axis=0)
            else:
                loss, ploss, vloss, ent = float('nan'), float('nan'), float('nan'), float('nan')

            torch.save(net.state_dict(), ckpt_path)

            elapsed = time.time() - t_start
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([round_idx, games_done, cum_plies, round(mean_margin, 2),
                                         loss, ploss, vloss, ent, len(buffer), round(elapsed, 1)])
            print(f'round {round_idx:4d}  games={games_done:6d}/{num_games}  '
                  f'mean_margin={mean_margin:5.1f}  loss={loss:.4f} (policy {ploss:.4f} + value {vloss:.4f})  '
                  f'entropy={ent:.3f}  buffer={len(buffer):6d}  elapsed={elapsed:.1f}s', flush=True)

            if round_idx % 20 == 0:
                torch.save(net.state_dict(), os.path.join(out_dir, f'net_round{round_idx}.pt'))

    torch.save(net.state_dict(), os.path.join(out_dir, 'final_net.pt'))
    print('done. games:', games_done, 'rounds:', round_idx, 'log at', log_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-games', type=int, default=20000)
    p.add_argument('--n-sims', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--games-per-round', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--train-steps-per-round', type=int, default=8)
    p.add_argument('--buffer-size', type=int, default=60000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--entropy-coef', type=float, default=0.02)
    p.add_argument('--time-budget', type=int, default=480)
    p.add_argument('--out-dir', type=str, default='runs/parallel_run')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init-checkpoint', type=str, default=None)
    p.add_argument('--rollout-blend', type=float, default=0.5,
                   help='weight for fast rollout value vs network value at leaf nodes (0=net only, 1=rollout only)')
    args = p.parse_args()
    main(args.num_games, args.n_sims, args.workers, args.games_per_round, args.batch_size,
         args.train_steps_per_round, args.buffer_size, args.lr, args.weight_decay, args.entropy_coef,
         args.time_budget, args.out_dir, args.seed, args.init_checkpoint, args.rollout_blend)
