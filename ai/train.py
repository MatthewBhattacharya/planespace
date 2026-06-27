"""Self-play training loop.

Loss per minibatch (AlphaZero's loss, applied to our continuous score-
differential target instead of discrete win/loss/draw):

    L(theta) = (z - v_theta(s))^2  -  pi^T log p_theta(s)  +  weight_decay * ||theta||^2

  - z        : the n-step return computed during self-play (see mcts.py),
               the same quantity MCTS used as its backed-up value target.
  - v_theta  : the network's value head prediction for that state.
  - pi       : MCTS visit-count distribution over legal actions (the search's
               improved policy), used as the *target* for the raw policy head.
  - p_theta  : softmax(masked policy logits).
  - the L2 term is folded into Adam via weight_decay (AdamW-style decoupled
    decay is not used here; this is the simpler coupled L2 to match the
    original AlphaZero paper's formula).

We log policy and value loss separately, since they tell different stories:
policy loss falling means the raw network is learning to imitate what
search discovers; value loss falling means it's learning to predict who's
ahead just from the position, without doing any search at all.
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from network import PlanespaceNet
from selfplay import play_game


def train_step(net, optimizer, batch, entropy_coef=0.0):
    encs, masks, pis, zs = zip(*batch)
    x = torch.tensor(np.stack(encs), dtype=torch.float32)
    mask = torch.tensor(np.stack(masks), dtype=torch.bool)
    pi = torch.tensor(np.stack(pis), dtype=torch.float32)
    z = torch.tensor(np.array(zs), dtype=torch.float32)

    net.train()
    logits, value = net(x)
    logits = logits.masked_fill(~mask, float('-inf'))
    logp = F.log_softmax(logits, dim=1)
    logp = logp.masked_fill(~mask, 0.0)  # pi is exactly 0 here too; avoids 0 * -inf = nan
    policy_loss = -(pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, z)

    # Entropy bonus: nothing in cross-entropy-vs-MCTS-visits training stops the
    # raw policy head from becoming overconfident once visit counts collapse onto
    # one action -- this explicitly rewards the network for keeping p_theta less
    # peaked, independent of MCTS's own (root-only) Dirichlet exploration noise.
    p = logp.exp()
    entropy = -(p * logp).sum(dim=1).mean()
    loss = policy_loss + value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def main(num_games, n_sims, batch_size, train_steps_per_game, buffer_size,
         lr, weight_decay, entropy_coef, time_budget_s, out_dir, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    net = PlanespaceNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    buffer = []

    log_path = os.path.join(out_dir, 'loss_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['game', 'plies', 'cum_plies', 'p0_score', 'p1_score',
                                 'loss', 'policy_loss', 'value_loss', 'entropy', 'buffer_size', 'elapsed_s'])

    t_start = time.time()
    cum_plies = 0
    g = 0
    while g < num_games:
        if time.time() - t_start > time_budget_s:
            print(f'time budget ({time_budget_s}s) reached at game {g}')
            break
        g += 1
        examples, scores, ply = play_game(net, n_sims=n_sims)
        if any(s > 0 for s in scores):
            buffer.extend(examples)
        if len(buffer) > buffer_size:
            buffer = buffer[-buffer_size:]
        cum_plies += ply

        step_losses = []
        if len(buffer) >= batch_size:
            for _ in range(train_steps_per_game):
                idx = np.random.randint(0, len(buffer), size=batch_size)
                batch = [buffer[i] for i in idx]
                step_losses.append(train_step(net, optimizer, batch, entropy_coef=entropy_coef))

        if step_losses:
            loss, ploss, vloss, ent = np.mean(step_losses, axis=0)
        else:
            loss, ploss, vloss, ent = float('nan'), float('nan'), float('nan'), float('nan')

        elapsed = time.time() - t_start
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([g, ply, cum_plies, scores[0], scores[1],
                                     loss, ploss, vloss, ent, len(buffer), round(elapsed, 1)])

        if g % 10 == 0 or g == 1:
            print(f'game {g}/{num_games}  plies={ply:3d}  scores={scores}  '
                  f'loss={loss:.4f} (policy {ploss:.4f} + value {vloss:.4f})  entropy={ent:.3f}  '
                  f'buffer={len(buffer)}  elapsed={elapsed:.1f}s', flush=True)

        if g % 50 == 0:
            torch.save(net.state_dict(), os.path.join(out_dir, f'net_game{g}.pt'))

    torch.save(net.state_dict(), os.path.join(out_dir, 'final_net.pt'))
    print('done. games:', g, 'log at', log_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-games', type=int, default=300)
    p.add_argument('--n-sims', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--train-steps-per-game', type=int, default=4)
    p.add_argument('--buffer-size', type=int, default=20000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--entropy-coef', type=float, default=0.02)
    p.add_argument('--time-budget', type=int, default=480)
    p.add_argument('--out-dir', type=str, default='runs/run1')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args.num_games, args.n_sims, args.batch_size, args.train_steps_per_game,
         args.buffer_size, args.lr, args.weight_decay, args.entropy_coef,
         args.time_budget, args.out_dir, args.seed)
