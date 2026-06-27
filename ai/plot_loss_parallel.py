import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(x, k=10):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        window = x[lo:i + 1]
        window = window[~np.isnan(window)]
        if len(window):
            out[i] = window.mean()
    return out


def main(log_path, out_path, prior_log=None):
    games, total, policy, value, margin = [], [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            games.append(int(row['games_so_far']))
            total.append(float(row['loss']))
            policy.append(float(row['policy_loss']))
            value.append(float(row['value_loss']))
            margin.append(float(row['mean_margin']))

    games = np.array(games)

    # prepend prior run's data if provided (for continuous curve across checkpoint resume)
    prior_games, prior_total, prior_margin = [], [], []
    if prior_log:
        with open(prior_log) as f:
            for row in csv.DictReader(f):
                prior_games.append(int(row['games_so_far']))
                prior_total.append(float(row['loss']))
                prior_margin.append(float(row['mean_margin']))
        offset = prior_games[-1] if prior_games else 0
        games = games + offset

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    if prior_log:
        ax.plot(prior_games, prior_total, alpha=0.25, color='tab:gray')
        ax.plot(prior_games, rolling_mean(prior_total), color='tab:gray', linewidth=1.5, linestyle='--', label='v2 (prior)')
        ax.axvline(prior_games[-1], color='gray', linestyle=':', linewidth=1)
    for y, label, color in [(total, 'total loss', 'tab:red'),
                             (policy, 'policy loss', 'tab:blue'),
                             (value, 'value loss', 'tab:green')]:
        ax.plot(games, y, alpha=0.18, color=color)
        ax.plot(games, rolling_mean(y), label=label, color=color, linewidth=2)
    ax.set_ylabel('loss')
    ax.set_title('Planespace AlphaZero-style training (parallel self-play): loss vs. self-play games')
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    if prior_log:
        ax2.plot(prior_games, prior_margin, alpha=0.2, color='tab:gray')
        ax2.plot(prior_games, rolling_mean(prior_margin), color='tab:gray', linewidth=1.5, linestyle='--', label='v2 (prior)')
        ax2.axvline(prior_games[-1], color='gray', linestyle=':', linewidth=1)
    ax2.plot(games, margin, alpha=0.25, color='tab:purple')
    ax2.plot(games, rolling_mean(margin), color='tab:purple', linewidth=2, label='v3 (continued)')
    ax2.set_xlabel('self-play games completed (cumulative)')
    ax2.set_ylabel('mean final score margin')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print('saved', out_path)


if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'runs/parallel_run/loss_log.csv'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/parallel_run/loss_curve.png'
    prior_log = sys.argv[3] if len(sys.argv) > 3 else None
    main(log_path, out_path, prior_log)
