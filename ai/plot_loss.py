import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(x, k=15):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        window = x[lo:i + 1]
        window = window[~np.isnan(window)]
        if len(window):
            out[i] = window.mean()
    return out


def main(log_path, out_path):
    games, total, policy, value, p0, p1 = [], [], [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            games.append(int(row['game']))
            total.append(float(row['loss']))
            policy.append(float(row['policy_loss']))
            value.append(float(row['value_loss']))
            p0.append(int(row['p0_score']))
            p1.append(int(row['p1_score']))

    games = np.array(games)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    for y, label, color in [(total, 'total loss', 'tab:red'),
                             (policy, 'policy loss', 'tab:blue'),
                             (value, 'value loss', 'tab:green')]:
        ax.plot(games, y, alpha=0.18, color=color)
        ax.plot(games, rolling_mean(y), label=label, color=color, linewidth=2)
    ax.set_ylabel('loss')
    ax.set_title('Planespace AlphaZero-style training: loss vs. self-play games')
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    margin = np.abs(np.array(p0) - np.array(p1))
    ax2.plot(games, margin, alpha=0.25, color='tab:purple')
    ax2.plot(games, rolling_mean(margin), color='tab:purple', linewidth=2, label='|score margin| (rolling avg)')
    ax2.set_xlabel('self-play game #')
    ax2.set_ylabel('final score margin')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print('saved', out_path)


if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'runs/main_run/loss_log.csv'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/main_run/loss_curve.png'
    main(log_path, out_path)
