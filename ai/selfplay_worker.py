"""Top-level worker function for multiprocessing self-play.

Must be a plain module-level function (not a closure) so ProcessPoolExecutor
can pickle and ship it to worker processes under the fork start method.
"""
import random

import numpy as np
import torch

from network import PlanespaceNet
from selfplay import play_game


def run_worker(args):
    ckpt_path, num_games, n_sims, seed, rollout_blend = args
    # Each worker is one OS process pinned to ~1 core's worth of work; without
    # this, torch's intra-op thread pool inside *every* worker would each try
    # to grab all 8 cores, oversubscribing the machine instead of using the
    # process-level parallelism we actually want.
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = PlanespaceNet()
    net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    net.eval()

    examples = []
    stats = []
    for _ in range(num_games):
        ex, scores, ply = play_game(net, n_sims=n_sims, rollout_blend=rollout_blend)
        if any(s > 0 for s in scores):   # discard games where nothing was placed
            examples.extend(ex)
        stats.append((scores, ply))
    return examples, stats
