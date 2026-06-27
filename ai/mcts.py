"""PUCT search adapted for Planespace's micro-action decomposition.

Three design choices worth noting:

1. Asymmetric turn length. In chess/Go every ply flips the player to move,
   so backup just negates the value at every step up the tree. Here most
   plies (vertex picks) do *not* change whose turn it is -- only PLACE and
   PASS do. So during backup we only flip the propagated value when we
   cross an edge where parent.player != child.player.

2. Reward-bearing backup (MuZero-style return). We fold each edge's
   immediate reward into the backed-up return so the search gets a dense
   training signal from the very first simulation, before the value head
   has learned anything useful.

3. Batched leaf evaluation with virtual loss. Running one network forward
   pass per simulation (batch_size=1) wastes most time on Python/PyTorch
   dispatch overhead rather than actual computation. Instead, we walk
   `leaf_batch` simulations to their leaves simultaneously, applying
   virtual loss (W[a] -= 1) at each edge so parallel walks diverge, then
   fire ONE forward pass for all leaves at once.

   N accounting: virtual loss pre-increments N[a] at each walked edge, so
   the backup must NOT add another +1. Whether to pre-increment is tracked
   by whether the walk traversed at least one edge (len(path) > 1).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from planespace_env import legal_actions_mask, step, is_terminal, outcome_for, VALUE_NORM
from network import encode


VIRTUAL_LOSS = -1.0


class Node:
    __slots__ = ('state', 'player', 'terminal', 'expanded', 'children', 'P', 'N', 'W', 'R', '_leaf_value')

    def __init__(self, state):
        self.state = state
        self.player = state.player
        self.terminal = is_terminal(state)
        self.expanded = False
        self.children = {}
        self.P = {}
        self.N = {}
        self.W = {}
        self.R = {}
        self._leaf_value = 0.0


class MCTS:
    # AlphaZero's own convention scales dirichlet_alpha ~= 10 / (avg. legal moves) --
    # they used 0.03 for Go's ~250 moves. We have ~100 legal actions at the root,
    # so 0.3 (our original default) was ~10x more diffuse than that heuristic
    # suggests -- noise spread that thin barely perturbs any single action.
    def __init__(self, net, c_puct=1.5, n_sims=24, leaf_batch=16,
                 dirichlet_eps=0.35, dirichlet_alpha=0.15, rollout_blend=0.0):
        self.net = net
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.leaf_batch = leaf_batch
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.rollout_blend = rollout_blend
        if rollout_blend > 0.0:
            from rollout import fast_rollout
            self._fast_rollout = fast_rollout
        else:
            self._fast_rollout = None

    def run(self, root_state):
        root = Node(root_state)
        if root.terminal:
            return {}, root
        self._expand_node(root, add_noise=True)

        sims_done = 0
        while sims_done < self.n_sims:
            batch_n = min(self.leaf_batch, self.n_sims - sims_done)
            if batch_n == 1:
                self._simulate_single(root)
            else:
                self._simulate_batch(root, batch_n)
            sims_done += batch_n

        total_n = sum(root.N.values()) or 1
        pi = {a: root.N[a] / total_n for a in root.P}
        return pi, root

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _expand_node(self, node, add_noise=False):
        mask = legal_actions_mask(node.state)
        priors, value = self.net.predict(node.state, mask)
        if self._fast_rollout is not None and self.rollout_blend > 0.0:
            rv = self._fast_rollout(node.state)
            value = (1.0 - self.rollout_blend) * value + self.rollout_blend * rv
        if add_noise and len(priors) > 1:
            actions = list(priors.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for a, eps_noise in zip(actions, noise):
                priors[a] = (1 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * float(eps_noise)
        node.P = priors
        node.N = {a: 0 for a in priors}
        node.W = {a: 0.0 for a in priors}
        node.expanded = True
        return value

    def _select_action(self, node):
        total_n = sum(node.N.values())
        sqrt_total = math.sqrt(total_n + 1)
        best_a, best_score = None, -1e18
        for a, p in node.P.items():
            n_a = node.N[a]
            q = 0.0 if n_a == 0 else node.W[a] / n_a
            u = self.c_puct * p * sqrt_total / (1 + n_a)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _backup(self, path, actions, leaf_value, n_pre_incremented=False, vl_cancel=0.0):
        """Back up leaf_value along path.

        n_pre_incremented: if True, N[a] was already incremented during the
          walk (virtual loss); don't increment again.
        vl_cancel: value added to W to cancel any virtual loss; equals
          -VIRTUAL_LOSS when n_pre_incremented is True.
        """
        v = leaf_value
        for i in range(len(path) - 1, 0, -1):
            parent, child, a = path[i - 1], path[i], actions[i - 1]
            if parent.player != child.player:
                v = -v
            v = parent.R.get(a, 0.0) / VALUE_NORM + v
            if not n_pre_incremented:
                parent.N[a] += 1
            parent.W[a] += v + vl_cancel

    # ------------------------------------------------------------------
    # Single-sim path
    # ------------------------------------------------------------------

    def _simulate_single(self, root):
        path = [root]
        actions = []
        node = root
        leaf_value = None
        while True:
            if node.terminal:
                leaf_value = outcome_for(node.state, node.player)
                break
            if not node.expanded:
                leaf_value = self._expand_node(node)
                break
            a = self._select_action(node)
            if a in node.children:
                node = node.children[a]
                path.append(node)
                actions.append(a)
                continue
            next_state, reward = step(path[-1].state, a)
            child = Node(next_state)
            path[-1].children[a] = child
            path[-1].R[a] = reward
            path.append(child)
            actions.append(a)
            node = child
            if node.terminal:
                leaf_value = outcome_for(node.state, node.player)
            else:
                leaf_value = self._expand_node(node)
            break
        self._backup(path, actions, leaf_value)

    # ------------------------------------------------------------------
    # Batched path
    # ------------------------------------------------------------------

    def _walk_to_leaf(self, root):
        """Select a path to a leaf, applying virtual loss at each edge so
        parallel walks in the same batch go to different leaves. N is
        pre-incremented here; backup must NOT add another +1."""
        path = [root]
        actions = []
        node = root
        while True:
            if node.terminal or not node.expanded:
                break
            a = self._select_action(node)
            node.N[a] += 1          # pre-increment for virtual loss
            node.W[a] += VIRTUAL_LOSS
            if a in node.children:
                node = node.children[a]
                path.append(node)
                actions.append(a)
                continue
            next_state, reward = step(path[-1].state, a)
            child = Node(next_state)
            path[-1].children[a] = child
            path[-1].R[a] = reward
            path.append(child)
            actions.append(a)
            node = child
            break
        return path, actions, node

    def _simulate_batch(self, root, batch_n):
        # Walk batch_n paths (virtual loss keeps them diverging)
        walks = [self._walk_to_leaf(root) for _ in range(batch_n)]

        # Batch-expand all unexpanded non-terminal leaves in one forward pass
        to_expand = [(i, n) for i, (_, _, n) in enumerate(walks)
                     if not n.terminal and not n.expanded]

        if to_expand:
            nodes = [n for _, n in to_expand]
            masks = [legal_actions_mask(n.state) for n in nodes]
            xs = torch.stack([encode(n.state) for n in nodes])
            self.net.eval()
            with torch.no_grad():
                logits_batch, values_batch = self.net(xs)
            for k, node in enumerate(nodes):
                mask_t = torch.tensor(masks[k], dtype=torch.bool)
                logk = logits_batch[k].masked_fill(~mask_t, float('-inf'))
                probs = F.softmax(logk, dim=0).numpy()
                legal = [j for j, m in enumerate(masks[k]) if m]
                node.P = {a: float(probs[a]) for a in legal}
                node.N = {a: 0 for a in node.P}
                node.W = {a: 0.0 for a in node.P}
                node.expanded = True
                net_val = float(values_batch[k].item())
                if self._fast_rollout is not None and self.rollout_blend > 0.0:
                    rv = self._fast_rollout(node.state)
                    net_val = (1.0 - self.rollout_blend) * net_val + self.rollout_blend * rv
                node._leaf_value = net_val

        # Backup all paths. len(path)>1 means the walk traversed at least one
        # edge and applied virtual loss; backup must cancel it and not re-increment N.
        for path, actions, node in walks:
            lv = (outcome_for(node.state, node.player) if node.terminal
                  else getattr(node, '_leaf_value', 0.0))
            walked = len(path) > 1
            self._backup(path, actions, lv,
                         n_pre_incremented=walked,
                         vl_cancel=(-VIRTUAL_LOSS if walked else 0.0))
