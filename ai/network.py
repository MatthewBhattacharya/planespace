"""Policy/value network for Planespace.

Design decisions (see write-up for the full rationale):
- Board is encoded as a small set of 10x10 feature planes, canonicalized to
  the player-to-move's perspective (plane 0 = "my" vertices, plane 1 =
  "opponent" vertices) rather than fixed Blue/Red channels. This halves the
  effective state space the network has to learn over -- it only ever has
  to learn one policy, not a Blue-policy and a mirrored Red-policy.
- A small CNN (not a flat MLP) is used because the legality and value of a
  move is locally and translation-equivariantly determined (a 45-45-90
  triangle scores the same wherever it sits on the board, and "does this
  point's polygon self-intersect" depends on local geometry around the
  point) -- convolution is the natural inductive bias here, the same
  argument AlphaZero makes for Go/Chess board encodings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from planespace_env import ACTION_SIZE, COLS, ROWS, idx_to_point, PLACE_ACTION, PASS_ACTION
from planespace_rules import GEM_SET

PLANES = 6


def encode(state):
    """State -> float32 tensor (PLANES, ROWS, COLS), canonicalized to state.player."""
    x = torch.zeros(PLANES, ROWS, COLS)
    me, opp = state.player, 1 - state.player
    for verts, owner in state.shapes:
        plane = 0 if owner == me else 1
        for (px, py) in verts:
            x[plane, py, px] = 1.0
    for (gx, gy) in GEM_SET:
        x[2, gy, gx] = 1.0
    for (gx, gy) in state.claimed_gems:
        x[3, gy, gx] = 1.0
    n = len(state.sel)
    for i, (px, py) in enumerate(state.sel):
        x[4, py, px] = 1.0
        x[5, py, px] = (i + 1) / 8.0
    return x


class PlanespaceNet(nn.Module):
    def __init__(self, channels=24, fc_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(PLANES, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.fc_shared = nn.Linear(channels * ROWS * COLS, fc_dim)
        self.policy_head = nn.Linear(fc_dim, ACTION_SIZE)
        self.value_head = nn.Linear(fc_dim, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)) + h)   # residual block
        h = F.relu(self.bn3(self.conv3(h)) + h)   # residual block
        h = h.flatten(1)
        h = F.relu(self.fc_shared(h))
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def predict(self, state, mask):
        """Single-state inference for MCTS leaf evaluation. Returns (priors dict, value)."""
        self.eval()
        x = encode(state).unsqueeze(0)
        logits, value = self.forward(x)
        logits = logits.squeeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        logits = logits.masked_fill(~mask_t, float('-inf'))
        probs = F.softmax(logits, dim=0).numpy()
        legal = [i for i, m in enumerate(mask) if m]
        priors = {a: float(probs[a]) for a in legal}
        return priors, float(value.item())
