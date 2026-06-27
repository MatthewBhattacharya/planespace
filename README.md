# Planespace

A two-player geometry strategy game where players place polygons on a 10×10 dot grid and score points based on area, shape classification, and bonus "gem" vertices.

## How to play

**Browser (2-player local)**  
Open `index.html` directly in a browser.

**vs AI**  
```bash
cd ai
python3 server.py          # starts on http://localhost:8765
# open http://localhost:8765 in browser, check "vs AI (Red = MacroMCTS)"
```

**Replay viewer**  
```bash
cd ai
python3 export_game.py     # writes ~/geometry-game/replay.json
# open replay.html in browser
```

## Rules summary

- Players alternate placing **simple polygons** (3–8 vertices, area ≤ 9) on integer grid points.
- Polygons may not overlap or share an edge with an opponent's shape.
- Score = base (from area) + shape bonus (triangle/quad/pentagon/hexagon tiers, with sub-type bonuses for right triangles, squares, etc.) + 10 pts per "gem" vertex first claimed.
- Game ends when both players pass consecutively.

See `RULESET.md` for full scoring tables.

---

## AI design

### The game as a search problem

The natural representation for MCTS is the **macro-action MDP**: each state is a complete board position (polygons placed), and each action is a full polygon placement. This gives ~15–25 candidate actions per state with immediate, dense rewards.

### Why not micro-actions?

An earlier attempt decomposed each turn into individual vertex picks (3–8 picks → PLACE). While this keeps the action space fixed at 102 (100 grid points + PLACE + PASS), it creates very deep trees: a single placement requires 4–9 steps. With typical MCTS simulation budgets (24–200 sims) and ~100 legal actions at each vertex-pick step, the search never reaches PLACE nodes and reward never propagates back to the root. Training on the resulting uniform visit distributions produces a degenerate fixed point where the policy network stays uniformly random throughout.

### MacroMCTS (`ai/macro_mcts.py`)

Pure UCT operating on macro-actions. No network required.

**Candidate generation** (`generate_candidates`): at each state, sample random polygons (triangles through hexagons, equal budget per size) and validate them. Return up to `n_cands=20` valid placements sorted by immediate score descending. Action set = `[PASS] + candidates`.

**Leaf evaluation** (`fast_rollout`): complete the game from the leaf by placing random valid triangles (triangles are always simple — skips the `is_simple` check). Returns `tanh(score_diff / 60)`.

**UCT backup**: asymmetric 2-player backup. Value is tracked from each node's own player's perspective; the sign flips when traversing edges where the player changes.

**Performance**: ~0.6–1 s/move at `n_sims=100` on a laptop, producing competitive games immediately.

### Macro value network (`ai/macro_value_train.py`)

Trains the value head of `PlanespaceNet` on game positions from MacroMCTS self-play.

**Why value-only?** Macro states are distinguishable — two positions where Blue claimed gem (4,4) with a large square vs a small triangle look different in the 6-plane encoding, and the network can learn which leads to better outcomes. In contrast, individual vertex-pick states (micro-action MDP) are nearly indistinguishable mid-polygon since the only difference is one extra selected vertex.

**Training loop**:
1. Run `games_per_iter` self-play games with MacroMCTS (parallel across 8 workers).
2. Collect `(state_encoding, z)` pairs, where `z = tanh((winner_score − loser_score) / 60)` for every full-turn position in the game.
3. Train value head with MSE loss for `train_steps` gradient steps.
4. Save checkpoint → workers load it next iteration, blending net value with rollouts.

**Network** (`ai/network.py`): 3-layer residual CNN (24 channels) over a 6-plane 10×10 board encoding + FC → value scalar. Planes: my vertices, opponent vertices, gem locations, claimed gems, current selection position, selection order.

**Integration**: `server.py` auto-loads `runs/macro_value/value_net.pt` if present and passes it to `MacroMCTS(value_net=..., value_blend=0.8)`. With the value net, leaf evaluation skips most rollouts (80% net / 20% rollout blend), roughly halving inference time while using a position-aware signal.

### Files

| File | Purpose |
|------|---------|
| `ai/planespace_rules.py` | Python port of the JS scoring/validation rules |
| `ai/planespace_env.py` | Micro-action MDP (kept for reference / potential future use) |
| `ai/macro_mcts.py` | Macro-action UCT agent |
| `ai/rollout.py` | Fast random triangle rollout for leaf evaluation |
| `ai/network.py` | Policy/value CNN |
| `ai/macro_value_train.py` | Self-play + value head training loop |
| `ai/server.py` | HTTP server: serves game files + `/ai_move` endpoint |
| `ai/export_game.py` | Play one game and write `replay.json` |
| `ai/mcts.py` | Original micro-action MCTS (batched virtual-loss PUCT) |
| `ai/train_parallel.py` | Original micro-action AlphaZero training loop |

### Training the value network

```bash
cd ai
python3 macro_value_train.py \
  --num-iters 50 \
  --games-per-iter 64 \
  --workers 8 \
  --n-sims 50
# checkpoint saved to runs/macro_value/value_net.pt
# server.py loads it automatically on next restart
```
