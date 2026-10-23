# Architecture & training pipeline

## 1. Interface with the tournament referee

The referee runs **PUCT Monte Carlo Tree Search with 100 simulations per move**. For every node it expands, it calls our ONNX model once:

| Tensor | Shape | Meaning |
|---|---|---|
| `observation` (input) | (B, 16, 8, 8) | Board encoded as 16 binary/scalar feature planes |
| `policy` (output) | (B, 4096) | Raw logits indexed by `from_square * 64 + to_square`. The referee masks illegal moves and softmaxes. |
| `value` (output) | (B,) | Expected outcome for the side to move, in [−1, 1] |

The policy steers which branches the search explores (priors). The value replaces random rollouts when a leaf is evaluated.

## 2. Network

```
            observation (B,16,8,8)
                    │
     Stem   Conv3×3(16→c, no bias) · BatchNorm · ReLU
                    │
     Trunk  6 × Residual block:
              y = ReLU(BN(Conv3×3(x)))
              y = BN(Conv3×3(y))
              x = ReLU(x + y)
                    │
        ┌───────────┴─────────────┐
 Policy head                 Value head
 Conv3×3(c→c)·BN·ReLU        Conv1×1(c→8)·BN·ReLU
 Conv1×1(c→32)               Flatten → 512
 (B,32,8,8) → (B,2048)       Linear 512→128 · ReLU
 index_select(GATHER)        Linear 128→1 · tanh
 → (B,4096)                  → (B,)
```

### Move-offset policy encoding
Every legal move in *Colony: One* is one of **32 displacement vectors**:

- 8 directions (N, S, E, W and the 4 diagonals) × step length 1, 2 or 3 → 24
- 8 knight jumps → 8

The head outputs one 8×8 plane per offset. Plane *k* at square *s* scores "move the piece on *s* by offset *k*". A precomputed `GATHER` table (4,096 entries) maps each from→to pair to its (offset, from-square) slot. Pairs that aren't one of the 32 offsets point to a dummy slot; they're always illegal and get masked by the referee.

This is the same idea as AlphaZero's 73-plane chess policy: the movement geometry is built into the network instead of being learned from 4,096 independent outputs.

### Parameter budget

| Component | c = 64 (`b64`) | c = 96 (`a96`, `c96`, `it1`) |
|---|---|---|
| Stem | 9,344 | 14,016 |
| 6 residual blocks | 443,904 | 997,632 |
| Policy head | 39,072 | 86,240 |
| Value head | 66,321 | 66,577 |
| **Total** | **558,641** | **1,164,465** |
| Inference cost (CPU, per game, under load) | ~2.6 s | ~4.8 s |

## 3. Training data: distilling a search teacher

1. **Engine:** a C reimplementation of the full rules (all six pieces, bodyguards, firewall, bare-kernel endgame), checked move-for-move against the official Python engine.
2. **Teacher:** alpha-beta search with capture quiescence, plus a faster transposition-table / iterative-deepening root search that returns exact scores for every move within a margin of the best. `tanh(score / 4)` is well calibrated against real game outcomes.
3. **Self-play generation:** parallel workers play games and store, for every position: the canonical legal-move indices, per-move search scores (from the side-to-move's perspective) and the final winner.
   - `d2` shards: 2 extra moves of lookahead (fast)
   - `pod4` / `pod4t` / `pod5` shards: 4 extra moves of lookahead (deeper, slower; generated on the GPU pod during the contest)
4. **Dataset at deadline:** 734 shards, ~**542,000 positions**.

## 4. Targets and loss

| Head | Target | Loss |
|---|---|---|
| Policy | `softmax(move_scores / 0.5)` over legal moves | Cross-entropy |
| Value | `0.7 · tanh(best_score / 4) + 0.3 · outcome` | Mean squared error |

The temperature 0.5 keeps the policy target sharp around the best moves. Mixing 30% real game outcome into the value target corrects the teacher's static-evaluation bias.

**Augmentation:** all 8 dihedral symmetries of the board. Positions and move indices are permuted consistently.

**Optimisation:** batch size 2,048, learning rate 2e-3 when training from scratch and 4e-4 to 7e-4 when fine-tuning, a fixed time budget (5–8 min) on a single NVIDIA A40.

Typical end-of-training validation numbers (96×6): policy cross-entropy 2.155–2.17, value MSE ~0.085, top-1 move agreement with the teacher ~45%.

## 5. Export & validation

- `torch.onnx.export` at opset 17 with a dynamic batch axis.
- A post-processing step converts `Constant` nodes into initializers, as the contest's operator allow-list requires.
- The official `validate_submission.py` had to print `ACCEPTED` before any model was considered.

## 6. Gating & the continuous-improvement loop

```
best = c96
loop:
    train new model from best.pt on all shards (incl. freshly generated deeper data)
    validate ONNX contract
    play 200 games vs best with the referee MCTS (colours alternated)
    if win_rate ≥ 0.55: best = new model; copy to deliver/best.onnx
```

While this ran, 24 CPU workers kept producing new `pod5` shards with deeper labels. In 30 minutes the loop promoted one model (`it1`, +45 Elo over `c96`) and rejected two.
