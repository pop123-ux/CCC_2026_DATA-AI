"""Colony: One policy/value network, as used for b64 / c96 / d96.

Copied from work/train.py on the RunPod pod (the model-definition part only).
  b64 = Net(c=64, nb=6)   -> 558,641 params  (deployed model)
  c96 = Net(c=96, nb=6)   -> 1,164,465 params
Input : observation (batch, 16, 8, 8) float32
Output: policy (batch, 4096) raw logits over from*64+to moves, value (batch,) in [-1, 1]
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- move offsets (all legal moves in Colony: One lie on these 32 offsets)
# 8 directions (orthogonal + diagonal) x step lengths 1..3  = 24 offsets
# + 8 knight jumps                                         = 8 offsets
OFFSETS = []
for d in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
    for k in (1, 2, 3):
        OFFSETS.append((d[0] * k, d[1] * k))
OFFSETS += [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
NOFF = len(OFFSETS)  # 32
OFF_ID = {o: i for i, o in enumerate(OFFSETS)}

# Map each of the 4096 (from, to) move slots to an index in the
# flattened (32 offset planes x 64 squares) policy head output.
# Moves that are not one of the 32 offsets map to index 0 (always illegal anyway).
GATHER = np.zeros(4096, dtype=np.int64)
for f in range(64):
    for t in range(64):
        o = (t // 8 - f // 8, t % 8 - f % 8)
        if o in OFF_ID:
            GATHER[f * 64 + t] = OFF_ID[o] * 64 + f


class Res(nn.Module):
    """Standard residual block: conv-BN-ReLU-conv-BN + skip, then ReLU."""

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    """AlphaZero-style residual policy/value net.

    c  = channel width (64 for b64, 96 for c96/d96)
    nb = number of residual blocks (6 for all current models)
    """

    def __init__(self, c=64, nb=4):
        super().__init__()
        # 16 input planes -> c feature maps
        self.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        # trunk
        self.blocks = nn.Sequential(*[Res(c) for _ in range(nb)])
        # policy head: 32 move-offset planes per square -> gathered into 4096 logits
        self.ph = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                                nn.Conv2d(c, NOFF, 1))
        # value head: 8 planes x 64 squares = 512 -> 128 -> 1, squashed by tanh
        self.vh = nn.Sequential(nn.Conv2d(c, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(),
                                nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.register_buffer("gidx", torch.from_numpy(GATHER))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        p = self.ph(h).flatten(1)                       # (B, 32*64)
        policy = torch.index_select(p, 1, self.gidx)    # (B, 4096) raw logits
        value = torch.tanh(self.vh(h)).reshape(-1)      # (B,) in [-1, 1]
        return policy, value


if __name__ == "__main__":
    for name, c in [("b64", 64), ("c96", 96)]:
        net = Net(c=c, nb=6).eval()
        n = sum(p.numel() for p in net.parameters())
        pol, val = net(torch.zeros(2, 16, 8, 8))
        print(f"{name}: {n:,} params | policy {tuple(pol.shape)} | value {tuple(val.shape)}")
