# Saved models

All files satisfy the CCC Level 4 submission contract: input `observation` (batch, 16, 8, 8) float32, outputs `policy` (batch, 4096) and `value` (batch,), ONNX opset 17, under 5 MiB, no `Constant` nodes.

| File | Architecture | Trainable params | Role in the contest | Verified strength (200-game matches) |
|---|---|---|---|---|
| `handwritten_baseline.onnx` | Hand-written evaluation graph (material + positional tables) | n/a | Team's model before the training work | Baseline |
| `b64.onnx` | ResNet 64 ch × 6 blocks | 558,641 | **Main arena model** (19:00–19:44): 25 of its first 30 arena games won (83%) | 90.5% vs baseline · 66.5% vs `a64` |
| `a96.onnx` | ResNet 96 ch × 6 blocks | 1,164,465 | Early wide model, uploaded in the final minutes | 86.0% vs baseline |
| `it1.onnx` | ResNet 96 ch × 6 blocks | 1,164,465 | **Strongest verified model**: `c96` fine-tuned on deeper (al+4) labels by the auto-improvement loop | 56.5% vs `c96`, which itself beat `b64` 61% |

`it1.onnx` was saved from the loop as `best.onnx`. `c96` itself wasn't saved.

## Quick check

```bash
pip install onnxruntime numpy
python - <<'EOF'
import numpy as np, onnxruntime as ort
s = ort.InferenceSession("models/b64.onnx")
policy, value = s.run(None, {"observation": np.zeros((1, 16, 8, 8), np.float32)})
print(policy.shape, value.shape)   # (1, 4096) (1,)
EOF
```

MD5 checksums:

```
7a818c28c82a85c6baea1af7205608c5  b64.onnx
6b13165b431270707611c5443c0090c0  it1.onnx
58d78a0f8c8687642266c4343eb7f0ef  a96.onnx
a1c8abaf43c81b80825783f2f200e9bd  handwritten_baseline.onnx
```
