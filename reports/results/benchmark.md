Measured by `scripts/bench_backbone.py` (torch 2.10.0); memory = tensors allocated during the forward pass and kept for the backward pass (activations), divided by the batch.

| device | resolution | batch | img/s (fwd+bwd+Adam) | forward activations MB per image |
|---|---|---|---|---|
| mps | 320x240 | 64 | 384 | 51 |
| mps | 320x240 | 16 | 344 | 120 |
| mps | 640x480 | 32 | 98 | 209 |
| mps | 640x480 | 8 | 93 | 269 |
