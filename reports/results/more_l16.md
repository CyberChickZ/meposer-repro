3-fold cross-validation by subject (folds 0-2: held out 0000+0005, 0001+0006, 0002+0007); every model on the same GPU-trained per-fold stage-1 features; last epoch; MPJPE cm per held-out subject. post: raw network output, or test-time wrist IK from the controllers + 4 Hz zero-phase low-pass (offline).

| model | post | 0000 | 0001 | 0002 | 0005 | 0006 | 0007 | MPJPE mean | PA | MPJRE | HandPE | LowerPE | Jitter (GT) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_long | raw | 2.07 | 3.49 | 2.44 | 3.05 | 4.88 | 2.04 | **2.99** | 2.49 | 3.54 | 5.79 | 3.81 | 233 (62) |
| best_long | IK+LP4 | 2.02 | 3.25 | 1.91 | 2.42 | 3.53 | 1.71 | **2.47** | 1.99 | 3.49 | 2.15 | 3.78 | 89 (62) |
| best_long | IK+2D+LP4 | 1.89 | 2.99 | 2.33 | 2.12 | 2.86 | 1.68 | **2.31** | 1.85 | 3.53 | 2.14 | 3.41 | 102 (62) |
| best_long | IK+legROI2D+LP4 | 1.79 | 2.40 | 1.96 | 1.66 | 2.39 | 1.55 | **1.96** | 1.54 | 3.56 | 2.16 | 2.53 | 93 (62) |
