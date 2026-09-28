3-fold cross-validation by subject (folds 0-2: held out 0000+0005, 0001+0006, 0002+0007); every model on the same GPU-trained per-fold stage-1 features; last epoch; MPJPE cm per held-out subject. post: raw network output, or test-time wrist IK from the controllers + 4 Hz zero-phase low-pass (offline).

| model | post | 0000 | 0001 | 0002 | 0005 | 0006 | 0007 | MPJPE mean | PA | MPJRE | HandPE | LowerPE | Jitter (GT) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_contact | raw | 2.25 | 3.81 | 2.57 | 3.68 | 5.50 | 2.43 | **3.37** | 2.85 | 3.83 | 6.29 | 4.39 | 233 (62) |
| best_contact | IK+LP4 | 2.11 | 3.50 | 2.05 | 2.82 | 4.10 | 1.91 | **2.75** | 2.26 | 3.70 | 2.16 | 4.37 | 90 (62) |
| best_contact | IK+2D+LP4 | 1.95 | 3.09 | 2.33 | 2.30 | 3.04 | 1.83 | **2.42** | 1.99 | 3.74 | 2.15 | 3.61 | 105 (62) |
| best_contact | IK+legROI2D+LP4 | 1.81 | 2.46 | 1.94 | 1.86 | 2.55 | 1.68 | **2.05** | 1.66 | 3.76 | 2.16 | 2.68 | 97 (62) |
