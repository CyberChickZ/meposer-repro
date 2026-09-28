All metrics, 3-fold cross-validation by folder (6 held-out folders: 0000+0005, 0001+0006, 0002+0007), mean over folders, last epoch. MPJRE in degrees; PE in cm; Jitter as in HMD-Poser's code (the paper gives no unit); FS = mean foot-joint speed in cm/s on frames where the ground-truth foot is planted (our metric, not in the paper; ground truth itself: 7.3). post: raw network output, IK = controller wrist IK, 2D = stereo reprojection fit, legROI2D = 2D fit on adaptive leg-crop peaks, LP4 = 4 Hz zero-phase low-pass, contact = contact fit. Rows of the ours models with 2D fitting were recomputed after the half-cell fix in peak decoding; the 2D-fit rows of the other models predate it and are about 0.05 cm pessimistic.

| model | post | MPJRE | MPJPE | PA-MPJPE | UpperPE | LowerPE | RootPE | Jitter | HandPE | FS |
|---|---|---|---|---|---|---|---|---|---|---|
| MEPoser-IMU (reproduced) | raw | 3.93 | 4.21 | 3.20 | 3.29 | 5.53 | 3.09 | 235 | 7.41 | 23.88 |
| MEPoser-IMU (reproduced) | IK+LP4 | 3.83 | 3.46 | 2.54 | 2.03 | 5.51 | 3.08 | 95 | 2.16 | 19.59 |
| MEPoser-IMU (reproduced) | IK+2D+LP4 | 3.83 | 3.46 | 2.54 | 2.03 | 5.51 | 3.08 | 95 | 2.16 | 19.59 |
| MEPoser-CV (reproduced) | raw | 4.78 | 4.02 | 2.88 | 3.46 | 4.84 | 2.25 | 328 | 8.85 | 30.34 |
| MEPoser-CV (reproduced) | IK+LP4 | 4.59 | 3.20 | 2.41 | 2.08 | 4.82 | 2.24 | 104 | 2.21 | 22.38 |
| MEPoser-CV (reproduced) | IK+2D+LP4 | 4.67 | 2.93 | 2.27 | 2.07 | 4.16 | 2.24 | 116 | 2.19 | 25.85 |
| MEPoser-Full (reproduced, two-stage) | raw | 3.83 | 3.53 | 2.89 | 2.99 | 4.30 | 2.19 | 247 | 7.24 | 23.68 |
| MEPoser-Full (reproduced, two-stage) | IK+LP4 | 3.72 | 2.80 | 2.21 | 1.78 | 4.27 | 2.18 | 92 | 2.17 | 17.56 |
| MEPoser-Full (reproduced, two-stage) | IK+2D+LP4 | 3.76 | 2.57 | 2.00 | 1.76 | 3.74 | 2.18 | 104 | 2.15 | 22.83 |
| MEPoser-Full, joint training (first config) | raw | 5.31 | 5.76 | 5.05 | 5.12 | 6.68 | 3.16 | 207 | 14.38 | 25.61 |
| MEPoser-Full, joint training (first config) | IK+LP4 | 4.87 | 4.06 | 3.27 | 2.26 | 6.66 | 3.15 | 92 | 2.18 | 23.15 |
| MEPoser-Full, joint training (first config) | IK+2D+LP4 | 4.87 | 4.06 | 3.27 | 2.26 | 6.66 | 3.15 | 92 | 2.18 | 23.15 |
| MEPoser-Full, joint training (effective batch 32) | raw | 4.49 | 4.26 | 3.71 | 3.65 | 5.14 | 2.35 | 249 | 9.22 | 23.45 |
| MEPoser-Full, joint training (effective batch 32) | IK+LP4 | 4.16 | 3.22 | 2.62 | 1.90 | 5.12 | 2.33 | 95 | 2.16 | 18.49 |
| MEPoser-Full, joint training (effective batch 32) | IK+2D+LP4 | 4.16 | 3.22 | 2.62 | 1.90 | 5.12 | 2.33 | 95 | 2.16 | 18.49 |
| ours: + augmentation | raw | 3.82 | 3.36 | 2.79 | 2.63 | 4.40 | 1.91 | 239 | 6.15 | 24.53 |
| ours: + augmentation | IK+LP4 | 3.68 | 2.76 | 2.23 | 1.64 | 4.38 | 1.90 | 92 | 2.17 | 18.94 |
| ours: + augmentation | IK+2D+LP4 | 3.75 | 2.47 | 2.04 | 1.61 | 3.71 | 1.90 | 104 | 2.15 | 23.35 |
| ours: + augmentation + contact head | raw | 3.83 | 3.37 | 2.85 | 2.67 | 4.39 | 1.89 | 233 | 6.29 | 21.05 |
| ours: + augmentation + contact head | IK+LP4 | 3.70 | 2.75 | 2.26 | 1.63 | 4.37 | 1.87 | 90 | 2.16 | 16.98 |
| ours: + augmentation + contact head | IK+2D+LP4 | 3.74 | 2.42 | 1.99 | 1.60 | 3.61 | 1.87 | 105 | 2.15 | 23.10 |
| ours: all training additions | raw | 3.96 | 3.83 | 3.29 | 2.91 | 5.17 | 2.25 | 196 | 6.71 | 19.20 |
| ours: all training additions | IK+LP4 | 3.80 | 3.15 | 2.62 | 1.76 | 5.16 | 2.24 | 89 | 2.17 | 16.87 |
| ours: all training additions | IK+2D+LP4 | 3.82 | 2.65 | 2.17 | 1.73 | 3.97 | 2.24 | 105 | 2.15 | 23.92 |
| ours: + augmentation + contact head | IK+legROI2D+LP4 | 3.76 | 2.05 | 1.66 | 1.61 | 2.68 | 1.87 | 97 | 2.16 | 18.67 |
| ours: + augmentation + contact head, 40 epochs | raw | 3.54 | 2.99 | 2.49 | 2.43 | 3.81 | 1.73 | 233 | 5.79 | 21.36 |
| ours: + augmentation + contact head, 40 epochs | IK+LP4 | 3.49 | 2.47 | 1.99 | 1.57 | 3.78 | 1.71 | 89 | 2.15 | 16.36 |
| ours: + augmentation + contact head, 40 epochs | IK+2D+LP4 | 3.53 | 2.31 | 1.85 | 1.55 | 3.41 | 1.71 | 102 | 2.14 | 22.69 |
| ours: + augmentation + contact head, 40 epochs | IK+legROI2D+LP4 | 3.56 | 1.96 | 1.54 | 1.56 | 2.53 | 1.71 | 93 | 2.16 | 17.72 |
| ours: + augmentation + contact head, 40 epochs | IK+legROI2D+LP4+contact | 3.62 | 1.99 | 1.58 | 1.56 | 2.61 | 1.71 | 110 | 2.16 | 10.16 |

EMHI paper, Table 2 (full dataset, different subjects and actions; not directly comparable):

| protocol | method | MPJRE | MPJPE | PA-MPJPE | UpperPE | LowerPE | RootPE | Jitter | HandPE | FS |
|---|---|---|---|---|---|---|---|---|---|---|
| Protocol 1 | UnrealEgo | — | 5.5 | 3.9 | 4.0 | 7.7 | 4.2 | 592.5 | n/r | n/r |
| Protocol 1 | HMD-Poser | 4.6 | 5.8 | 2.8 | 4.8 | 7.1 | 5.8 | 114.9 | n/r | n/r |
| Protocol 1 | MEPoser-CV | 5.4 | 4.5 | 2.9 | 3.3 | 6.3 | 3.8 | 511.0 | n/r | n/r |
| Protocol 1 | MEPoser-IMU | 5.0 | 6.2 | 3.6 | 5.0 | 8.0 | 5.2 | 121.7 | n/r | n/r |
| Protocol 1 | MEPoser-Full | 4.1 | 3.7 | 2.5 | 2.7 | 5.1 | 3.2 | 161.8 | n/r | n/r |
| Protocol 2 | UnrealEgo | — | 6.4 | 4.3 | 4.6 | 8.9 | 5.0 | 610.5 | n/r | n/r |
| Protocol 2 | HMD-Poser | 4.9 | 7.0 | 3.4 | 5.2 | 9.7 | 7.2 | 165.7 | n/r | n/r |
| Protocol 2 | MEPoser-CV | 6.0 | 5.4 | 3.5 | 3.8 | 7.8 | 4.4 | 566.5 | n/r | n/r |
| Protocol 2 | MEPoser-IMU | 5.7 | 7.1 | 4.2 | 5.4 | 9.9 | 6.6 | 161.7 | n/r | n/r |
| Protocol 2 | MEPoser-Full | 4.7 | 4.8 | 2.9 | 3.2 | 7.0 | 3.8 | 204.9 | n/r | n/r |

Protocol 1: unseen subjects, seen actions. Protocol 2: unseen subjects and unseen actions. n/r: not reported.
