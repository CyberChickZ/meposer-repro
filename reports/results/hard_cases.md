Fold holding out 0001 + 0006 (the two most active subjects); per-frame joint error in cm, head-aligned; each row adds one change to the previous row.

**all frames** (6987 frames)

| variant | MPJPE | legs | arms |
|---|---|---|---|
| MEPoser (reproduced) | 5.46 | 8.92 | 9.69 |
| + augmentation + contact head | 4.65 | 8.76 | 6.87 |
| + wrist IK | 3.83 | 8.76 | 2.53 |
| + stereo 2D fit | 3.11 | 6.22 | 2.40 |
| + adaptive leg ROI | 2.58 | 4.21 | 2.45 |
| + low-pass (final) | 2.50 | 4.01 | 2.38 |

**fast motion (top 25% joint speed)** (1747 frames)

| variant | MPJPE | legs | arms |
|---|---|---|---|
| MEPoser (reproduced) | 6.84 | 11.07 | 12.14 |
| + augmentation + contact head | 4.60 | 9.85 | 5.05 |
| + wrist IK | 4.10 | 9.85 | 2.36 |
| + stereo 2D fit | 3.44 | 7.53 | 2.25 |
| + adaptive leg ROI | 2.84 | 5.28 | 2.30 |
| + low-pass (final) | 2.73 | 4.94 | 2.24 |

**legs occluded (<50% leg joints visible)** (28 frames)

| variant | MPJPE | legs | arms |
|---|---|---|---|
| MEPoser (reproduced) | 2.59 | 4.32 | 2.58 |
| + augmentation + contact head | 5.35 | 8.18 | 7.25 |
| + wrist IK | 4.39 | 8.18 | 2.25 |
| + stereo 2D fit | 6.63 | 16.43 | 2.25 |
| + adaptive leg ROI | 7.39 | 19.20 | 2.31 |
| + low-pass (final) | 6.34 | 15.43 | 2.24 |

**legs visible (all leg joints visible)** (6930 frames)

| variant | MPJPE | legs | arms |
|---|---|---|---|
| MEPoser (reproduced) | 5.49 | 8.96 | 9.75 |
| + augmentation + contact head | 4.65 | 8.78 | 6.88 |
| + wrist IK | 3.83 | 8.78 | 2.53 |
| + stereo 2D fit | 3.09 | 6.15 | 2.41 |
| + adaptive leg ROI | 2.56 | 4.15 | 2.45 |
| + low-pass (final) | 2.49 | 3.96 | 2.38 |

**arms out of view (no arm joint visible)** (20 frames)

| variant | MPJPE | legs | arms |
|---|---|---|---|
| MEPoser (reproduced) | 3.59 | 5.92 | 4.90 |
| + augmentation + contact head | 5.72 | 8.17 | 8.82 |
| + wrist IK | 4.43 | 8.17 | 2.02 |
| + stereo 2D fit | 4.75 | 9.44 | 1.96 |
| + adaptive leg ROI | 6.57 | 16.15 | 1.97 |
| + low-pass (final) | 5.28 | 11.49 | 1.97 |

**fast + legs occluded** (0 frames)

| variant | MPJPE | legs | arms |
|---|---|---|---|
| MEPoser (reproduced) | nan | nan | nan |
| + augmentation + contact head | nan | nan | nan |
| + wrist IK | nan | nan | nan |
| + stereo 2D fit | nan | nan | nan |
| + adaptive leg ROI | nan | nan | nan |
| + low-pass (final) | nan | nan | nan |
