# Appendix to the MEPoser reproduction report

Detailed tables and diagnostics referenced from `REPORT.md`; every number traces to a run directory listed in `reports/results/EXPERIMENTS.md`.

## A. Motion content per subject

**Motion content per subject** (from the SMPL annotations; this table explains most of the variance in the results):

| subject | 0000 | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 | 0009 |
|---|---|---|---|---|---|---|---|---|---|---|
| upper-body joint speed cm/s | 34 | 35 | 29 | 30 | 26 | 30 | 28 | 4 | 22 | 11 |
| lower-body joint speed cm/s | 33 | 35 | 28 | 22 | 12 | 15 | 11 | 2 | 4 | 3 |
| knee flexion deg (mean) | 38 | 25 | 41 | 24 | 24 | 20 | 30 | 9 | 8 | 10 |
| both feet planted (fraction) | 0.23 | 0.36 | 0.19 | 0.69 | 0.87 | 0.77 | 0.79 | 1.00 | 1.00 | 1.00 |


## B. Every design decision, classified

**Every decision, classified.** (a) stated in the paper and followed: backbone, weight sharing, 22-joint heatmaps
with BCE, local/global 3D pathway with headset 6DoF, IMU MLP, concatenation, 3 x (LSTM + FFN) with residual + LN,
two 2-layer heads, FK, the five losses and their weights, Adam / 20 epochs / batch 32 / lr 1e-3 / decay at 7 and 14 for
the temporal model. (b) stated and changed, each with its reason in Section 4: end-to-end -> two stages; 640x480 ->
320x240; image-branch schedule 10 epochs with decays at 5 and 8; lr 1e-4 for the image-only ablation; sweeps in Appendix D
use decays at 10 and 16. (c) not stated and chosen, all in `configs/default.yaml` with comments: FPN to 256 channels,
conv heatmap encoder, shape loss 0.1, IMU feature set, temporal dim 512 / FFN 1024 / dropout 0.1 / unidirectional,
head hidden 512, heatmap sigma 1.5, sub-weights 1/1/1/1, window 32 frames, "local camera" = left camera, head
anchoring, hands = identity, ImageNet initialisation, identity initialisation of the pose head, LeakyReLU in MLPs and
GELU in the FFN, post-LN.


## C. Development-split results, seeds, backend

### Development split (train 0000-0007, validate 0008-0009), seed 0, paper recipe unless stated

| model | inputs | MPJRE | MPJPE | PA-MPJPE | UpperPE | LowerPE | RootPE | HandPE | MPJVE | Jitter |
|---|---|---|---|---|---|---|---|---|---|---|
| stage-1 image branch alone (per frame, no SMPL; per subject in `reports/results/feature_quality.md`: 0.4-1.2 cm on training subjects, 3.80 / 2.40 on 0008 / 0009) | stereo | - | 3.11 (local joints) | - | - | - | - | - | - | - |
| MEPoser-IMU (`configs/imu_only.yaml`) | 5 IMUs | 3.96 | 3.59 | 2.91 | 3.71 | 3.41 | 3.39 | 9.29 | 11.34 | 113 |
| MEPoser-CV (`configs/cv_only.yaml`, lr 1e-4, decays 10/16) | stereo + head 6DoF | 4.80 | 3.29 | 3.35 | 3.93 | 2.35 | 1.13 | 13.07 | 14.65 | 244 |
| **MEPoser-Full (`configs/default.yaml`)** | stereo + 5 IMUs | **3.96** | **2.90** | **2.73** | **3.30** | **2.33** | **1.69** | **8.84** | **11.06** | 146 |
| MEPoser-Full, literal variant (`configs/paper_literal.yaml`) | stereo + 5 IMUs | 4.05 | 3.09 | 2.88 | 3.37 | 2.68 | 1.71 | 9.28 | 12.96 | 201 | |
| MEPoser-Full, end-to-end from scratch (`configs/e2e_paper.yaml`) | stereo + 5 IMUs | 6.20 | 6.30 | 6.49 | 8.27 | 3.47 | 3.05 | 27.86 | 20.09 | 213 | |
| MEPoser-Full, end-to-end fine-tuned from the two-stage model (lr 1e-4, 3 epochs) | stereo + 5 IMUs | 4.09 | 3.35 | 3.41 | 3.94 | 2.49 | 1.45 | 12.01 | 13.17 | 195 |
| ground-truth annotations | | | | | | | | | | 64 |
| *paper, Protocol 1, full dataset: IMU / CV / Full* | | *4.6 / 5.0 / 4.1* | *6.2 / 4.5 / 3.7* | *3.6 / 2.9 / 2.5* | | | | | | *122 / 511 / 162* |

Per sequence the full model gets 3.69 cm on 0008 and 2.11 cm on 0009 (IMU-only 4.55 / 2.62). Seeds: full 2.90 / 2.82 /
2.92, IMU-only 3.59 / 3.61 / 3.68. The same full configuration trained on CPU gives 3.04 cm, outside the MPS seed range;
MPS is not bit-deterministic and this difference is unexplained. Training subjects: full 0.78 cm, IMU-only 0.89 cm.

### Leave-one-subject-out, IMU-only (paper recipe, last epoch)

| held out | 0000 | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 | 0009 | mean +- std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MPJPE cm | 2.19 | 5.71 | 3.08 | 3.83 | 3.82 | 4.46 | 6.31 | 2.72 | 4.37 | 2.68 | 3.92 +- 1.30 |

The three low-motion subjects (0007-0009) are the three easiest; the two most active ones (0001, 0006) are at the
paper's level for IMU-only (6.2). The development-split number 3.59 sits below the cross-validated mean because of
which subjects it holds out, not because the model is better than the paper's.

## D. Diagnostics and sweeps

### Sweeps (development split, seed 0; full config differences in EXPERIMENTS.md)

| question | runs (config difference from default) | answer |
|---|---|---|
| Is image-only limited by feature quality or by the joints -> SMPL decoding? | image-only fed **ground-truth** local joints ("oracle"): 6.84 at lr 1e-3 / decays 7,14; **2.17** at lr 1e-4 / decays 10,16 | at 1e-3 the optimisation is stuck (train joint loss 2.9 cm vs 0.4 cm for IMU-only; `curves_stage2.png`); at 1e-4 the decoder recovers GT joints to 2.2 cm and predicted ones to 3.3 |
| learning rate, fused model | 1e-3: 2.90 (decays 7,14); 1e-3 + grad-clip 1.0: 2.84; 3e-4: 2.94; 1e-4: 3.18 (all three with decays 10,16) | paper recipe is right for the fused model; the IMU path drives early training |
| learning rate, image-only | 1e-3: 6.92 (7,14); 1e-3 + clip: 5.84; 3e-4 + clip: 3.33; 1e-4: 3.29 (all 10,16) | clipping does not rescue 1e-3; the rate does |
| learning rate, IMU-only | 1e-3: 3.59 (7,14); 3e-4: 3.64; 1e-4: 3.71 (10,16) | paper recipe |
| temporal encoder size (lr 1e-3, decays 7,14) | blocks 1 / 2 / 3 / 5: 2.92 / 3.03 / 2.90 / 7.71; dim 256 / 512 / 768: 2.95 / 2.90 / 6.47; FFN 4x: 6.50 | one block is enough on 25 k frames; wider or deeper encoders diverge at 1e-3, the recipe's stability is specific to the paper's scale |
| noise on the cached image outputs in stage 2 (to close the train/held-out feature-quality gap) | joint noise 0 / 1 / 2 / 4 cm (+ feature dropout 0.1): 2.90 / 2.97 / 2.95 / 3.15 | no gain |
| joints expressed in the head frame + head rotation as extra input | full 3.45, image-only 5.87 | worse than the paper's world frame |
| window length, IMU-only | 32: 3.59; 64 (stride 8): 3.53 | within seed noise (std 0.05) |
| smoothness loss, IMU-only | with: 3.59 / jitter 113; without: 3.64 / 116 | negligible here; the LSTM already smooths; all models sit at ~2x the annotation jitter |
| end-to-end fine-tuning from the two-stage model (images in the loop, lr 1e-4, batch 4 x 16, 3 epochs) | 3.87 -> 3.47 -> 3.35 | hurts, starting from 2.90; batch-norm with 64 images per step and further backbone overfitting are the suspected causes (untested) |
| feature-noise augmentation on the folds where the image features degrade most (folds 1 and 0 of the 5-fold run, stage 2 retrained on CPU) | fold 1 (0001 + 0006): none 8.54 / 12.70; 3 cm + 0.2: 5.76 / 5.50; 5 cm + 0.3: 5.81 / 5.65. fold 0 (0000 + 0005): none 2.09 / 3.44; 3 cm: 2.07 / 3.85; 5 cm: 2.33 / 4.08 | yes where it matters: the failure fold drops from 10.6 to 5.6 cm (below IMU-only 6.5 and image-only 6.5 on that fold) at a cost of 0.3 cm on an easy fold; the augmentation that did nothing on the development split is the one that makes fusion robust across subjects. Applied to all five folds (`runs/cv5_full_noise3`), the same augmentation gives **3.62 +- 1.21 cm (median 3.59, frame-weighted 3.72)**: better than IMU-only on all ten subjects (4.19 +- 1.44) and better than image-only on average (3.80 +- 1.54, six of ten subjects); per subject 2.07 / 5.76 / 2.91 / 3.58 / 3.61 / 3.85 / 5.50 / 2.32 / 4.23 / 2.37. |

## E. Extensions beyond the paper


*Synthetic left-hand acceleration.* The dataset's left-hand acceleration is unusable (Section 2). Using the right hand,
where the real signal exists, as a calibration set, the second difference of the controller position low-passed at 3 Hz
(2nd-order zero-phase Butterworth) correlates 0.81 with the real acceleration. IMU-only, development split: baseline
3.59; + synthetic left acc 3.60; both hands synthetic 3.60; right real -> synthetic 3.66; no hand accelerations at all
3.66. Full model: 2.90 -> 3.00 (+ left) / 2.93 (both). Hand acceleration channels carry almost no information beyond the
positions on this subset; the option stays off.

*Foot contact.* Pseudo labels from the SMPL feet (ankle or toe within 3 cm of the floor and moving < 1 cm/frame),
a two-logit contact head on the temporal feature (BCE, weight 0.5), and a foot-skate penalty (predicted foot velocity
while the GT foot is planted, weight 5); at inference the head predicts contact from IMU + image + history
(`configs/imu_contact.yaml`). IMU-only, development split, CPU, same seed; the contact run was interrupted in its 20th
epoch, so its numbers are from epoch 19 of 20 and the control's from epoch 20: MPJPE 4.71 -> 4.52 (0008) and
2.57 -> 2.46 (0009), LowerPE 4.59 -> 3.96 / 2.60 -> 2.64, foot skate 16.0 -> 11.5 and 8.8 -> 6.8 cm/s (annotations: 3.3
/ 3.0), contact accuracy 0.96 / 1.00 against the pseudo labels. One seed, and the development validation subjects stand
still (contact is trivially "always planted" there), so this is a promising direction rather than a result; it needs the
5-fold protocol on the active subjects.

## F. Compute and backend incidents

Fold 1 of the 5-fold run was restarted once: its stage-2 process crashed because the code was edited while it was running (DataLoader worker processes re-import the package), which is why its `run_info.json` says `git_dirty: true`; the restarted stage 2 reused the fold's finished stage-1 checkpoint and features.


One Apple M4 Max, 36 GB unified memory, PyTorch 2.10 MPS backend. Stage 1: 44 min (10 epochs, 24.6 k frames).
Feature cache: 1 min. Stage 2: 2 min per run (20 epochs) on MPS, 18 min on CPU. 5-fold cross-validation of the full model: about 3.5 h (five stage-1 trainings of ~40 min plus five 2-min stage-2 runs); the
IMU-only and image-only folds ran on the CPU in parallel (~18 min each).
End-to-end from scratch: 2 h 17 min (20 epochs, 749 steps of 4 x 16 frames per epoch). Two MPS numerical incidents: one
non-deterministic NaN in a stage-2 run (same seed clean on CPU and on a second MPS run; the trainer now skips non-finite
steps and counts them), and one wrong result from a broadcasting batched matmul in eval mode (rewritten as `points @ R^T`,
verified equal to CPU). All development numbers were re-evaluated on CPU from the saved checkpoints.

## G. Status of every component

| component | status |
|---|---|
| data loading, frame-id synchronisation, gap-aware windows, calibration, inspection report | implemented, verified by tests and `inspect` |
| weight-shared RegNetY-400MF, 22-joint heatmaps + BCE, local/global 3D pathway, IMU MLP, concatenation | implemented as in the paper |
| 256-channel feature map (FPN), conv heatmap encoder, shape loss | additions; literal alternatives implemented (`paper_literal.yaml`), the literal variant scores 3.09 cm vs 2.90 on the development split (stage-1 per-frame 4.17 vs 3.11 cm, 18.8 M vs 6.9 M parameters); my additions help the image branch but are not what the paper describes, both are kept |
| 3 x (LSTM + FFN, residual + LN), causal | implemented as in the paper; width/FFN/dropout chosen |
| 2-layer pose/shape heads, FK, five losses with Appendix-B weights | implemented as in the paper |
| paper training recipe | followed for the temporal model; two-stage training and 320x240 are compute-driven approximations; the paper-faithful end-to-end run with the batch reduced to fit memory gives 6.30 cm (Section 4) |
| protocol metrics (HMD-Poser / UnrealEgo code), causal full-sequence evaluation, K-fold | implemented |
| failed attempts (with numbers) | end-to-end fine-tuning from the two-stage model, head-frame joints, wider/deeper temporal encoders at lr 1e-3, synthetic hand acceleration |
| helped only under cross-validation | feature-noise augmentation in stage 2: no gain on the development split, best 5-fold mean (3.62 cm); its strength was chosen on two of the five folds (Appendix D) |
| untested ideas | 640x480 input, world-yaw / IMU-noise / leg-offset augmentation, explicit stereo triangulation from the two heatmap peaks, wrist end-effector loss from controller positions, causal output filter, freezing batch-norm in end-to-end training, out-of-fold feature extraction |
