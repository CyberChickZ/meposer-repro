#!/usr/bin/env bash
# Full reproduction of every number in reports/REPORT.md from the raw EMHI subset (one CUDA GPU, ~10 h on an H200).
#   data/raw/<subject>/{annots/*.npy, camera/cam{0,1}/*.png}   (obtain the EMHI subset yourself, see README)
# Protocol: 3-fold cross-validation by subject, fold k holds out subjects k and k+5 (0000+0005, 0001+0006, 0002+0007).
set -euo pipefail
cd "$(dirname "$0")/.."
M=${M:-meposer}; P=${P:-python}
S="device=cuda threads=4 data.num_workers=8"

$M prepare --raw data/raw --out data/processed                                   # 320x240 memmaps, calibration, inspection
$M prepare --raw data/raw --out data/processed_640 --image-size 640x480          # native resolution, leg crops only

for k in 0 1 2; do
  read -r TR VA < <($P -c "import json;s=[f'{i:04d}' for i in range(10)];h=s[$k::5];print(json.dumps([x for x in s if x not in h]).replace(' ',''), json.dumps(h).replace(' ',''))")
  SPLIT=("data.train_subjects=$TR" "data.val_subjects=$VA")
  F=runs/cv/fold$k/stage1_image/features
  # stage 1: image branch per fold (never sees the held-out subjects), then its cached outputs
  $M train -c configs/default.yaml -o runs/cv/fold$k --stage image    --set "${SPLIT[@]}" $S model.image.amp=true
  $M train -c configs/default.yaml -o runs/cv/fold$k --stage features --set "${SPLIT[@]}" $S model.image.amp=true
  # stage 2: reproduction (two-stage MEPoser and single-modality ablations) and our model
  $M train -c configs/imu_only.yaml -o runs/final/imu_only/fold$k     --stage full --set "${SPLIT[@]}" $S
  $M train -c configs/cv_only.yaml  -o runs/final/cv_only/fold$k      --stage full --set "${SPLIT[@]}" $S model.image_features=$F
  $M train -c configs/default.yaml  -o runs/final/two_stage/fold$k    --stage full --set "${SPLIT[@]}" $S model.image_features=$F
  $M train -c configs/best.yaml     -o runs/final/best_contact/fold$k --stage full --set "${SPLIT[@]}" $S model.image_features=$F
  # joint (end-to-end) training, warm-started from the fold's image branch
  $M train -c configs/joint_b32_short.yaml -o runs/final/joint_b32/fold$k --stage full --set "${SPLIT[@]}" $S \
    model.image.init_from=runs/cv/fold$k/stage1_image/best.pt
  # adaptive lower-body crop: leg network on 640x480 crops, then peaks around our model's own leg prediction
  TRS=$($P -c "import json;print(' '.join(json.loads('$TR')))"); VAS=$($P -c "import json;print(' '.join(json.loads('$VA')))")
  $P scripts/leg_roi.py train --train $TRS --val $VAS --init runs/cv/fold$k/stage1_image/best.pt --out runs/leg_roi_fold$k
  $P scripts/leg_roi.py infer --train $TRS --val $VAS --ckpt runs/final/best_contact/fold$k/last.pt --out runs/leg_roi_fold$k \
    --set model.image_features=$F
done

$P scripts/final_table.py runs/final                       # -> runs/final/final_table.{md,json}   (report section 4)
$P scripts/hard_cases.py                                   # -> reports/results/hard_cases.md      (report section 5)
$P scripts/render_mesh.py --ckpt runs/final/best_contact/fold1/last.pt --baseline runs/final/two_stage/fold1/last.pt \
  --subject 0006 --start 991 --seconds 6 --no-images --png reports/figures/hard_0006.png --out reports/figures/hard_0006 \
  --set model.image_features=runs/cv/fold1/stage1_image/features refine_2d_features=runs/leg_roi_fold1/features \
        refine_2d_weight=3 refine_prior=0.03 refine_iters=150
