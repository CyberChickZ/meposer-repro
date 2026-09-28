#!/usr/bin/env bash
# Final batch on one GPU: for folds 0-2, extract outer-fold features, train every stage-2 variant on them, and evaluate
# every model (incl. the joint-training folds) with and without test-time wrist IK + 4 Hz zero-phase low-pass.
# Usage: COMBO="key=value ..." bash scripts/final_pot.sh
set -e
M=${M:-meposer}; OUT=runs/final
S="device=cuda threads=4 data.num_workers=0"
for k in 0 1 2; do
  read -r TR VA < <(python3 -c "import json;s=[f'{i:04d}' for i in range(10)];h=s[$k::5];print(json.dumps([x for x in s if x not in h]).replace(' ',''), json.dumps(h).replace(' ',''))")
  F=runs/cv5_jointwarm/fold$k/stage1_image/features
  [ -f $F/meta.json ] || $M train -c configs/default.yaml -o runs/cv5_jointwarm/fold$k --stage features --set "data.train_subjects=$TR" "data.val_subjects=$VA" device=cuda threads=4 model.image.amp=true
  (
  $M train -c configs/imu_only.yaml -o $OUT/imu_only/fold$k --stage full --set "data.train_subjects=$TR" "data.val_subjects=$VA" $S > /dev/null 2>&1
  $M train -c configs/cv_only.yaml  -o $OUT/cv_only/fold$k  --stage full --set model.image_features=$F "data.train_subjects=$TR" "data.val_subjects=$VA" $S > /dev/null 2>&1
  ) &
  (
  $M train -c configs/default.yaml  -o $OUT/two_stage/fold$k --stage full --set model.image_features=$F "data.train_subjects=$TR" "data.val_subjects=$VA" $S > /dev/null 2>&1
  $M train -c ${COMBO_CONFIG:-configs/default.yaml} -o $OUT/best/fold$k --stage full --set model.image_features=$F "data.train_subjects=$TR" "data.val_subjects=$VA" $S $COMBO > /dev/null 2>&1
  ) &
done
wait
until [ $(grep -l "\] done" runs/cv5_jointwarm/fold{0,1,2}/log.txt 2>/dev/null | wc -l) -eq 3 ]; do sleep 60; done
mkdir -p $OUT/joint; for k in 0 1 2; do ln -sfn ../../cv5_jointwarm/fold$k $OUT/joint/fold$k; done
python3 scripts/final_table.py $OUT
