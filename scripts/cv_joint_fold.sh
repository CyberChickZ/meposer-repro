#!/usr/bin/env bash
# One fold of joint training for K-fold CV: fold-specific stage-1 warm start (no held-out subject seen), then joint training.
[ "$1" -ge "${MAX_FOLDS:-3}" ] && { echo "skipped fold $1 (MAX_FOLDS=${MAX_FOLDS:-3})"; exit 0; }
set -e
k=$1; M=${M:-meposer}
read -r TR VA < <(python3 -c "import json;s=[f'{i:04d}' for i in range(10)];h=s[$k::5];print(json.dumps([x for x in s if x not in h]).replace(' ',''), json.dumps(h).replace(' ',''))")
$M train -c configs/default.yaml -o runs/cv5_jointwarm/fold$k --stage image --set "data.train_subjects=$TR" "data.val_subjects=$VA" device=cuda threads=4 data.num_workers=8 model.image.amp=true
$M train -c configs/joint_steps_warm.yaml -o runs/cv5_jointwarm/fold$k --stage full --set "data.train_subjects=$TR" "data.val_subjects=$VA" model.image.init_from=runs/cv5_jointwarm/fold$k/stage1_image/best.pt device=cuda threads=4 data.num_workers=8
