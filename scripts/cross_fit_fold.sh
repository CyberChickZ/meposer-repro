#!/usr/bin/env bash
# Cross-fitted (out-of-fold) image features for outer fold k: the training subjects' features come from stage-1 models
# that never saw them (2 inner halves), the held-out subjects' features from the outer stage-1 model; then stage 2.
set -e
k=$1; M=${M:-meposer}; O=runs/cf/fold$k
read -r TR VA < <(python3 -c "import json;s=[f'{i:04d}' for i in range(10)];h=s[$k::5];print(json.dumps([x for x in s if x not in h]).replace(' ',''), json.dumps(h).replace(' ',''))")
read -r A B < <(python3 -c "import json;t=json.loads('$TR');print(json.dumps(t[0::2]).replace(' ',''), json.dumps(t[1::2]).replace(' ',''))")
S="device=cuda threads=4 data.num_workers=8 model.image.amp=true"
$M train -c configs/default.yaml -o $O/inner_a --stage image --set "data.train_subjects=$A" "data.val_subjects=$B" $S
$M train -c configs/default.yaml -o $O/inner_a --stage features --set "data.train_subjects=$A" "data.val_subjects=$B" $S
$M train -c configs/default.yaml -o $O/inner_b --stage image --set "data.train_subjects=$B" "data.val_subjects=$A" $S
$M train -c configs/default.yaml -o $O/inner_b --stage features --set "data.train_subjects=$B" "data.val_subjects=$A" $S
$M train -c configs/default.yaml -o runs/cv5_jointwarm/fold$k --stage features --set "data.train_subjects=$TR" "data.val_subjects=$VA" $S
mkdir -p $O/features
python3 - <<PY
import json, shutil
A, B, VA = json.loads('$A'), json.loads('$B'), json.loads('$VA')
for s in B: shutil.copy(f"$O/inner_a/stage1_image/features/{s}.npz", f"$O/features/{s}.npz")
for s in A: shutil.copy(f"$O/inner_b/stage1_image/features/{s}.npz", f"$O/features/{s}.npz")
for s in VA: shutil.copy(f"runs/cv5_jointwarm/fold$k/stage1_image/features/{s}.npz", f"$O/features/{s}.npz")
json.dump({"out_of_fold_from_inner_a": B, "out_of_fold_from_inner_b": A, "held_out_from_outer": VA}, open("$O/features/meta.json", "w"), indent=1)
PY
for v in plain noise; do
  x=""; [ $v = noise ] && x="model.feature_noise.joints_m=0.03 model.feature_noise.image_feat_dropout=0.2"
  $M train -c configs/default.yaml -o $O/stage2_$v --stage full --set model.image_features=$O/features "data.train_subjects=$TR" "data.val_subjects=$VA" device=cuda threads=4 data.num_workers=0 $x
done
