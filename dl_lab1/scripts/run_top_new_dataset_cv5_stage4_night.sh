#!/bin/zsh
set -euo pipefail

# Night pipeline for top_new_dataset:
# 1) Train CV5 zoo (overnight_10h preset)
# 2) Apply stage4 confidence-aware finetune in-place to the 3 useful model families (skip vit_base_augreg_lite)
# 3) Build final CV5 submission with LR(MSE) fold weights + geo8 TTA
#
# Use after fold0 probe confirms positive signal for stage4.

PY="/opt/homebrew/bin/python3.11"
ROOT="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1"

BASE_DATASET="${ROOT}/top_new_dataset"
FOLDS_CSV="${ROOT}/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"

# Main output root for the full CV5 zoo (20 runs = 4 models x 5 folds).
ZOO_ROOT="${ROOT}/outputs_night_model_zoo_cv5_top_new_stage4"

# Final submission path (built from the same ZOO_ROOT after stage4 in-place updates).
SUB_CSV="${ZOO_ROOT}/submission_cv5_top_new_stage4_lr_geo8_equal.csv"

echo "=== [1/3] Base CV5 zoo training on top_new_dataset ==="
"${PY}" "${ROOT}/scripts/night_model_zoo_autopilot_mps.py" \
  --base "${BASE_DATASET}" \
  --clean-variant raw \
  --folds-csv "${FOLDS_CSV}" \
  --cv-all \
  --preset overnight_10h \
  --seed 42 \
  --fold-seed 42 \
  --num-workers 0 \
  --device mps \
  --out-root "${ZOO_ROOT}"

echo "=== [2/3] Stage4 finetune in-place for 3x5 runs (skip vit_base) ==="
"${PY}" "${ROOT}/scripts/apply_stage4_to_zoo_runs.py" \
  --zoo-root "${ZOO_ROOT}" \
  --base "${BASE_DATASET}" \
  --clean-variant raw \
  --folds-csv "${FOLDS_CSV}" \
  --device mps \
  --num-workers 0 \
  --mode inplace \
  --include-names "cnn_convnext_small_sam_swa,cnn_effnetv2_s_sam_swa,vit_deit3_small_color_safe" \
  --exclude-names "vit_base_augreg_lite" \
  --epochs 2 \
  --lr 4e-5 \
  --lr-min-scale 0.25 \
  --label-smoothing 0.02 \
  --hard-frac 0.25 \
  --hard-rotate-deg 10 \
  --easy-crop-scale-min 0.30 \
  --easy-crop-scale-max 0.92 \
  --easy-rotate-deg 30 \
  --easy-affine-p 0.55 \
  --easy-dropout-p 0.25 \
  --easy-degrade-p 0.20

echo "=== [3/3] Build submission (CV5 LR/MSE weights + geo8 TTA) ==="
"${PY}" "${ROOT}/scripts/make_submission_from_cv5_all20_lr_tta.py" \
  --base "${BASE_DATASET}" \
  --zoo-root "${ZOO_ROOT}" \
  --device mps \
  --num-workers 0 \
  --tta-mode geo8 \
  --fold-aggregation equal \
  --out-csv "${SUB_CSV}" \
  --save-test-probs

echo "Done."
echo "Artifacts:"
echo "  ${ZOO_ROOT}/autopilot_summary.json"
echo "  ${ZOO_ROOT}/stage4_apply_summary.json"
echo "  ${SUB_CSV}"
echo "  ${ZOO_ROOT}/$(basename "${SUB_CSV}" .csv)_meta.json"
