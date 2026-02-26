#!/bin/zsh
set -euo pipefail

# Adaptive overnight pipeline on top_new_dataset:
# 1) Feature probes on one-fold (ConvNeXt-S) -> accept/reject improvements
# 2) Mini-zoo probe on one-fold -> decide ConvNeXt@256 / heavy models
# 3) Full CV zoo on selected models
# 4) Final LR(MSE)+TTA submission
# 5) Meta-stack OOF benchmark -> conditional meta submission(s)

PY="/opt/homebrew/bin/python3.11"
ROOT="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1"

BASE_DATASET="${ROOT}/top_new_dataset"
FOLDS_CSV="${ROOT}/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"
OUT_ROOT="${ROOT}/outputs_top_new_adaptive_night_run1"

# Optional: add one extra seed for CNNs after first successful night.
# EXTRA_SEEDS="133"
EXTRA_SEEDS=""

echo "=== Adaptive Night Pipeline ==="
echo "Base dataset: ${BASE_DATASET}"
echo "Folds CSV:     ${FOLDS_CSV}"
echo "Out root:      ${OUT_ROOT}"

"${PY}" "${ROOT}/scripts/adaptive_top_new_night_pipeline.py" \
  --base "${BASE_DATASET}" \
  --clean-variant raw \
  --folds-csv "${FOLDS_CSV}" \
  --cv-folds "0,1,2,3,4" \
  --seed 42 \
  --fold-seed 42 \
  --device mps \
  --num-workers 0 \
  --out-root "${OUT_ROOT}" \
  --continue-on-error \
  --probe-convnext256 \
  --probe-effnetv2-m \
  --probe-convnext-base \
  --extra-seeds "${EXTRA_SEEDS}" \
  --extra-seed-target "cnn" \
  --tta-mode geo8 \
  --fold-aggregation equal \
  --meta-methods "logreg,catboost,attn" \
  --meta-max-test-methods 1 \
  --kaggle-auto-submit \
  --kaggle-competition "dl-lab-1-image-classification" \
  --kaggle-config-dir "/Users/fedorgracev/Desktop/NeuralNetwork" \
  --kaggle-submit-threshold 0.97000 \
  --kaggle-max-submits 2 \
  --kaggle-list-after-submit

echo "Done."
echo "Main summary:"
echo "  ${OUT_ROOT}/adaptive_summary.json"
