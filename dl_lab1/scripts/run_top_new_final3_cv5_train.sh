#!/bin/zsh
set -euo pipefail

PY_BIN="/opt/homebrew/bin/python3.11"
SCRIPT="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/top_new_final3_cv5_train_orchestrator.py"

BASE="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset"
FOLDS_CSV="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"
OUT_ROOT="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_final3_cv5_train"

echo "=== Final3 CV5 Train ==="
echo "Base dataset: ${BASE}"
echo "Folds CSV:     ${FOLDS_CSV}"
echo "Out root:      ${OUT_ROOT}"
echo
echo "Models:        convnext_small,effnetv2_s,deit3_small"
echo "Folds:         0,1,2,3 (deadline mode)"
echo "Epochs:        14 (override), stage1=9"
echo "Profile notes: CNNs use phase1-proven tweaks (label_smoothing=0.03, swa_start_epoch=5); DeiT3 kept conservative."
echo

exec "${PY_BIN}" "${SCRIPT}" \
  --base "${BASE}" \
  --clean-variant raw \
  --folds-csv "${FOLDS_CSV}" \
  --device mps \
  --num-workers 0 \
  --models "convnext_small,effnetv2_s,deit3_small" \
  --folds "0,1,2,3" \
  --epochs-override 14 \
  --stage1-epochs-override 9 \
  --out-root "${OUT_ROOT}" \
  --resume \
  --continue-on-error \
  "$@"
