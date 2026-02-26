#!/bin/zsh
set -euo pipefail

PY="/opt/homebrew/bin/python3.11"
ROOT="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1"

NEW_RUN_ROOT="${ROOT}/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42"
NEW_RUN_F0="${NEW_RUN_ROOT}/runs/001_cnn_convnext_small_sam_swa_f0"
FOLDS_CSV="${ROOT}/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"
DATASET_BASE="${ROOT}/top_new_dataset"

OLD_CKPT_F0="${ROOT}/outputs_night_model_zoo_cv5/runs/001_cnn_convnext_small_sam_swa_f0/best_by_val_loss.pt"

COMPARE_DIR="${ROOT}/outputs_post_tinder_convnext_cv2_compare/analysis_fold0_old_vs_new_on_top_new_dataset"
STAGE4_DIR="${ROOT}/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42_stage4_probe_f0"
STAGE4_COMPARE_DIR="${ROOT}/outputs_post_tinder_convnext_cv2_compare/analysis_fold0_base_vs_stage4_on_top_new_dataset"

if [[ ! -f "${NEW_RUN_F0}/summary.json" ]]; then
  echo "Missing ${NEW_RUN_F0}/summary.json (fold0 not finished yet)." >&2
  exit 1
fi

echo "[1/3] Compare old ConvNeXt checkpoint vs new post-tinder fold0 checkpoint on same top_new_dataset fold0"
"${PY}" "${ROOT}/scripts/analyze_post_clean_delta.py" \
  --base "${DATASET_BASE}" \
  --folds-csv "${FOLDS_CSV}" \
  --fold-idx 0 \
  --old-ckpt "${OLD_CKPT_F0}" \
  --new-ckpt "${NEW_RUN_F0}/best_by_val_loss.pt" \
  --out-dir "${COMPARE_DIR}"

echo "[2/3] Stage4 confidence-aware finetune probe on fold0 (MPS)"
"${PY}" "${ROOT}/scripts/finetune_stage4_confidence_mps.py" \
  --base "${DATASET_BASE}" \
  --clean-variant raw \
  --folds-csv "${FOLDS_CSV}" \
  --fold-idx 0 \
  --seed 42 \
  --device mps \
  --checkpoint "${NEW_RUN_F0}/best_by_val_loss.pt" \
  --model-name convnext_small.fb_in22k_ft_in1k \
  --img-size 224 \
  --use-channels-last \
  --batch-size 16 \
  --num-workers 0 \
  --epochs 3 \
  --lr 4e-5 \
  --label-smoothing 0.02 \
  --hard-frac 0.25 \
  --hard-rotate-deg 10 \
  --easy-crop-scale-min 0.30 \
  --easy-crop-scale-max 0.92 \
  --easy-rotate-deg 30 \
  --easy-affine-p 0.55 \
  --easy-dropout-p 0.25 \
  --easy-degrade-p 0.20 \
  --out-dir "${STAGE4_DIR}"

echo "[3/3] Compare base post-tinder fold0 checkpoint vs stage4 fold0 checkpoint on same top_new_dataset fold0"
"${PY}" "${ROOT}/scripts/analyze_post_clean_delta.py" \
  --base "${DATASET_BASE}" \
  --folds-csv "${FOLDS_CSV}" \
  --fold-idx 0 \
  --old-ckpt "${NEW_RUN_F0}/best_by_val_loss.pt" \
  --new-ckpt "${STAGE4_DIR}/best_by_val_loss.pt" \
  --out-dir "${STAGE4_COMPARE_DIR}"

echo "Done."
echo "Artifacts:"
echo "  ${COMPARE_DIR}/summary.json"
echo "  ${STAGE4_DIR}/summary.json"
echo "  ${STAGE4_COMPARE_DIR}/summary.json"
