#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[check] repo root: $ROOT"

need_paths=(
  "$ROOT/dl_lab1/scripts/make_submission_from_cv5_all20_lr_tta.py"
  "$ROOT/dl_lab1/scripts/mixed_old_new_orchestrator_submit.py"
  "$ROOT/dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"
  "$ROOT/dl_lab1/top_new_dataset/train.csv"
  "$ROOT/dl_lab1/top_new_dataset/test.csv"
  "$ROOT/dl_lab1/top_new_dataset/sample_submission.csv"
  "$ROOT/dl_lab1/top_new_dataset/train/train"
  "$ROOT/dl_lab1/top_new_dataset/test_images/test_images"
)

missing=0
for p in "${need_paths[@]}"; do
  if [ ! -e "$p" ]; then
    echo "[missing] $p"
    missing=1
  else
    echo "[ok] $p"
  fi
done

if [ "$missing" -ne 0 ]; then
  echo
  echo "Layout check failed. See data/README.md and weights/README.md."
  exit 2
fi

echo
echo "Layout check passed."
