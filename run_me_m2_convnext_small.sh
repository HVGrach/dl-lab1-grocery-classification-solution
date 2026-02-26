#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY_BIN="${PYTHON_BIN:-python3}"
exec "$PY_BIN" "$ROOT/dl_lab1/scripts/top_new_final3_cv5_train_orchestrator.py" \
  --base "$ROOT/dl_lab1/top_new_dataset" \
  --clean-variant raw \
  --folds-csv "$ROOT/dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv" \
  --device mps \
  --num-workers 0 \
  --models convnext_small \
  --folds "0,1,2,3" \
  --epochs-override 14 \
  --stage1-epochs-override 9 \
  --out-root "$ROOT/dl_lab1/outputs_final3_split_convnext_small" \
  --resume \
  --continue-on-error \
  "$@"
