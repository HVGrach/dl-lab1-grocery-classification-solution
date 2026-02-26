#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"

TOP_NEW_DATASET="${TOP_NEW_DATASET:-$ROOT/dl_lab1/top_new_dataset}"
FOLDS_CSV="${FOLDS_CSV:-$ROOT/dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv}"

OLD_ZOO_ROOT="${OLD_ZOO_ROOT:-$ROOT/dl_lab1/outputs_night_model_zoo_cv5}"
NEW_ZOO_ROOT="${NEW_ZOO_ROOT:-$ROOT/dl_lab1/outputs_final3_common01_3models_zoo_canonical}"
OUT_ROOT="${OUT_ROOT:-$ROOT/dl_lab1/outputs_mixed_old_new_orchestrator_repro}"

for p in "$TOP_NEW_DATASET" "$FOLDS_CSV" "$OLD_ZOO_ROOT" "$NEW_ZOO_ROOT"; do
  if [ ! -e "$p" ]; then
    echo "[error] missing required path: $p"
    exit 2
  fi
done

echo "[run] mixed old+new orchestrator"
echo "      TOP_NEW_DATASET=$TOP_NEW_DATASET"
echo "      FOLDS_CSV=$FOLDS_CSV"
echo "      OLD_ZOO_ROOT=$OLD_ZOO_ROOT"
echo "      NEW_ZOO_ROOT=$NEW_ZOO_ROOT"
echo "      OUT_ROOT=$OUT_ROOT"

exec "$PY_BIN" "$ROOT/dl_lab1/scripts/mixed_old_new_orchestrator_submit.py" \
  --base "$TOP_NEW_DATASET" \
  --folds-csv "$FOLDS_CSV" \
  --old-zoo-root "$OLD_ZOO_ROOT" \
  --new-zoo-root "$NEW_ZOO_ROOT" \
  --device "$DEVICE" \
  --out-root "$OUT_ROOT"

