#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
TTA_MODE="${TTA_MODE:-geo8}"

ZOO_ROOT="${ZOO_ROOT:-$ROOT/dl_lab1/outputs_night_model_zoo_cv5}"
RUN_RANKING_CSV="${RUN_RANKING_CSV:-$ZOO_ROOT/run_ranking.csv}"
OUT_CSV="${OUT_CSV:-$ROOT/dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8_equal.csv}"

if [ ! -f "$RUN_RANKING_CSV" ]; then
  echo "[error] run_ranking.csv not found: $RUN_RANKING_CSV"
  echo "Provide ZOO_ROOT/RUN_RANKING_CSV with prepared CV5 run artifacts."
  exit 2
fi

echo "[run] full-CV5 LR(MSE)+TTA submission"
echo "      ZOO_ROOT=$ZOO_ROOT"
echo "      RUN_RANKING_CSV=$RUN_RANKING_CSV"
echo "      OUT_CSV=$OUT_CSV"

exec "$PY_BIN" "$ROOT/dl_lab1/scripts/make_submission_from_cv5_all20_lr_tta.py" \
  --zoo-root "$ZOO_ROOT" \
  --run-ranking-csv "$RUN_RANKING_CSV" \
  --tta-mode "$TTA_MODE" \
  --device "$DEVICE" \
  --out-csv "$OUT_CSV"

