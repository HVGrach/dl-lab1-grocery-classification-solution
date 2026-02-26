#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-12}"
OUT_ROOT="$ROOT/dl_lab1/outputs_final3_split_effnetv2_s_classaware_harden"
RUNS_DIR="$OUT_ROOT/runs"
mkdir -p "$RUNS_DIR"

echo "=== Colab Experimental Run: EffNetV2-S + Class-Aware Final Hardening ==="
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Out root: ${OUT_ROOT}"
echo "Experiment: strong no-color aug on easy classes + light rotate-style aug on others in last 4 epochs"

for FOLD in 0 1 2 3; do
  RUN_DIR="$RUNS_DIR/cnn_effnetv2_s_classaware_harden_ls03_swa5_f${FOLD}"
  if [ -f "$RUN_DIR/summary.json" ]; then
    echo "[skip] fold=${FOLD} (summary.json exists)"
    continue
  fi

  mkdir -p "$RUN_DIR"
  CMD=(
    "$PY_BIN" "$ROOT/dl_lab1/scripts/train_onefold_no_color_innov_mps.py"
    --base "$ROOT/dl_lab1/top_new_dataset"
    --clean-variant raw
    --folds-csv "$ROOT/dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"
    --fold-idx "$FOLD"
    --seed 42
    --device "$DEVICE"
    --model-name tf_efficientnetv2_s.in21k_ft_in1k
    --img-size 224
    --batch-size "$BATCH_SIZE"
    --num-workers 0
    --epochs 14
    --stage1-epochs 9
    --warmup-epochs 2
    --lr 0.00025
    --lr-drop-factor 4.0
    --weight-decay 0.0001
    --grad-clip-norm 1.0
    --label-smoothing 0.03
    --mixup-alpha 0.2
    --mixup-prob 0.2
    --cutmix-alpha 1.0
    --cutmix-prob 0.2
    --sam-rho 0.05
    --swa-start-epoch 5
    --class-aware-harden-last-epochs 4
    --class-aware-easy-topk 5
    --out-dir "$RUN_DIR"
  )

  printf '[run] fold=%s\n' "$FOLD"
  printf '      %q ' "${CMD[@]}"; printf '\n'
  "${CMD[@]}"
done

echo
echo "Done. Outputs: $OUT_ROOT"
