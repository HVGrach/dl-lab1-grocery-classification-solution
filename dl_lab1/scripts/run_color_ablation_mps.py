#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B test: full aug vs no_color aug on identical training setup.")
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--clean-variant", type=str, default="strict")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-models", type=int, default=1, help="Use 1 to keep only convnext_small for quick A/B.")
    p.add_argument("--ensemble-trials", type=int, default=50)
    p.add_argument(
        "--out-root",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_color_ablation_mps",
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    return p.parse_args()


def run_train(
    python_bin: str,
    train_script: Path,
    base: str,
    clean_variant: str,
    epochs: int,
    folds: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_models: int,
    ensemble_trials: int,
    out_dir: Path,
    aug_profile: str,
) -> Tuple[Dict, float]:
    cmd = [
        python_bin,
        str(train_script),
        "--base",
        base,
        "--clean-variant",
        clean_variant,
        "--epochs",
        str(epochs),
        "--folds",
        str(folds),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--max-models",
        str(max_models),
        "--ensemble-trials",
        str(ensemble_trials),
        "--out-dir",
        str(out_dir),
        "--aug-profile",
        aug_profile,
    ]
    print("\n[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0

    metrics_path = out_dir / "all_model_oof_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics, dt


def extract_single_model_metrics(metrics: Dict) -> Tuple[str, Dict[str, float]]:
    if not metrics:
        raise RuntimeError("Empty metrics")
    alias = next(iter(metrics))
    return alias, metrics[alias]


def main() -> None:
    args = parse_args()
    root = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")
    train_script = root / "train_top1_mps.py"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    out_full = out_root / "full"
    out_no_color = out_root / "no_color"
    out_full.mkdir(parents=True, exist_ok=True)
    out_no_color.mkdir(parents=True, exist_ok=True)

    full_metrics, full_sec = run_train(
        python_bin=args.python_bin,
        train_script=train_script,
        base=args.base,
        clean_variant=args.clean_variant,
        epochs=args.epochs,
        folds=args.folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_models=args.max_models,
        ensemble_trials=args.ensemble_trials,
        out_dir=out_full,
        aug_profile="full",
    )
    no_color_metrics, no_color_sec = run_train(
        python_bin=args.python_bin,
        train_script=train_script,
        base=args.base,
        clean_variant=args.clean_variant,
        epochs=args.epochs,
        folds=args.folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_models=args.max_models,
        ensemble_trials=args.ensemble_trials,
        out_dir=out_no_color,
        aug_profile="no_color",
    )

    alias_full, m_full = extract_single_model_metrics(full_metrics)
    alias_nc, m_nc = extract_single_model_metrics(no_color_metrics)
    if alias_full != alias_nc:
        raise RuntimeError(f"Model mismatch: {alias_full} vs {alias_nc}")

    delta_acc = float(m_nc["acc"] - m_full["acc"])
    delta_f1 = float(m_nc["f1_macro"] - m_full["f1_macro"])
    winner = "no_color" if delta_acc > 0 else ("full" if delta_acc < 0 else "tie")

    summary = {
        "model_alias": alias_full,
        "setup": {
            "epochs": args.epochs,
            "folds": args.folds,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "max_models": args.max_models,
            "clean_variant": args.clean_variant,
        },
        "full": {"metrics": m_full, "seconds": full_sec, "out_dir": str(out_full)},
        "no_color": {"metrics": m_nc, "seconds": no_color_sec, "out_dir": str(out_no_color)},
        "delta_no_color_minus_full": {"acc": delta_acc, "f1_macro": delta_f1},
        "winner_by_acc": winner,
    }

    out_summary = out_root / "color_ablation_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== COLOR ABLATION SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", out_summary)


if __name__ == "__main__":
    main()
