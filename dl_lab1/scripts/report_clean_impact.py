#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_new_metrics(out_root: Path) -> dict:
    summary_path = out_root / "one_fold_ablation_summary.json"
    if summary_path.exists():
        s = load_json(summary_path)
        # Expected when profile=no_color: results.no_color.metrics_on_same_fold
        if "results" in s and "no_color" in s["results"]:
            m = s["results"]["no_color"]["metrics_on_same_fold"]
            return {
                "acc": float(m["acc"]),
                "f1_macro": float(m["f1_macro"]),
                "best_val_loss": float(m["best_val_loss_from_training"]),
                "best_epoch": int(m["best_epoch"]),
                "source": str(summary_path),
            }

    single = out_root / "no_color" / "one_fold_metrics.json"
    if single.exists():
        s = load_json(single)
        m = s["metrics_on_same_fold"]
        return {
            "acc": float(m["acc"]),
            "f1_macro": float(m["f1_macro"]),
            "best_val_loss": float(m["best_val_loss_from_training"]),
            "best_epoch": int(m["best_epoch"]),
            "source": str(single),
        }

    raise FileNotFoundError(
        f"Cannot find one-fold metrics under {out_root}. Expected one_fold_ablation_summary.json or no_color/one_fold_metrics.json"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compare new one-fold metrics vs historical baselines.")
    p.add_argument(
        "--new-out-root",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps",
    )
    p.add_argument(
        "--old-onefold-no-color",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_color_ablation_onefold_mps/no_color/one_fold_metrics.json",
    )
    p.add_argument(
        "--old-overnight-oof",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_overnight_oof_no_color_convnext/convnext_small/metrics.json",
    )
    args = p.parse_args()

    new_m = find_new_metrics(Path(args.new_out_root))
    old_one = load_json(Path(args.old_onefold_no_color))
    old_one_m = old_one["metrics_on_same_fold"]
    old_oof_m = load_json(Path(args.old_overnight_oof))

    report = {
        "new_onefold": new_m,
        "old_onefold_no_color": {
            "acc": float(old_one_m["acc"]),
            "f1_macro": float(old_one_m["f1_macro"]),
            "best_val_loss": float(old_one_m["best_val_loss_from_training"]),
            "best_epoch": int(old_one_m["best_epoch"]),
            "source": str(Path(args.old_onefold_no_color)),
        },
        "old_overnight_oof_5fold": {
            "acc": float(old_oof_m["acc"]),
            "f1_macro": float(old_oof_m["f1_macro"]),
            "source": str(Path(args.old_overnight_oof)),
        },
        "delta_vs_old_onefold_no_color": {
            "acc": float(new_m["acc"] - float(old_one_m["acc"])),
            "f1_macro": float(new_m["f1_macro"] - float(old_one_m["f1_macro"])),
            "best_val_loss": float(new_m["best_val_loss"] - float(old_one_m["best_val_loss_from_training"])),
        },
        "delta_vs_old_overnight_oof_5fold": {
            "acc": float(new_m["acc"] - float(old_oof_m["acc"])),
            "f1_macro": float(new_m["f1_macro"] - float(old_oof_m["f1_macro"])),
        },
        "notes": [
            "One-fold metrics and 5-fold OOF are not identical protocols; use this delta as directional signal.",
            "For strict comparability to old one-fold no_color, keep same fold index/seed/epochs.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
