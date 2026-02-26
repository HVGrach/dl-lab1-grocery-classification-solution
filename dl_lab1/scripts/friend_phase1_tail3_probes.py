#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import adaptive_top_new_night_pipeline as adaptive
import night_model_zoo_autopilot_mps as zoo


ROOT = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parallel helper for a friend: run the tail 3 phase1 feature probes on one fold, "
            "starting from the current best-known frontier config."
        )
    )
    p.add_argument("--base", type=str, default=str(ROOT / "top_new_dataset"))
    p.add_argument("--clean-variant", type=str, default="raw", choices=["strict", "aggressive", "raw"])
    p.add_argument(
        "--folds-csv",
        type=str,
        default=str(ROOT / "outputs_post_tinder_convnext_cv2_compare" / "folds_used_top_new_dataset_aligned_hybrid.csv"),
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--python-bin", type=str, default="/opt/homebrew/bin/python3.11")
    p.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "outputs_friend_tail3_phase1_probes"),
        help="Output dir for base+tail3 runs and comparison summary.",
    )

    # Probe profile should match adaptive phase1
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--probe-stage1-epochs", type=int, default=6)
    p.add_argument("--probe-warmup-epochs", type=int, default=2)

    # Current best-known accepted features from live run (can be updated before sending to friend)
    p.add_argument(
        "--base-features",
        type=str,
        default="swa_earlier,label_smoothing_low",
        help="Comma-separated accepted phase1 features that define the current frontier base.",
    )
    p.add_argument(
        "--probe-names",
        type=str,
        default="weighted_sampler_off,lr_down_15,lr_up_10",
        help="Comma-separated probe names to test (default = tail3 in current adaptive feature list).",
    )

    p.add_argument("--skip-base", action="store_true", help="Skip running the frontier base and only run probes.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_args_for_runner(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        python_bin=args.python_bin,
        base=args.base,
        clean_variant=args.clean_variant,
        folds_csv=args.folds_csv,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        continue_on_error=True,
        # thresholds used by adaptive.is_feature_better
        feature_min_acc_gain=8e-4,
        feature_min_f1_gain=8e-4,
        feature_acc_tie=4e-4,
        feature_f1_tie=4e-4,
        feature_min_loss_gain=0.010,
    )


def build_frontier_base(args: argparse.Namespace) -> zoo.Experiment:
    exp_map = adaptive.build_catalog_maps()
    base = exp_map["cnn_convnext_small_sam_swa"]
    base = adaptive.shrink_for_probe(
        base,
        epochs=int(args.probe_epochs),
        stage1_epochs=int(args.probe_stage1_epochs),
        warmup_epochs=int(args.probe_warmup_epochs),
    )
    for feat in parse_csv(args.base_features):
        if not adaptive.can_apply_feature(base, feat):
            raise ValueError(f"Cannot apply base feature '{feat}' to {base.name}")
        base = adaptive.apply_feature_trial(base, feat)
    base = adaptive.ensure_stage_consistency(base)
    return base


def compare_to_base(
    base_row: Dict[str, Any] | None,
    probe_row: Dict[str, Any],
) -> Dict[str, Any]:
    if base_row is None:
        return {"status": "no_base_row"}
    d_acc = float(probe_row.get("val_acc", np.nan)) - float(base_row.get("val_acc", np.nan))
    d_f1 = float(probe_row.get("val_f1_macro", np.nan)) - float(base_row.get("val_f1_macro", np.nan))
    d_loss = float(base_row.get("val_loss", np.nan)) - float(probe_row.get("val_loss", np.nan))
    return {
        "delta_acc": d_acc,
        "delta_f1_macro": d_f1,
        "delta_val_loss_improvement": d_loss,
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    frontier_base = build_frontier_base(args)
    probe_names = parse_csv(args.probe_names)
    if not probe_names:
        raise ValueError("No probe names provided")

    runner_args = build_args_for_runner(args)
    meta: Dict[str, Any] = {
        "task": "friend_phase1_tail3_probes",
        "base_features": parse_csv(args.base_features),
        "probe_names": probe_names,
        "fold_idx": int(args.fold_idx),
        "seed": int(args.seed),
        "device": args.device,
        "dry_run": bool(args.dry_run),
    }
    (out_root / "task_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    base_row: Dict[str, Any] | None = None
    if not args.skip_base:
        base_dir = out_root / "runs" / "000_frontier_base"
        row = adaptive.run_onefold(
            exp=frontier_base,
            run_dir=base_dir,
            args=runner_args,
            fold_idx=int(args.fold_idx),
            run_seed=int(args.seed),
        )
        row["probe_name"] = "frontier_base"
        rows.append(row)
        if row.get("status") == "ok":
            base_row = row

    for idx, probe_name in enumerate(probe_names, start=1):
        if not adaptive.can_apply_feature(frontier_base, probe_name):
            rows.append(
                {
                    "probe_name": probe_name,
                    "status": "skipped_incompatible",
                    "error": "",
                    "fold_idx": int(args.fold_idx),
                }
            )
            continue

        exp = adaptive.apply_feature_trial(frontier_base, probe_name)
        exp = adaptive.ensure_stage_consistency(exp)
        exp = zoo.Experiment(**{**asdict(exp), "name": f"{frontier_base.name}__friend_{probe_name}"})
        run_dir = out_root / "runs" / f"{idx:03d}_{probe_name}"
        row = adaptive.run_onefold(
            exp=exp,
            run_dir=run_dir,
            args=runner_args,
            fold_idx=int(args.fold_idx),
            run_seed=int(args.seed),
        )
        row["probe_name"] = probe_name
        if row.get("status") == "ok" and base_row is not None:
            row.update(compare_to_base(base_row, row))
            accepted, reason = adaptive.is_feature_better(row, base_row, runner_args)
            row["beats_frontier_base"] = bool(accepted)
            row["beats_reason"] = reason
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_root / "friend_tail3_results.csv", index=False)

    summary_rows = []
    for r in rows:
        if r.get("probe_name") == "frontier_base":
            continue
        summary_rows.append(
            {
                "probe_name": r.get("probe_name"),
                "status": r.get("status"),
                "val_acc": r.get("val_acc"),
                "val_f1_macro": r.get("val_f1_macro"),
                "val_loss": r.get("val_loss"),
                "beats_frontier_base": r.get("beats_frontier_base"),
                "beats_reason": r.get("beats_reason", ""),
                "delta_acc": r.get("delta_acc"),
                "delta_f1_macro": r.get("delta_f1_macro"),
                "delta_val_loss_improvement": r.get("delta_val_loss_improvement"),
            }
        )
    result = {
        "status": "ok" if not args.dry_run else "dry_run",
        "frontier_base_config": asdict(frontier_base),
        "frontier_base_metrics": None
        if base_row is None
        else {
            "val_acc": base_row.get("val_acc"),
            "val_f1_macro": base_row.get("val_f1_macro"),
            "val_loss": base_row.get("val_loss"),
        },
        "probe_names": probe_names,
        "results_csv": str(out_root / "friend_tail3_results.csv"),
        "probe_comparison": summary_rows,
    }
    (out_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
