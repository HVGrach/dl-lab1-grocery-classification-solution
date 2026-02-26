#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import night_model_zoo_autopilot_mps as zoo
import make_submission_from_cv5_all20_lr_tta as lr_submit


ROOT = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")
META_STACK_SCRIPT = ROOT / "scripts" / "meta_stack_cv5_attention.py"
MAKE_SUBMISSION_SCRIPT = ROOT / "scripts" / "make_submission_from_cv5_all20_lr_tta.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Adaptive overnight pipeline on top_new_dataset: "
            "feature probes -> mini-zoo model selection -> full CV zoo -> LR(MSE)+TTA submit -> meta-stack gating."
        )
    )

    p.add_argument("--base", type=str, default=str(ROOT / "top_new_dataset"))
    p.add_argument("--clean-variant", type=str, default="raw", choices=["strict", "aggressive", "raw"])
    p.add_argument(
        "--folds-csv",
        type=str,
        default=str(ROOT / "outputs_post_tinder_convnext_cv2_compare" / "folds_used_top_new_dataset_aligned_hybrid.csv"),
        help="Optional fixed folds mapping for honest comparisons across phases.",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-fold", type=int, default=0)
    p.add_argument("--cv-folds", type=str, default="0,1,2,3,4")

    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--python-bin", type=str, default="/opt/homebrew/bin/python3.11")

    p.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "outputs_top_new_adaptive_night"),
        help="Root dir for all phases.",
    )
    p.add_argument("--resume", action="store_true", help="Reuse completed runs if summary.json exists.")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--continue-on-error", action="store_true", help="Continue grid runs on failures.")
    p.add_argument("--dry-run", action="store_true", help="Only materialize commands/configs; skip execution.")

    # Probe budgets (shorter than full CV)
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--probe-stage1-epochs", type=int, default=6)
    p.add_argument("--probe-warmup-epochs", type=int, default=2)
    p.add_argument("--probe-zoo-epochs", type=int, default=12)
    p.add_argument("--probe-zoo-stage1-epochs", type=int, default=8)
    p.add_argument("--probe-zoo-warmup-epochs", type=int, default=2)

    # Feature acceptance thresholds
    p.add_argument("--feature-min-acc-gain", type=float, default=8e-4)
    p.add_argument("--feature-min-f1-gain", type=float, default=8e-4)
    p.add_argument("--feature-acc-tie", type=float, default=4e-4)
    p.add_argument("--feature-f1-tie", type=float, default=4e-4)
    p.add_argument("--feature-min-loss-gain", type=float, default=0.010)

    # Model probe selection thresholds
    p.add_argument("--probe-ensemble-trials", type=int, default=3000)
    p.add_argument("--probe-ensemble-max-models", type=int, default=4)
    p.add_argument("--heavy-min-ensemble-acc-gain", type=float, default=8e-4)
    p.add_argument("--heavy-min-ensemble-f1-gain", type=float, default=8e-4)
    p.add_argument("--convnext256-single-acc-gain", type=float, default=8e-4)
    p.add_argument("--convnext256-single-f1-gain", type=float, default=8e-4)

    # Full CV zoo + final ensemble settings
    p.add_argument("--full-ensemble-trials", type=int, default=6000)
    p.add_argument("--full-ensemble-max-models", type=int, default=4)
    p.add_argument("--extra-seeds", type=str, default="")
    p.add_argument("--extra-seed-target", type=str, default="cnn")

    # Final inference
    p.add_argument("--tta-mode", type=str, default="geo8", choices=["none", "flip", "geo4", "geo8"])
    p.add_argument("--tta-views", type=int, default=0)
    p.add_argument("--fold-aggregation", type=str, default="equal", choices=["equal", "val_acc"])

    # Meta stacking
    p.add_argument("--meta-methods", type=str, default="logreg,catboost,attn")
    p.add_argument("--meta-oof-acc-promote", type=float, default=5e-4)
    p.add_argument("--meta-oof-f1-promote", type=float, default=5e-4)
    p.add_argument("--meta-max-test-methods", type=int, default=1)
    p.add_argument("--meta-fold-aggregation", type=str, default="equal", choices=["equal", "oof_acc"])

    # Kaggle auto-submit (gated by internal OOF metrics)
    p.add_argument("--kaggle-auto-submit", action="store_true", help="Submit best candidate(s) to Kaggle if OOF threshold is met.")
    p.add_argument("--kaggle-competition", type=str, default="dl-lab-1-image-classification")
    p.add_argument("--kaggle-config-dir", type=str, default=str(ROOT.parent), help="Directory containing kaggle.json.")
    p.add_argument("--kaggle-submit-threshold", type=float, default=0.97000, help="Minimum OOF accuracy to allow auto-submit.")
    p.add_argument("--kaggle-submit-threshold-f1", type=float, default=0.0, help="Optional minimum OOF macro-F1 (0 disables).")
    p.add_argument("--kaggle-max-submits", type=int, default=2, help="Maximum number of auto-submissions in one run.")
    p.add_argument(
        "--kaggle-submit-candidates",
        type=str,
        default="meta,lr",
        help="Priority order of candidate sources to consider: comma-separated from {meta,lr}.",
    )
    p.add_argument("--kaggle-list-after-submit", action="store_true", help="Fetch Kaggle submissions list after auto-submit.")

    # Phase toggles
    p.add_argument("--skip-feature-probes", action="store_true")
    p.add_argument("--skip-model-probe-zoo", action="store_true")
    p.add_argument("--skip-full-cv", action="store_true")
    p.add_argument("--skip-lr-submit", action="store_true")
    p.add_argument("--skip-meta", action="store_true")

    # Candidate families in mini-zoo
    p.add_argument("--probe-convnext256", action="store_true", help="Probe ConvNeXt-S at 256 as replacement candidate.")
    p.add_argument("--probe-convnext-base", action="store_true", help="Probe ConvNeXt-Base@256 as optional heavy model.")
    p.add_argument("--probe-effnetv2-m", action="store_true", help="Probe EffNetV2-M as optional heavy model.")
    p.add_argument("--probe-vit-base", action="store_true", help="Probe ViT-Base augreg again (off by default).")
    p.add_argument("--force-include-models", type=str, default="", help="Comma-separated experiment names to force into final CV.")
    p.add_argument("--force-exclude-models", type=str, default="", help="Comma-separated experiment names to remove from final CV.")

    p.add_argument("--smoke", action="store_true", help="Fast sanity mode for command chain (1 epoch-ish behavior).")
    return p.parse_args()


def parse_int_csv(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_stage_consistency(exp: zoo.Experiment) -> zoo.Experiment:
    epochs = max(2, int(exp.epochs))
    stage1 = max(0, min(int(exp.stage1_epochs), epochs - 1))
    warmup = max(1, min(int(exp.warmup_epochs), epochs))
    swa_start = int(exp.swa_start_epoch)
    swa_start = max(1, min(swa_start, epochs))
    return replace(exp, epochs=epochs, stage1_epochs=stage1, warmup_epochs=warmup, swa_start_epoch=swa_start)


def shrink_for_probe(
    exp: zoo.Experiment,
    epochs: int,
    stage1_epochs: int,
    warmup_epochs: int,
) -> zoo.Experiment:
    out = replace(
        exp,
        epochs=epochs,
        stage1_epochs=stage1_epochs,
        warmup_epochs=warmup_epochs,
    )
    # Move SWA earlier for short probe while keeping a few epochs to average.
    if out.use_swa:
        probe_swa_start = max(2, min(out.epochs - 1, max(out.stage1_epochs, out.epochs - 3)))
        out = replace(out, swa_start_epoch=probe_swa_start)
    return ensure_stage_consistency(out)


def feature_trial_names() -> List[str]:
    # Ordered to test "parameter tuning" before hard disabling regularizers.
    return [
        "sam_rho_up",
        "sam_rho_down",
        "swa_earlier",
        "label_smoothing_low",
        "label_smoothing_high",
        "mix_low",
        "mixup_only",
        "cutmix_only",
        "sam_off",
        "swa_off",
        "weighted_sampler_off",
        "lr_down_15",
        "lr_up_10",
    ]


def can_apply_feature(exp: zoo.Experiment, trial: str) -> bool:
    if trial.startswith("sam_") and trial not in {"sam_off"} and not exp.use_sam:
        return False
    if trial.startswith("swa_") and trial not in {"swa_off"} and not exp.use_swa:
        return False
    if trial in {"mixup_only"} and not exp.use_mixup:
        return False
    if trial in {"cutmix_only"} and not exp.use_cutmix:
        return False
    return True


def apply_feature_trial(exp: zoo.Experiment, trial: str) -> zoo.Experiment:
    e = exp
    if trial == "sam_rho_up":
        e = replace(e, use_sam=True, sam_rho=0.07)
    elif trial == "sam_rho_down":
        e = replace(e, use_sam=True, sam_rho=0.03)
    elif trial == "swa_earlier":
        e = replace(e, use_swa=True, swa_start_epoch=max(2, e.swa_start_epoch - 2))
    elif trial == "label_smoothing_low":
        e = replace(e, label_smoothing=0.03)
    elif trial == "label_smoothing_high":
        e = replace(e, label_smoothing=0.08)
    elif trial == "mix_low":
        e = replace(e, use_mixup=True, mixup_prob=0.10, use_cutmix=True, cutmix_prob=0.10)
    elif trial == "mixup_only":
        e = replace(e, use_mixup=True, mixup_prob=max(0.25, e.mixup_prob), use_cutmix=False, cutmix_prob=0.0)
    elif trial == "cutmix_only":
        e = replace(e, use_cutmix=True, cutmix_prob=max(0.25, e.cutmix_prob), use_mixup=False, mixup_prob=0.0)
    elif trial == "sam_off":
        e = replace(e, use_sam=False)
    elif trial == "swa_off":
        e = replace(e, use_swa=False)
    elif trial == "weighted_sampler_off":
        e = replace(e, use_weighted_sampler=False)
    elif trial == "lr_down_15":
        e = replace(e, lr=e.lr * 0.85)
    elif trial == "lr_up_10":
        e = replace(e, lr=e.lr * 1.10)
    else:
        raise ValueError(f"Unknown feature trial: {trial}")
    return ensure_stage_consistency(e)


def feature_affects_cnn(trial: str) -> bool:
    return True


def make_autopilot_namespace(args: argparse.Namespace) -> SimpleNamespace:
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
    )


def summarize_run_dir(run_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fm = summary.get("final_metrics", {})
        row["val_acc"] = float(fm.get("val_acc", np.nan))
        row["val_f1_macro"] = float(fm.get("val_f1_macro", np.nan))
        row["val_loss"] = float(fm.get("val_loss", np.nan))
        row["val_errors"] = int(fm.get("val_errors", -1))
        row["val_size"] = int(fm.get("val_size", -1))
        row["final_model_selected"] = str(summary.get("final_model_selected", ""))
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        row["img_size"] = int(cfg.get("img_size", -1))
        row["model_name_cfg"] = str(cfg.get("model_name", ""))
    return row


def run_onefold(
    exp: zoo.Experiment,
    run_dir: Path,
    args: argparse.Namespace,
    fold_idx: int,
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ns = make_autopilot_namespace(args)
    if run_seed is not None:
        ns.seed = int(run_seed)
    cmd = zoo.experiment_to_cmd(exp=exp, args=ns, out_dir=run_dir, n_splits=args.n_splits, fold_idx=fold_idx)
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    row: Dict[str, Any] = {
        "name": exp.name,
        "group": exp.group,
        "model_name": exp.model_name,
        "seed": int(run_seed if run_seed is not None else (exp.run_seed if exp.run_seed is not None else args.seed)),
        "fold_idx": int(fold_idx),
        "run_dir": str(run_dir),
    }

    if args.resume and (run_dir / "summary.json").exists():
        row.update({"status": "ok", "error": "", "seconds": 0.0, "reused": 1})
        row.update(summarize_run_dir(run_dir))
        return row

    if args.dry_run:
        row.update({"status": "dry_run", "error": "", "seconds": 0.0, "reused": 0})
        return row

    t0 = time.time()
    status = "ok"
    error = ""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = f"returncode={e.returncode}"
    dt = time.time() - t0

    row.update({"status": status, "error": error, "seconds": float(dt), "reused": 0})
    if status == "ok":
        row.update(summarize_run_dir(run_dir))
    return row


def write_grid_outputs(df: pd.DataFrame, out_root: Path) -> None:
    if df.empty:
        return
    rank_df = df.copy()
    for c in ["val_acc", "val_f1_macro", "val_loss"]:
        if c not in rank_df.columns:
            rank_df[c] = np.nan
    status_order = {"ok": 0, "dry_run": 1, "failed": 2}
    rank_df["_status_ord"] = rank_df["status"].map(status_order).fillna(9)
    rank_df = rank_df.sort_values(["_status_ord", "val_acc", "val_f1_macro"], ascending=[True, False, False]).drop(columns=["_status_ord"])
    rank_df.to_csv(out_root / "run_ranking.csv", index=False)

    if "fold_idx" in rank_df.columns and rank_df["fold_idx"].nunique() > 1:
        ok_df = rank_df[rank_df["status"] == "ok"].copy()
        if not ok_df.empty:
            cv = (
                ok_df.groupby(["name", "group", "model_name", "seed"], as_index=False)
                .agg(
                    folds_run=("fold_idx", "nunique"),
                    val_acc_mean=("val_acc", "mean"),
                    val_acc_std=("val_acc", "std"),
                    val_f1_macro_mean=("val_f1_macro", "mean"),
                    val_f1_macro_std=("val_f1_macro", "std"),
                    val_loss_mean=("val_loss", "mean"),
                    val_loss_std=("val_loss", "std"),
                    total_seconds=("seconds", "sum"),
                )
                .sort_values(["val_acc_mean", "val_f1_macro_mean"], ascending=[False, False])
            )
            cv.to_csv(out_root / "cv_model_summary.csv", index=False)


def run_experiment_grid(
    *,
    phase_name: str,
    out_root: Path,
    experiments: List[zoo.Experiment],
    fold_indices: List[int],
    args: argparse.Namespace,
    ensemble_trials: int,
    ensemble_max_models: int,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "phase_name": phase_name,
        "timestamp": int(time.time()),
        "base": args.base,
        "clean_variant": args.clean_variant,
        "folds_csv": args.folds_csv if args.folds_csv and Path(args.folds_csv).exists() else "",
        "n_splits": int(args.n_splits),
        "fold_seed": int(args.fold_seed),
        "fold_indices": [int(x) for x in fold_indices],
        "seed": int(args.seed),
        "device": args.device,
        "num_workers": int(args.num_workers),
        "num_experiments": int(len(experiments)),
        "num_runs_planned": int(len(experiments) * len(fold_indices)),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
    }
    (out_root / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    records: List[Dict[str, Any]] = []
    run_items: List[Tuple[zoo.Experiment, int]] = []
    for fid in fold_indices:
        for exp in experiments:
            run_items.append((exp, int(fid)))

    log(f"\n=== {phase_name} ===")
    log(json.dumps(run_meta, ensure_ascii=False, indent=2))

    for idx, (exp, fid) in enumerate(run_items, start=1):
        run_dir = runs_root / f"{idx:03d}_{exp.name}_f{fid}"
        log(f"[{idx}/{len(run_items)}] {exp.name} fold={fid} seed={exp.run_seed if exp.run_seed is not None else args.seed}")
        row = run_onefold(exp=exp, run_dir=run_dir, args=args, fold_idx=fid, run_seed=exp.run_seed)
        records.append(row)
        if row.get("status") == "failed" and not args.continue_on_error:
            break

    df = pd.DataFrame(records)
    if not df.empty:
        write_grid_outputs(df, out_root)

    ensemble_summary: Dict[str, Any]
    if args.dry_run:
        ensemble_summary = {"status": "skipped_dry_run"}
    else:
        success_rows = df[df["status"] == "ok"].to_dict(orient="records") if not df.empty and "status" in df.columns else []
        ensemble_summary = zoo.build_ensemble_report(
            success_rows=success_rows,
            out_root=out_root,
            trials=ensemble_trials,
            max_models=ensemble_max_models,
            seed=int(args.seed),
        )

    final_summary = {
        "run_config": run_meta,
        "num_total": int(len(records)),
        "num_success": int(sum(1 for r in records if r.get("status") == "ok")),
        "num_failed": int(sum(1 for r in records if r.get("status") == "failed")),
        "num_dry_run": int(sum(1 for r in records if r.get("status") == "dry_run")),
        "ranking_csv": str(out_root / "run_ranking.csv"),
        "cv_model_summary_csv": str(out_root / "cv_model_summary.csv") if (out_root / "cv_model_summary.csv").exists() else "",
        "ensemble": ensemble_summary,
        "records_preview": records[:3],
    }
    (out_root / "autopilot_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": records, "summary": final_summary}


def metric_tuple(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("val_acc", np.nan)),
        float(row.get("val_f1_macro", np.nan)),
        float(row.get("val_loss", np.nan)),
    )


def is_feature_better(new_row: Dict[str, Any], best_row: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    n_acc, n_f1, n_loss = metric_tuple(new_row)
    b_acc, b_f1, b_loss = metric_tuple(best_row)
    if not np.isfinite(n_acc) or not np.isfinite(n_f1) or not np.isfinite(n_loss):
        return False, "invalid_metrics"
    if not np.isfinite(b_acc) or not np.isfinite(b_f1) or not np.isfinite(b_loss):
        return True, "baseline_missing_metrics"

    d_acc = n_acc - b_acc
    d_f1 = n_f1 - b_f1
    d_loss = b_loss - n_loss  # positive is better

    if d_acc >= args.feature_min_acc_gain:
        return True, f"acc_gain={d_acc:+.6f}"
    if d_acc >= -args.feature_acc_tie and d_f1 >= args.feature_min_f1_gain:
        return True, f"f1_gain_under_acc_tie acc={d_acc:+.6f} f1={d_f1:+.6f}"
    if abs(d_acc) <= args.feature_acc_tie and abs(d_f1) <= args.feature_f1_tie and d_loss >= args.feature_min_loss_gain:
        return True, f"loss_gain_under_metric_tie loss={d_loss:+.6f}"
    return False, f"no_gain acc={d_acc:+.6f} f1={d_f1:+.6f} loss={d_loss:+.6f}"


def build_catalog_maps() -> Dict[str, zoo.Experiment]:
    catalog = zoo.build_catalog()
    all_exps: Dict[str, zoo.Experiment] = {}
    for preset_name, exps in catalog.items():
        for exp in exps:
            if exp.name not in all_exps:
                all_exps[exp.name] = exp
    return all_exps


def get_experiment(name: str, exp_map: Dict[str, zoo.Experiment]) -> zoo.Experiment:
    if name not in exp_map:
        raise KeyError(f"Experiment {name} not found in catalogs")
    return exp_map[name]


def apply_cnn_feature_policy(exp: zoo.Experiment, accepted_trials: Sequence[str]) -> zoo.Experiment:
    e = exp
    if not e.group.startswith("cnn"):
        return e
    for trial in accepted_trials:
        if feature_affects_cnn(trial):
            if can_apply_feature(e, trial):
                e = apply_feature_trial(e, trial)
    return ensure_stage_consistency(e)


def phase1_feature_probes(args: argparse.Namespace, exp_map: Dict[str, zoo.Experiment], phase_root: Path) -> Dict[str, Any]:
    phase_root.mkdir(parents=True, exist_ok=True)
    baseline = get_experiment("cnn_convnext_small_sam_swa", exp_map)
    baseline = shrink_for_probe(
        baseline,
        epochs=args.probe_epochs,
        stage1_epochs=args.probe_stage1_epochs,
        warmup_epochs=args.probe_warmup_epochs,
    )
    if args.smoke:
        baseline = zoo.apply_smoke(baseline)

    fold_idx = int(args.probe_fold)
    rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    baseline_dir = phase_root / "runs" / "000_baseline"
    log("\n=== PHASE 1: FEATURE PROBES (ConvNeXt-S, one-fold) ===")
    best_row = run_onefold(exp=baseline, run_dir=baseline_dir, args=args, fold_idx=fold_idx, run_seed=args.seed)
    best_exp = baseline
    best_name = "baseline"
    rows.append({"probe_name": "baseline", **best_row})

    for idx, trial in enumerate(feature_trial_names(), start=1):
        if not can_apply_feature(best_exp, trial):
            decisions.append({"trial": trial, "status": "skipped_incompatible", "best_after": best_name})
            continue
        cand_exp = apply_feature_trial(best_exp, trial)
        cand_exp = replace(cand_exp, name=f"{best_exp.name}__{trial}")
        if args.smoke:
            cand_exp = zoo.apply_smoke(cand_exp)
        run_dir = phase_root / "runs" / f"{idx:03d}_{trial}"
        row = run_onefold(exp=cand_exp, run_dir=run_dir, args=args, fold_idx=fold_idx, run_seed=args.seed)
        row_wrap = {"probe_name": trial, **row}
        rows.append(row_wrap)

        if row.get("status") != "ok":
            decisions.append({"trial": trial, "status": "failed_or_skipped", "best_after": best_name})
            continue

        accepted, reason = is_feature_better(row, best_row, args)
        decisions.append(
            {
                "trial": trial,
                "status": "accepted" if accepted else "rejected",
                "reason": reason,
                "candidate_metrics": {
                    "val_acc": row.get("val_acc"),
                    "val_f1_macro": row.get("val_f1_macro"),
                    "val_loss": row.get("val_loss"),
                },
                "best_before": best_name,
            }
        )
        if accepted:
            best_row = row
            best_exp = replace(cand_exp, name="cnn_convnext_small_sam_swa")
            best_name = trial

    rows_df = pd.DataFrame(rows)
    if not rows_df.empty:
        rows_df.to_csv(phase_root / "feature_probe_results.csv", index=False)

    accepted_trials = [d["trial"] for d in decisions if d.get("status") == "accepted"]
    summary = {
        "status": "ok" if not args.dry_run else "dry_run",
        "probe_fold": fold_idx,
        "baseline": asdict(baseline),
        "accepted_trials": accepted_trials,
        "best_policy_source": best_name,
        "best_metrics": {
            "val_acc": best_row.get("val_acc"),
            "val_f1_macro": best_row.get("val_f1_macro"),
            "val_loss": best_row.get("val_loss"),
        },
        "decisions": decisions,
        "artifacts": {
            "results_csv": str(phase_root / "feature_probe_results.csv"),
        },
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"accepted_trials": accepted_trials, "best_cnn_probe_exp": best_exp, "summary": summary}


def build_mini_probe_experiments(
    args: argparse.Namespace,
    exp_map: Dict[str, zoo.Experiment],
    accepted_cnn_trials: Sequence[str],
) -> Tuple[List[zoo.Experiment], Dict[str, str]]:
    exps: List[zoo.Experiment] = []
    alias_to_base_name: Dict[str, str] = {}

    # Core 3 (mandatory final candidates)
    convnext = apply_cnn_feature_policy(get_experiment("cnn_convnext_small_sam_swa", exp_map), accepted_cnn_trials)
    effnet = apply_cnn_feature_policy(get_experiment("cnn_effnetv2_s_sam_swa", exp_map), accepted_cnn_trials)
    deit = get_experiment("vit_deit3_small_color_safe", exp_map)

    convnext = shrink_for_probe(convnext, args.probe_zoo_epochs, args.probe_zoo_stage1_epochs, args.probe_zoo_warmup_epochs)
    effnet = shrink_for_probe(effnet, args.probe_zoo_epochs, args.probe_zoo_stage1_epochs, args.probe_zoo_warmup_epochs)
    deit = shrink_for_probe(deit, args.probe_zoo_epochs, args.probe_zoo_stage1_epochs, args.probe_zoo_warmup_epochs)

    if args.smoke:
        convnext = zoo.apply_smoke(convnext)
        effnet = zoo.apply_smoke(effnet)
        deit = zoo.apply_smoke(deit)

    exps.extend([convnext, effnet, deit])
    alias_to_base_name[convnext.name] = "cnn_convnext_small_sam_swa"
    alias_to_base_name[effnet.name] = "cnn_effnetv2_s_sam_swa"
    alias_to_base_name[deit.name] = "vit_deit3_small_color_safe"

    # ConvNeXt-S@256 replacement probe
    if args.probe_convnext256:
        c256 = replace(convnext, name="cnn_convnext_small_sam_swa_img256_probe", img_size=256, batch_size=min(convnext.batch_size, 10))
        c256 = ensure_stage_consistency(c256)
        if args.smoke:
            c256 = zoo.apply_smoke(c256)
        exps.append(c256)
        alias_to_base_name[c256.name] = "cnn_convnext_small_sam_swa"

    # Optional heavy candidates from full preset
    if args.probe_convnext_base:
        cb = apply_cnn_feature_policy(get_experiment("cnn_convnext_base_256", exp_map), accepted_cnn_trials)
        cb = shrink_for_probe(cb, args.probe_zoo_epochs, args.probe_zoo_stage1_epochs, args.probe_zoo_warmup_epochs)
        if args.smoke:
            cb = zoo.apply_smoke(cb)
        exps.append(cb)
        alias_to_base_name[cb.name] = "cnn_convnext_base_256"
    if args.probe_effnetv2_m:
        em = apply_cnn_feature_policy(get_experiment("cnn_effnetv2_m", exp_map), accepted_cnn_trials)
        em = shrink_for_probe(em, args.probe_zoo_epochs, args.probe_zoo_stage1_epochs, args.probe_zoo_warmup_epochs)
        if args.smoke:
            em = zoo.apply_smoke(em)
        exps.append(em)
        alias_to_base_name[em.name] = "cnn_effnetv2_m"
    if args.probe_vit_base:
        vb = get_experiment("vit_base_augreg_lite", exp_map)
        vb = shrink_for_probe(vb, args.probe_zoo_epochs, args.probe_zoo_stage1_epochs, args.probe_zoo_warmup_epochs)
        if args.smoke:
            vb = zoo.apply_smoke(vb)
        exps.append(vb)
        alias_to_base_name[vb.name] = "vit_base_augreg_lite"

    return exps, alias_to_base_name


def _parse_models_cell(cell: str) -> List[str]:
    if not isinstance(cell, str):
        return []
    return [x.strip() for x in cell.split("|") if x.strip()]


def _best_ensemble_row_by_filter(ens_df: pd.DataFrame, pred) -> Optional[pd.Series]:
    if ens_df.empty:
        return None
    tmp = ens_df.copy()
    tmp["models_list"] = tmp["models"].map(_parse_models_cell)
    tmp = tmp[tmp["models_list"].map(pred)]
    if tmp.empty:
        return None
    tmp = tmp.sort_values(["acc", "f1_macro"], ascending=False)
    return tmp.iloc[0]


def phase2_model_probe_zoo(
    args: argparse.Namespace,
    exp_map: Dict[str, zoo.Experiment],
    accepted_cnn_trials: Sequence[str],
    phase_root: Path,
) -> Dict[str, Any]:
    exps, alias_to_base = build_mini_probe_experiments(args=args, exp_map=exp_map, accepted_cnn_trials=accepted_cnn_trials)
    result = run_experiment_grid(
        phase_name="PHASE 2: MINI-ZOO MODEL PROBE",
        out_root=phase_root,
        experiments=exps,
        fold_indices=[int(args.probe_fold)],
        args=args,
        ensemble_trials=int(args.probe_ensemble_trials),
        ensemble_max_models=int(args.probe_ensemble_max_models),
    )

    if args.dry_run:
        summary = {
            "status": "dry_run",
            "selected_models": ["cnn_convnext_small_sam_swa", "cnn_effnetv2_s_sam_swa", "vit_deit3_small_color_safe"],
            "convnext_use_img256": False,
            "included_optional_models": [],
            "notes": "No metric-based selection in dry-run mode.",
        }
        (phase_root / "model_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    rank_path = phase_root / "run_ranking.csv"
    ens_path = phase_root / "ensemble_candidates.csv"
    rank_df = pd.read_csv(rank_path) if rank_path.exists() else pd.DataFrame()
    rank_df = rank_df[rank_df["status"] == "ok"].copy() if not rank_df.empty else rank_df
    ens_df = pd.read_csv(ens_path) if ens_path.exists() else pd.DataFrame()

    core_names = {"cnn_convnext_small_sam_swa", "cnn_effnetv2_s_sam_swa", "vit_deit3_small_color_safe"}
    selected_models = set(core_names)
    convnext_use_img256 = False
    evidence: Dict[str, Any] = {}

    # Replacement decision: ConvNeXt-S@256 vs current ConvNeXt-S single-metric on same fold.
    if "cnn_convnext_small_sam_swa_img256_probe" in set(rank_df.get("name", [])):
        base_row = rank_df[rank_df["name"] == "cnn_convnext_small_sam_swa"].sort_values(["val_acc", "val_f1_macro"], ascending=False)
        c256_row = rank_df[rank_df["name"] == "cnn_convnext_small_sam_swa_img256_probe"].sort_values(["val_acc", "val_f1_macro"], ascending=False)
        if not base_row.empty and not c256_row.empty:
            b = base_row.iloc[0]
            c = c256_row.iloc[0]
            d_acc = float(c["val_acc"]) - float(b["val_acc"])
            d_f1 = float(c["val_f1_macro"]) - float(b["val_f1_macro"])
            evidence["convnext256_vs_224_single"] = {
                "base": {"acc": float(b["val_acc"]), "f1_macro": float(b["val_f1_macro"]), "val_loss": float(b["val_loss"])},
                "img256": {"acc": float(c["val_acc"]), "f1_macro": float(c["val_f1_macro"]), "val_loss": float(c["val_loss"])},
                "delta_acc": d_acc,
                "delta_f1_macro": d_f1,
            }
            if d_acc >= args.convnext256_single_acc_gain or (d_acc >= -args.feature_acc_tie and d_f1 >= args.convnext256_single_f1_gain):
                convnext_use_img256 = True

    # Heavy inclusion decision based on mini-zoo ensemble gain over core-only best.
    if not ens_df.empty and "models" in ens_df.columns:
        core_only_best = _best_ensemble_row_by_filter(ens_df, lambda ms: all(m in core_names for m in ms))
        overall_best = _best_ensemble_row_by_filter(ens_df, lambda ms: True)
        if core_only_best is not None:
            evidence["core_only_best_ensemble"] = {
                "models": _parse_models_cell(str(core_only_best["models"])),
                "acc": float(core_only_best["acc"]),
                "f1_macro": float(core_only_best["f1_macro"]),
            }
        if overall_best is not None:
            evidence["overall_best_ensemble"] = {
                "models": _parse_models_cell(str(overall_best["models"])),
                "acc": float(overall_best["acc"]),
                "f1_macro": float(overall_best["f1_macro"]),
            }

        optional_candidates = ["cnn_convnext_base_256", "cnn_effnetv2_m", "vit_base_augreg_lite"]
        if convnext_use_img256:
            core_names_probe = (core_names - {"cnn_convnext_small_sam_swa"}) | {"cnn_convnext_small_sam_swa_img256_probe"}
        else:
            core_names_probe = set(core_names)

        core_best_for_current_convnext = _best_ensemble_row_by_filter(ens_df, lambda ms: all(m in core_names_probe for m in ms))
        for cand in optional_candidates:
            if cand not in set(rank_df.get("name", [])):
                continue
            best_with_cand = _best_ensemble_row_by_filter(ens_df, lambda ms, cand=cand: cand in ms)
            if best_with_cand is None or core_best_for_current_convnext is None:
                continue
            d_acc = float(best_with_cand["acc"]) - float(core_best_for_current_convnext["acc"])
            d_f1 = float(best_with_cand["f1_macro"]) - float(core_best_for_current_convnext["f1_macro"])
            evidence[f"{cand}_ensemble_delta_vs_core"] = {
                "best_with_candidate": {
                    "models": _parse_models_cell(str(best_with_cand["models"])),
                    "acc": float(best_with_cand["acc"]),
                    "f1_macro": float(best_with_cand["f1_macro"]),
                },
                "core_reference": {
                    "models": _parse_models_cell(str(core_best_for_current_convnext["models"])),
                    "acc": float(core_best_for_current_convnext["acc"]),
                    "f1_macro": float(core_best_for_current_convnext["f1_macro"]),
                },
                "delta_acc": d_acc,
                "delta_f1_macro": d_f1,
            }
            if d_acc >= args.heavy_min_ensemble_acc_gain or (d_acc >= -args.feature_acc_tie and d_f1 >= args.heavy_min_ensemble_f1_gain):
                selected_models.add(cand)

    if convnext_use_img256:
        selected_models.discard("cnn_convnext_small_sam_swa")
        selected_models.add("cnn_convnext_small_sam_swa_img256_probe")

    for name in parse_str_csv(args.force_include_models):
        selected_models.add(name)
    for name in parse_str_csv(args.force_exclude_models):
        selected_models.discard(name)

    selected_models_sorted = sorted(selected_models)
    included_optional = sorted([m for m in selected_models_sorted if m not in core_names and m != "cnn_convnext_small_sam_swa_img256_probe"])
    summary = {
        "status": "ok",
        "selected_models": selected_models_sorted,
        "convnext_use_img256": bool(convnext_use_img256),
        "included_optional_models": included_optional,
        "accepted_cnn_trials": list(accepted_cnn_trials),
        "evidence": evidence,
    }
    (phase_root / "model_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_final_experiments(
    args: argparse.Namespace,
    exp_map: Dict[str, zoo.Experiment],
    accepted_cnn_trials: Sequence[str],
    model_selection: Dict[str, Any],
) -> List[zoo.Experiment]:
    selected_names = list(model_selection.get("selected_models") or [])
    if not selected_names:
        selected_names = ["cnn_convnext_small_sam_swa", "cnn_effnetv2_s_sam_swa", "vit_deit3_small_color_safe"]

    final_exps: List[zoo.Experiment] = []
    for name in selected_names:
        if name == "cnn_convnext_small_sam_swa_img256_probe":
            base = apply_cnn_feature_policy(get_experiment("cnn_convnext_small_sam_swa", exp_map), accepted_cnn_trials)
            exp = replace(base, name="cnn_convnext_small_sam_swa_img256", img_size=256, batch_size=min(base.batch_size, 10))
        else:
            exp = get_experiment(name, exp_map)
            exp = apply_cnn_feature_policy(exp, accepted_cnn_trials)
        exp = ensure_stage_consistency(exp)
        final_exps.append(exp)

    final_exps = zoo.expand_seed_clones(
        experiments=final_exps,
        base_seed=int(args.seed),
        extra_seeds_raw=args.extra_seeds,
        target_raw=args.extra_seed_target,
    )
    return final_exps


def run_lr_submission(args: argparse.Namespace, zoo_root: Path, phase_root: Path) -> Dict[str, Any]:
    phase_root.mkdir(parents=True, exist_ok=True)
    out_csv = phase_root / "submission_adaptive_lr_mse_geo.csv"
    cmd = [
        args.python_bin,
        str(MAKE_SUBMISSION_SCRIPT),
        "--base",
        args.base,
        "--zoo-root",
        str(zoo_root),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--tta-mode",
        args.tta_mode,
        "--tta-views",
        str(int(args.tta_views)),
        "--fold-aggregation",
        args.fold_aggregation,
        "--out-csv",
        str(out_csv),
        "--save-test-probs",
    ]
    (phase_root / "cmd_lr_submit.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    if args.dry_run:
        return {"status": "dry_run", "cmd": cmd, "out_csv": str(out_csv)}
    subprocess.run(cmd, check=True)
    meta_path = out_csv.parent / f"{out_csv.stem}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {"status": "ok", "cmd": cmd, "out_csv": str(out_csv), "meta_json": str(meta_path), "meta": meta}


def compute_exact_lr_cv_oof_metrics(zoo_root: Path) -> Dict[str, Any]:
    rr = pd.read_csv(zoo_root / "run_ranking.csv")
    rr = rr[rr["status"] == "ok"].copy()
    rr["fold_idx"] = rr["fold_idx"].astype(int)
    if rr.empty:
        raise RuntimeError("No successful runs in run_ranking.csv for LR OOF computation")

    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    per_fold: Dict[str, Any] = {}

    for fid in sorted(rr["fold_idx"].unique().tolist()):
        fr = rr[rr["fold_idx"] == fid].sort_values("name")
        prob_list: List[np.ndarray] = []
        y_ref: Optional[np.ndarray] = None
        ids_ref: Optional[List[str]] = None
        for _, row in fr.iterrows():
            run_dir = Path(str(row["run_dir"]))
            probs = np.load(run_dir / "val_probs.npy")
            y = np.load(run_dir / "val_labels.npy").astype(np.int64)
            ids = pd.read_csv(run_dir / "val_predictions.csv")["image_id"].astype(str).tolist()
            if y_ref is None:
                y_ref = y
                ids_ref = ids
            else:
                if not np.array_equal(y_ref, y):
                    raise RuntimeError(f"Fold {fid}: val_labels mismatch")
                if ids_ref != ids:
                    raise RuntimeError(f"Fold {fid}: val image order mismatch")
            prob_list.append(probs)
        assert y_ref is not None
        w, fold_metric = lr_submit.fit_lr_weights(prob_list, y_ref)
        prob_stack = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list], axis=-1)
        blend = np.tensordot(prob_stack, w, axes=([2], [0])).astype(np.float32)
        all_probs.append(blend)
        all_y.append(y_ref)
        per_fold[str(fid)] = {
            "weights": [float(x) for x in w],
            "acc": float(fold_metric["acc"]),
            "f1_macro": float(fold_metric["f1_macro"]),
            "n": int(len(y_ref)),
        }

    y_all = np.concatenate(all_y, axis=0)
    p_all = np.concatenate(all_probs, axis=0)
    pred = p_all.argmax(1)
    metrics = {
        "acc": float(accuracy_score(y_all, pred)),
        "f1_macro": float(f1_score(y_all, pred, average="macro")),
        "n": int(len(y_all)),
    }
    return {"overall": metrics, "per_fold": per_fold}


def run_meta_oof_and_conditional_submit(
    args: argparse.Namespace,
    zoo_root: Path,
    lr_oof: Dict[str, Any],
    phase_root: Path,
) -> Dict[str, Any]:
    phase_root.mkdir(parents=True, exist_ok=True)
    oof_dir = phase_root / "oof_benchmark"
    final_dir = phase_root / "final_submit"
    oof_dir.mkdir(parents=True, exist_ok=True)

    cmd_oof = [
        args.python_bin,
        str(META_STACK_SCRIPT),
        "--base",
        args.base,
        "--zoo-root",
        str(zoo_root),
        "--out-dir",
        str(oof_dir),
        "--seed",
        str(int(args.seed)),
        "--device",
        args.device,
        "--num-workers",
        str(int(args.num_workers)),
        "--tta-mode",
        args.tta_mode,
        "--tta-views",
        str(int(args.tta_views)),
        "--methods",
        args.meta_methods,
        "--fold-aggregation",
        args.meta_fold_aggregation,
        "--skip-test-infer",
    ]
    (phase_root / "cmd_meta_oof.txt").write_text(" ".join(cmd_oof) + "\n", encoding="utf-8")

    if args.dry_run:
        return {"status": "dry_run", "cmd_oof": cmd_oof}

    subprocess.run(cmd_oof, check=True)
    ranking_path = oof_dir / "meta_methods_oof_ranking.csv"
    if not ranking_path.exists():
        return {"status": "no_ranking", "oof_dir": str(oof_dir)}

    meta_rank = pd.read_csv(ranking_path).sort_values(["acc", "f1_macro"], ascending=False)
    meta_rows = meta_rank.to_dict(orient="records")

    lr_acc = float(lr_oof["overall"]["acc"])
    lr_f1 = float(lr_oof["overall"]["f1_macro"])
    promote_methods: List[str] = []
    promote_reason = "no_meta_gain"

    if meta_rows:
        best = meta_rows[0]
        d_acc = float(best["acc"]) - lr_acc
        d_f1 = float(best["f1_macro"]) - lr_f1
        if d_acc >= args.meta_oof_acc_promote or (d_acc >= -args.feature_acc_tie and d_f1 >= args.meta_oof_f1_promote):
            promote_reason = f"best_meta_gain acc={d_acc:+.6f} f1={d_f1:+.6f}"
            top_acc = float(best["acc"])
            top_f1 = float(best["f1_macro"])
            for row in meta_rows:
                if len(promote_methods) >= max(1, int(args.meta_max_test_methods)):
                    break
                if (top_acc - float(row["acc"]) <= args.feature_acc_tie) and (top_f1 - float(row["f1_macro"]) <= args.feature_f1_tie):
                    promote_methods.append(str(row["method"]))

    result: Dict[str, Any] = {
        "status": "oof_only_done",
        "oof_dir": str(oof_dir),
        "meta_oof_ranking_csv": str(ranking_path),
        "lr_oof_baseline": lr_oof,
        "promote_methods": promote_methods,
        "promote_reason": promote_reason,
    }

    if not promote_methods:
        (phase_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    final_dir.mkdir(parents=True, exist_ok=True)
    cmd_final = [
        args.python_bin,
        str(META_STACK_SCRIPT),
        "--base",
        args.base,
        "--zoo-root",
        str(zoo_root),
        "--out-dir",
        str(final_dir),
        "--seed",
        str(int(args.seed)),
        "--device",
        args.device,
        "--num-workers",
        str(int(args.num_workers)),
        "--batch-size",
        "16",
        "--tta-mode",
        args.tta_mode,
        "--tta-views",
        str(int(args.tta_views)),
        "--methods",
        ",".join(promote_methods),
        "--fold-aggregation",
        args.meta_fold_aggregation,
        "--reuse-test-cache",
        "--save-test-cache",
    ]
    (phase_root / "cmd_meta_final.txt").write_text(" ".join(cmd_final) + "\n", encoding="utf-8")
    subprocess.run(cmd_final, check=True)
    final_summary_path = final_dir / "summary.json"
    if final_summary_path.exists():
        result["final_summary"] = json.loads(final_summary_path.read_text(encoding="utf-8"))
    result["status"] = "meta_submit_done"
    result["final_dir"] = str(final_dir)
    (phase_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def collect_kaggle_submit_candidates(
    *,
    args: argparse.Namespace,
    lr_submit_info: Dict[str, Any],
    lr_oof: Dict[str, Any],
    meta_phase: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    # LR(MSE) baseline candidate
    if lr_submit_info.get("status") == "ok" and lr_submit_info.get("out_csv") and lr_oof:
        overall = lr_oof.get("overall", {})
        candidates.append(
            {
                "source": "lr",
                "name": "lr_mse",
                "csv_path": str(lr_submit_info["out_csv"]),
                "oof_acc": float(overall.get("acc", np.nan)),
                "oof_f1_macro": float(overall.get("f1_macro", np.nan)),
            }
        )

    # Meta candidates (only those actually built in final submit phase)
    if isinstance(meta_phase, dict):
        final_summary = meta_phase.get("final_summary", {})
        if isinstance(final_summary, dict):
            submissions = final_summary.get("submissions", {}) or {}
            results = final_summary.get("results", {}) or {}
            for method, csv_path in submissions.items():
                rec = results.get(method, {})
                oof = rec.get("oof_metrics", {}) if isinstance(rec, dict) else {}
                candidates.append(
                    {
                        "source": "meta",
                        "name": f"meta_{method}",
                        "method": method,
                        "csv_path": str(csv_path),
                        "oof_acc": float(oof.get("acc", np.nan)),
                        "oof_f1_macro": float(oof.get("f1_macro", np.nan)),
                    }
                )

    # Filter invalid/duplicate files.
    seen_paths = set()
    valid: List[Dict[str, Any]] = []
    for c in candidates:
        p = str(c.get("csv_path", ""))
        if not p or p in seen_paths:
            continue
        if not Path(p).exists():
            continue
        if not np.isfinite(float(c.get("oof_acc", np.nan))):
            continue
        seen_paths.add(p)
        valid.append(c)

    # Priority: source order then best OOF metrics.
    prio = {name.strip(): i for i, name in enumerate(parse_str_csv(args.kaggle_submit_candidates))}
    valid = sorted(
        valid,
        key=lambda c: (
            prio.get(str(c.get("source", "")), 999),
            -float(c.get("oof_acc", -1e9)),
            -float(c.get("oof_f1_macro", -1e9)),
        ),
    )
    return valid


def kaggle_candidate_passes_threshold(c: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    acc = float(c.get("oof_acc", np.nan))
    f1m = float(c.get("oof_f1_macro", np.nan))
    if not np.isfinite(acc):
        return False, "invalid_oof_acc"
    if acc < float(args.kaggle_submit_threshold):
        return False, f"oof_acc={acc:.5f} < {args.kaggle_submit_threshold:.5f}"
    thr_f1 = float(args.kaggle_submit_threshold_f1)
    if thr_f1 > 0.0 and (not np.isfinite(f1m) or f1m < thr_f1):
        return False, f"oof_f1={f1m:.5f} < {thr_f1:.5f}"
    return True, f"oof_acc={acc:.5f}, oof_f1={f1m:.5f}"


def submit_to_kaggle(
    *,
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    rank_idx: int,
) -> Dict[str, Any]:
    csv_path = Path(str(candidate["csv_path"]))
    msg = (
        f"adaptive-night {candidate['name']} "
        f"oof={float(candidate['oof_acc']):.5f}/{float(candidate.get('oof_f1_macro', np.nan)):.5f}"
    )
    # Kaggle message length is limited; keep it short and deterministic.
    msg = msg[:80]

    env = os.environ.copy()
    if str(args.kaggle_config_dir).strip():
        env["KAGGLE_CONFIG_DIR"] = str(args.kaggle_config_dir)

    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        args.kaggle_competition,
        "-f",
        str(csv_path),
        "-m",
        msg,
    ]

    rec: Dict[str, Any] = {
        "rank_idx": int(rank_idx),
        "candidate": candidate,
        "cmd": cmd,
        "message": msg,
        "status": "pending",
        "kaggle_config_dir": env.get("KAGGLE_CONFIG_DIR", ""),
    }
    if args.dry_run:
        rec["status"] = "dry_run"
        return rec

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
        rec["status"] = "ok"
        rec["seconds"] = float(time.time() - t0)
        rec["stdout"] = proc.stdout[-4000:] if proc.stdout else ""
        rec["stderr"] = proc.stderr[-4000:] if proc.stderr else ""
    except subprocess.CalledProcessError as e:
        rec["status"] = "failed"
        rec["seconds"] = float(time.time() - t0)
        rec["returncode"] = int(e.returncode)
        rec["stdout"] = (e.stdout or "")[-4000:]
        rec["stderr"] = (e.stderr or "")[-4000:]
    return rec


def maybe_kaggle_auto_submit(
    *,
    args: argparse.Namespace,
    out_root: Path,
    lr_submit_info: Dict[str, Any],
    lr_oof: Dict[str, Any],
    meta_phase: Dict[str, Any],
) -> Dict[str, Any]:
    phase_root = out_root / "phase5_kaggle_submit"
    phase_root.mkdir(parents=True, exist_ok=True)

    if not args.kaggle_auto_submit:
        result = {"status": "disabled"}
        (phase_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    candidates = collect_kaggle_submit_candidates(args=args, lr_submit_info=lr_submit_info, lr_oof=lr_oof, meta_phase=meta_phase)
    evaluated: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    for c in candidates:
        ok, reason = kaggle_candidate_passes_threshold(c, args)
        row = {**c, "threshold_pass": bool(ok), "threshold_reason": reason}
        evaluated.append(row)
        if ok:
            accepted.append(row)

    submit_records: List[Dict[str, Any]] = []
    for i, cand in enumerate(accepted[: max(0, int(args.kaggle_max_submits))], start=1):
        rec = submit_to_kaggle(args=args, candidate=cand, rank_idx=i)
        submit_records.append(rec)
        # Save per-submit logs immediately for postmortem robustness.
        (phase_root / f"submit_{i:02d}_{cand['name']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    submissions_list_rec: Dict[str, Any] | None = None
    if args.kaggle_list_after_submit and submit_records and (not args.dry_run):
        env = os.environ.copy()
        if str(args.kaggle_config_dir).strip():
            env["KAGGLE_CONFIG_DIR"] = str(args.kaggle_config_dir)
        cmd_list = ["kaggle", "competitions", "submissions", "-c", args.kaggle_competition, "-v"]
        try:
            proc = subprocess.run(cmd_list, check=True, text=True, capture_output=True, env=env)
            submissions_list_rec = {
                "status": "ok",
                "cmd": cmd_list,
                "stdout": proc.stdout[-12000:] if proc.stdout else "",
                "stderr": proc.stderr[-4000:] if proc.stderr else "",
            }
            (phase_root / "kaggle_submissions_list.txt").write_text(proc.stdout or "", encoding="utf-8")
        except subprocess.CalledProcessError as e:
            submissions_list_rec = {
                "status": "failed",
                "cmd": cmd_list,
                "returncode": int(e.returncode),
                "stdout": (e.stdout or "")[-12000:],
                "stderr": (e.stderr or "")[-4000:],
            }

    result = {
        "status": "ok",
        "kaggle_auto_submit": True,
        "competition": args.kaggle_competition,
        "thresholds": {
            "oof_acc_min": float(args.kaggle_submit_threshold),
            "oof_f1_min": float(args.kaggle_submit_threshold_f1),
        },
        "max_submits": int(args.kaggle_max_submits),
        "candidates_evaluated": evaluated,
        "submits_attempted": len(submit_records),
        "submit_records": submit_records,
        "submissions_list": submissions_list_rec,
    }
    (phase_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def cv_fold_indices(args: argparse.Namespace) -> List[int]:
    if args.cv_folds.strip():
        return sorted(set(parse_int_csv(args.cv_folds)))
    return list(range(int(args.n_splits)))


def persist_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    persist_json(out_root / "adaptive_run_config.json", vars(args))

    exp_map = build_catalog_maps()
    phase1_root = out_root / "phase1_feature_probes"
    phase2_root = out_root / "phase2_model_probe_zoo"
    phase3_root = out_root / "phase3_full_cv_zoo"
    phase4_root = out_root / "phase4_meta"
    submit_root = out_root / "phase3_lr_submit"

    adaptive_summary: Dict[str, Any] = {
        "config": vars(args),
        "phases": {},
        "recommendation": {},
    }

    # Phase 1: CNN feature probes
    accepted_cnn_trials: List[str] = []
    if args.skip_feature_probes:
        log("[skip] phase1 feature probes")
        adaptive_summary["phases"]["phase1_feature_probes"] = {"status": "skipped"}
    else:
        p1 = phase1_feature_probes(args=args, exp_map=exp_map, phase_root=phase1_root)
        accepted_cnn_trials = list(p1.get("accepted_trials", []))
        adaptive_summary["phases"]["phase1_feature_probes"] = p1["summary"]

    # Phase 2: mini-zoo probe for replacement / heavy models
    model_selection: Dict[str, Any]
    if args.skip_model_probe_zoo:
        log("[skip] phase2 mini-zoo model probe")
        model_selection = {
            "status": "skipped",
            "selected_models": ["cnn_convnext_small_sam_swa", "cnn_effnetv2_s_sam_swa", "vit_deit3_small_color_safe"],
            "convnext_use_img256": False,
            "included_optional_models": [],
        }
        adaptive_summary["phases"]["phase2_model_probe_zoo"] = model_selection
    else:
        model_selection = phase2_model_probe_zoo(
            args=args,
            exp_map=exp_map,
            accepted_cnn_trials=accepted_cnn_trials,
            phase_root=phase2_root,
        )
        adaptive_summary["phases"]["phase2_model_probe_zoo"] = model_selection

    # Phase 3: full CV zoo on selected models
    final_exps = build_final_experiments(
        args=args,
        exp_map=exp_map,
        accepted_cnn_trials=accepted_cnn_trials,
        model_selection=model_selection,
    )
    adaptive_summary["phases"]["phase3_plan"] = {
        "cv_folds": cv_fold_indices(args),
        "final_experiments": [asdict(e) for e in final_exps],
        "num_final_experiments": len(final_exps),
        "num_final_runs": len(final_exps) * len(cv_fold_indices(args)),
    }

    if args.skip_full_cv:
        log("[skip] phase3 full CV zoo")
        adaptive_summary["phases"]["phase3_full_cv_zoo"] = {"status": "skipped"}
        persist_json(out_root / "adaptive_summary.json", adaptive_summary)
        return

    p3 = run_experiment_grid(
        phase_name="PHASE 3: FULL CV ZOO (adaptive-selected)",
        out_root=phase3_root,
        experiments=final_exps,
        fold_indices=cv_fold_indices(args),
        args=args,
        ensemble_trials=int(args.full_ensemble_trials),
        ensemble_max_models=int(args.full_ensemble_max_models),
    )
    adaptive_summary["phases"]["phase3_full_cv_zoo"] = p3["summary"]

    if args.dry_run:
        adaptive_summary["phases"]["phase3_lr_submit"] = {"status": "skipped_dry_run"}
        adaptive_summary["phases"]["phase4_meta"] = {"status": "skipped_dry_run"}
        adaptive_summary["phases"]["phase5_kaggle_submit"] = {"status": "skipped_dry_run"}
        persist_json(out_root / "adaptive_summary.json", adaptive_summary)
        return

    # LR(MSE)+TTA submission (base final)
    lr_submit_info: Dict[str, Any] = {"status": "skipped"}
    lr_oof: Dict[str, Any] = {}
    if args.skip_lr_submit:
        log("[skip] LR submission phase")
    else:
        log("\n=== PHASE 4A: LR(MSE)+TTA submission ===")
        lr_submit_info = run_lr_submission(args=args, zoo_root=phase3_root, phase_root=submit_root)
        adaptive_summary["phases"]["phase3_lr_submit"] = {
            k: v for k, v in lr_submit_info.items() if k != "meta"
        }
        lr_oof = compute_exact_lr_cv_oof_metrics(phase3_root)
        persist_json(submit_root / "lr_cv_oof_metrics.json", lr_oof)
        adaptive_summary["phases"]["phase3_lr_submit"]["lr_cv_oof_metrics"] = lr_oof

    # Meta stacking OOF benchmark + conditional test inference
    if args.skip_meta:
        log("[skip] meta stacking phase")
        adaptive_summary["phases"]["phase4_meta"] = {"status": "skipped"}
    else:
        if not lr_oof:
            lr_oof = compute_exact_lr_cv_oof_metrics(phase3_root)
        log("\n=== PHASE 4B: Meta-stack OOF benchmark + conditional submit ===")
        meta_info = run_meta_oof_and_conditional_submit(args=args, zoo_root=phase3_root, lr_oof=lr_oof, phase_root=phase4_root)
        adaptive_summary["phases"]["phase4_meta"] = meta_info

    # Optional Kaggle auto-submit (gated by internal OOF threshold)
    kaggle_phase = maybe_kaggle_auto_submit(
        args=args,
        out_root=out_root,
        lr_submit_info=lr_submit_info,
        lr_oof=lr_oof,
        meta_phase=adaptive_summary["phases"].get("phase4_meta", {}),
    )
    adaptive_summary["phases"]["phase5_kaggle_submit"] = kaggle_phase

    # Final recommendation
    recommendation: Dict[str, Any] = {
        "base_lr_submission_csv": lr_submit_info.get("out_csv", ""),
        "keep_as_default": "lr_mse_tta" if not adaptive_summary["phases"].get("phase4_meta", {}).get("promote_methods") else "compare_lr_vs_meta",
        "accepted_cnn_trials": accepted_cnn_trials,
        "selected_model_families": model_selection.get("selected_models", []),
        "kaggle_auto_submit": {
            "enabled": bool(args.kaggle_auto_submit),
            "threshold_oof_acc": float(args.kaggle_submit_threshold),
            "threshold_oof_f1": float(args.kaggle_submit_threshold_f1),
            "attempted": int(kaggle_phase.get("submits_attempted", 0)) if isinstance(kaggle_phase, dict) else 0,
        },
    }
    meta_phase = adaptive_summary["phases"].get("phase4_meta", {})
    if isinstance(meta_phase, dict) and meta_phase.get("promote_methods"):
        recommendation["meta_candidates"] = meta_phase.get("promote_methods", [])
        final_summary = meta_phase.get("final_summary", {})
        if isinstance(final_summary, dict):
            recommendation["meta_submission_csvs"] = final_summary.get("submissions", {})

    adaptive_summary["recommendation"] = recommendation
    persist_json(out_root / "adaptive_summary.json", adaptive_summary)

    log("\n=== ADAPTIVE NIGHT PIPELINE DONE ===")
    log(json.dumps(recommendation, ensure_ascii=False, indent=2))
    log(f"Summary: {out_root / 'adaptive_summary.json'}")


if __name__ == "__main__":
    main()
