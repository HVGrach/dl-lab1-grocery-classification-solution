#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")
TRAIN_SCRIPT = ROOT / "scripts" / "train_onefold_no_color_innov_mps.py"


@dataclass
class Experiment:
    name: str
    group: str
    model_name: str
    img_size: int
    batch_size: int
    epochs: int
    stage1_epochs: int
    warmup_epochs: int
    lr: float
    lr_drop_factor: float
    weight_decay: float
    label_smoothing: float
    use_channels_last: bool
    use_weighted_sampler: bool
    use_mixup: bool
    mixup_alpha: float
    mixup_prob: float
    use_cutmix: bool
    cutmix_alpha: float
    cutmix_prob: float
    use_sam: bool
    sam_rho: float
    sam_adaptive: bool
    use_swa: bool
    swa_start_epoch: int
    run_seed: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Night autopilot: diverse one-fold training zoo + ensemble ranking.")
    p.add_argument(
        "--base",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1",
    )
    p.add_argument("--clean-variant", type=str, default="strict", choices=["strict", "aggressive", "raw"])
    p.add_argument(
        "--folds-csv",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps/folds_used.csv",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--cv-folds", type=str, default="", help="Comma-separated folds to run, e.g. '0,1,2,3,4'.")
    p.add_argument("--cv-all", action="store_true", help="Run all folds (from folds-csv if present, else range(n-splits)).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--preset", type=str, default="overnight_10h", choices=["quick", "overnight_10h", "full"])
    p.add_argument(
        "--extra-seeds",
        type=str,
        default="",
        help="Comma-separated extra seeds for seeded clones, e.g. '133,777'.",
    )
    p.add_argument(
        "--extra-seed-target",
        type=str,
        default="cnn",
        help="Apply --extra-seeds to experiments matching these tokens (name/group/model), comma-separated; use 'all'.",
    )
    p.add_argument("--max-experiments", type=int, default=0)
    p.add_argument("--ensemble-trials", type=int, default=6000)
    p.add_argument("--ensemble-max-models", type=int, default=4)
    p.add_argument("--python-bin", type=str, default="/opt/homebrew/bin/python3.11")
    p.add_argument(
        "--out-root",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps",
    )
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1-epoch sanity mode")
    p.set_defaults(continue_on_error=True)
    return p.parse_args()


def build_catalog() -> Dict[str, List[Experiment]]:
    quick = [
        Experiment(
            name="quick_convnext_small",
            group="quick",
            model_name="convnext_small.fb_in22k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=8,
            stage1_epochs=5,
            warmup_epochs=2,
            lr=3e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            label_smoothing=0.05,
            use_channels_last=True,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=6,
        ),
        Experiment(
            name="quick_effnetv2_s",
            group="quick",
            model_name="tf_efficientnetv2_s.in21k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=8,
            stage1_epochs=5,
            warmup_epochs=2,
            lr=2.5e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            label_smoothing=0.05,
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=6,
        ),
    ]

    overnight_10h = [
        Experiment(
            name="cnn_convnext_small_sam_swa",
            group="cnn",
            model_name="convnext_small.fb_in22k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=16,
            stage1_epochs=10,
            warmup_epochs=2,
            lr=3e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            label_smoothing=0.05,
            use_channels_last=True,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=12,
        ),
        Experiment(
            name="cnn_effnetv2_s_sam_swa",
            group="cnn",
            model_name="tf_efficientnetv2_s.in21k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=16,
            stage1_epochs=10,
            warmup_epochs=2,
            lr=2.5e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            label_smoothing=0.05,
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=12,
        ),
        Experiment(
            name="vit_deit3_small_color_safe",
            group="transformer",
            model_name="deit3_small_patch16_224.fb_in22k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=14,
            stage1_epochs=9,
            warmup_epochs=2,
            lr=2e-4,
            lr_drop_factor=3.0,
            weight_decay=8e-5,
            label_smoothing=0.08,
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.25,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=False,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=11,
        ),
        Experiment(
            name="vit_base_augreg_lite",
            group="transformer",
            model_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
            img_size=224,
            batch_size=10,
            epochs=12,
            stage1_epochs=8,
            warmup_epochs=2,
            lr=1.8e-4,
            lr_drop_factor=3.0,
            weight_decay=8e-5,
            label_smoothing=0.08,
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.25,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=False,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=10,
        ),
    ]

    full = overnight_10h + [
        Experiment(
            name="cnn_convnext_base_256",
            group="cnn_heavy",
            model_name="convnext_base.fb_in22k_ft_in1k",
            img_size=256,
            batch_size=8,
            epochs=12,
            stage1_epochs=8,
            warmup_epochs=2,
            lr=2.2e-4,
            lr_drop_factor=3.5,
            weight_decay=1e-4,
            label_smoothing=0.05,
            use_channels_last=True,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=False,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=10,
        ),
        Experiment(
            name="cnn_effnetv2_m",
            group="cnn_heavy",
            model_name="tf_efficientnetv2_m.in21k_ft_in1k",
            img_size=224,
            batch_size=10,
            epochs=12,
            stage1_epochs=8,
            warmup_epochs=2,
            lr=2.0e-4,
            lr_drop_factor=3.5,
            weight_decay=1e-4,
            label_smoothing=0.05,
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=False,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=10,
        ),
        Experiment(
            name="cnn_convnext_small_lowmix",
            group="ablation",
            model_name="convnext_small.fb_in22k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=14,
            stage1_epochs=9,
            warmup_epochs=2,
            lr=3e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            label_smoothing=0.03,
            use_channels_last=True,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.10,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.10,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=11,
        ),
    ]

    return {"quick": quick, "overnight_10h": overnight_10h, "full": full}


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        s = x.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def parse_tokens(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def match_seed_target(exp: Experiment, tokens: List[str]) -> bool:
    if not tokens:
        return False
    if "all" in tokens:
        return True
    hay = f"{exp.name} {exp.group} {exp.model_name}".lower()
    return any(t in hay for t in tokens)


def expand_seed_clones(experiments: List[Experiment], base_seed: int, extra_seeds_raw: str, target_raw: str) -> List[Experiment]:
    extra = [s for s in parse_int_csv(extra_seeds_raw) if s != base_seed]
    targets = parse_tokens(target_raw)
    if not extra or not targets:
        return experiments

    out = list(experiments)
    seen = {e.name for e in out}
    for exp in experiments:
        if not match_seed_target(exp, targets):
            continue
        for seed in extra:
            clone_name = f"{exp.name}_seed{seed}"
            if clone_name in seen:
                continue
            out.append(replace(exp, name=clone_name, run_seed=seed))
            seen.add(clone_name)
    return out


def resolve_fold_indices(args: argparse.Namespace) -> List[int]:
    if args.cv_folds.strip():
        folds = sorted(set(parse_int_csv(args.cv_folds)))
        if not folds:
            raise ValueError("cv-folds provided but no valid fold indices parsed.")
        return folds

    if args.cv_all:
        folds_csv = Path(args.folds_csv) if args.folds_csv else None
        if folds_csv and folds_csv.exists():
            fdf = pd.read_csv(folds_csv)
            if "fold" not in fdf.columns:
                raise ValueError(f"{folds_csv} must contain 'fold' column for --cv-all")
            folds = sorted(int(x) for x in fdf["fold"].dropna().unique().tolist())
            if not folds:
                raise ValueError(f"{folds_csv} has no fold values")
            return folds
        return list(range(int(args.n_splits)))

    return [int(args.fold_idx)]


def experiment_to_cmd(
    exp: Experiment,
    args: argparse.Namespace,
    out_dir: Path,
    n_splits: int,
    fold_idx: int,
) -> List[str]:
    run_seed = exp.run_seed if exp.run_seed is not None else args.seed
    cmd = [
        args.python_bin,
        str(TRAIN_SCRIPT),
        "--base",
        args.base,
        "--clean-variant",
        args.clean_variant,
        "--fold-idx",
        str(fold_idx),
        "--seed",
        str(run_seed),
        "--device",
        args.device,
        "--model-name",
        exp.model_name,
        "--img-size",
        str(exp.img_size),
        "--batch-size",
        str(exp.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(exp.epochs),
        "--stage1-epochs",
        str(exp.stage1_epochs),
        "--warmup-epochs",
        str(exp.warmup_epochs),
        "--lr",
        str(exp.lr),
        "--lr-drop-factor",
        str(exp.lr_drop_factor),
        "--weight-decay",
        str(exp.weight_decay),
        "--grad-clip-norm",
        "1.0",
        "--label-smoothing",
        str(exp.label_smoothing),
        "--mixup-alpha",
        str(exp.mixup_alpha),
        "--mixup-prob",
        str(exp.mixup_prob),
        "--cutmix-alpha",
        str(exp.cutmix_alpha),
        "--cutmix-prob",
        str(exp.cutmix_prob),
        "--sam-rho",
        str(exp.sam_rho),
        "--swa-start-epoch",
        str(exp.swa_start_epoch),
        "--out-dir",
        str(out_dir),
    ]

    folds_csv = Path(args.folds_csv) if args.folds_csv else None
    if folds_csv and folds_csv.exists():
        cmd.extend(["--folds-csv", str(folds_csv)])
    else:
        cmd.extend(["--n-splits", str(n_splits), "--fold-seed", str(args.fold_seed)])

    if exp.use_channels_last:
        cmd.append("--use-channels-last")

    if not exp.use_weighted_sampler:
        cmd.append("--no-weighted-sampler")
    if not exp.use_mixup:
        cmd.append("--no-mixup")
    if not exp.use_cutmix:
        cmd.append("--no-cutmix")

    if not exp.use_sam:
        cmd.append("--no-sam")
    elif exp.sam_adaptive:
        cmd.append("--sam-adaptive")

    if not exp.use_swa:
        cmd.append("--no-swa")

    return cmd


def apply_smoke(exp: Experiment) -> Experiment:
    # Fast sanity profile: checks full launch chain, not model quality.
    return Experiment(
        name=exp.name,
        group=exp.group,
        model_name=exp.model_name,
        img_size=exp.img_size,
        batch_size=min(exp.batch_size, 12),
        epochs=1,
        stage1_epochs=0,
        warmup_epochs=1,
        lr=exp.lr,
        lr_drop_factor=exp.lr_drop_factor,
        weight_decay=exp.weight_decay,
        label_smoothing=exp.label_smoothing,
        use_channels_last=exp.use_channels_last,
        use_weighted_sampler=exp.use_weighted_sampler,
        use_mixup=False,
        mixup_alpha=exp.mixup_alpha,
        mixup_prob=0.0,
        use_cutmix=False,
        cutmix_alpha=exp.cutmix_alpha,
        cutmix_prob=0.0,
        use_sam=False,
        sam_rho=exp.sam_rho,
        sam_adaptive=False,
        use_swa=False,
        swa_start_epoch=1,
        run_seed=exp.run_seed,
    )


def load_run_payload(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    val_probs = run_dir / "val_probs.npy"
    val_labels = run_dir / "val_labels.npy"
    val_preds = run_dir / "val_predictions.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    out = {
        "summary": summary,
        "val_probs": np.load(val_probs) if val_probs.exists() else None,
        "val_labels": np.load(val_labels) if val_labels.exists() else None,
        "val_predictions": pd.read_csv(val_preds) if val_preds.exists() else None,
    }
    return out


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
    }


def search_best_weights(prob_list: Sequence[np.ndarray], y_true: np.ndarray, trials: int, seed: int) -> Tuple[np.ndarray, Dict[str, float]]:
    # Kept args for backward-compatible call sites/CLI. Weight fitting is now deterministic.
    del trials, seed
    n_models = len(prob_list)
    if n_models == 0:
        raise ValueError("prob_list must contain at least one model")

    prob_stack = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list], axis=-1)  # [N, C, M]
    n_samples, n_classes, _ = prob_stack.shape
    y_idx = np.asarray(y_true, dtype=np.int64)
    if y_idx.shape[0] != n_samples:
        raise ValueError(f"Label/prob shape mismatch: labels={y_idx.shape[0]} vs probs={n_samples}")

    # Minimize MSE against one-hot targets with a shared weight vector across classes.
    y_onehot = np.eye(n_classes, dtype=np.float64)[y_idx]
    X = prob_stack.reshape(n_samples * n_classes, n_models)
    y = y_onehot.reshape(n_samples * n_classes)

    w = None
    try:
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y)
        w = np.asarray(reg.coef_, dtype=np.float64).reshape(-1)
    except TypeError:
        # Older sklearn versions may not support positive=True.
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        w = np.clip(np.asarray(reg.coef_, dtype=np.float64).reshape(-1), 0.0, None)
    except Exception:
        w = None

    if w is None or (not np.all(np.isfinite(w))) or float(np.sum(w)) <= 0.0:
        w = np.ones(n_models, dtype=np.float64) / n_models
    else:
        w = np.clip(w, 0.0, None)
        s = float(np.sum(w))
        if s <= 0.0:
            w = np.ones(n_models, dtype=np.float64) / n_models
        else:
            # Convex normalization keeps blended probabilities on a comparable scale.
            w = w / s

    blend = np.tensordot(prob_stack, w, axes=([2], [0]))
    best = evaluate_probs(y_true, blend)
    return w.astype(np.float64), best


def build_ensemble_report(success_rows: List[Dict], out_root: Path, trials: int, max_models: int, seed: int) -> Dict:
    enriched = []
    for row in success_rows:
        run_dir = Path(row["run_dir"])
        payload = load_run_payload(run_dir)
        probs = payload["val_probs"]
        labels = payload["val_labels"]
        pred_df = payload["val_predictions"]
        if probs is None or labels is None or pred_df is None:
            continue
        enriched.append(
            {
                **row,
                "probs": probs,
                "labels": labels,
                "image_ids": pred_df["image_id"].to_numpy(),
            }
        )

    if len(enriched) < 2:
        return {"status": "not_enough_models_for_ensemble", "num_models": len(enriched)}

    # Build ensembles separately for each fold (supports k-fold mode).
    by_fold: Dict[int, List[Dict]] = {}
    for row in enriched:
        fid = int(row.get("fold_idx", -1))
        by_fold.setdefault(fid, []).append(row)

    all_results = []
    per_fold = {}
    skipped_global = []

    for fid, fold_rows in sorted(by_fold.items(), key=lambda x: x[0]):
        if len(fold_rows) < 2:
            per_fold[fid] = {"status": "not_enough_models", "num_models": len(fold_rows)}
            continue

        ref_ids = fold_rows[0]["image_ids"]
        ref_y = fold_rows[0]["labels"]
        aligned = [fold_rows[0]]
        skipped = []
        for row in fold_rows[1:]:
            if row["probs"].shape[0] != ref_ids.shape[0]:
                skipped.append({"name": row["name"], "reason": "n_val_mismatch"})
                continue
            if not np.array_equal(row["image_ids"], ref_ids):
                skipped.append({"name": row["name"], "reason": "image_order_mismatch"})
                continue
            if not np.array_equal(row["labels"], ref_y):
                skipped.append({"name": row["name"], "reason": "label_mismatch"})
                continue
            aligned.append(row)

        aligned = sorted(aligned, key=lambda r: r["val_acc"], reverse=True)
        top_pool = aligned[: max(2, min(6, len(aligned)))]
        fold_results = []
        for k in range(2, min(max_models, len(top_pool)) + 1):
            for combo in itertools.combinations(top_pool, k):
                names = [r["name"] for r in combo]
                probs = [r["probs"] for r in combo]
                w, m = search_best_weights(probs, ref_y, trials=trials, seed=seed + fid * 1009 + k * 101 + len(fold_results))
                rec = {
                    "fold_idx": int(fid),
                    "k": k,
                    "models": names,
                    "weights": [float(x) for x in w],
                    "acc": float(m["acc"]),
                    "f1_macro": float(m["f1_macro"]),
                }
                fold_results.append(rec)
                all_results.append(rec)

        if fold_results:
            fold_results = sorted(fold_results, key=lambda r: (r["acc"], r["f1_macro"]), reverse=True)
            per_fold[fid] = {
                "status": "ok",
                "num_models": len(aligned),
                "skipped": skipped,
                "best": fold_results[0],
                "top5": fold_results[:5],
            }
        else:
            per_fold[fid] = {"status": "no_ensemble_results", "num_models": len(aligned), "skipped": skipped}

        skipped_global.extend([{"fold_idx": fid, **x} for x in skipped])

    if not all_results:
        return {
            "status": "no_ensemble_results",
            "per_fold": per_fold,
            "skipped": skipped_global,
        }

    all_results = sorted(all_results, key=lambda r: (r["acc"], r["f1_macro"]), reverse=True)
    best = all_results[0]

    flat_rows = []
    for r in all_results:
        flat_rows.append(
            {
                "fold_idx": int(r["fold_idx"]),
                "k": r["k"],
                "models": " | ".join(r["models"]),
                "weights": json.dumps(r["weights"], ensure_ascii=False),
                "acc": r["acc"],
                "f1_macro": r["f1_macro"],
            }
        )
    pd.DataFrame(flat_rows).to_csv(out_root / "ensemble_candidates.csv", index=False)

    return {
        "status": "ok",
        "num_models": len(enriched),
        "per_fold": per_fold,
        "skipped": skipped_global,
        "best": best,
        "top10": all_results[:10],
        "csv": str(out_root / "ensemble_candidates.csv"),
    }


def main() -> None:
    args = parse_args()
    catalog = build_catalog()
    experiments = catalog[args.preset]

    experiments = expand_seed_clones(
        experiments=experiments,
        base_seed=args.seed,
        extra_seeds_raw=args.extra_seeds,
        target_raw=args.extra_seed_target,
    )
    fold_indices = resolve_fold_indices(args)

    if args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]

    if args.smoke:
        experiments = [apply_smoke(e) for e in experiments]

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp": int(time.time()),
        "preset": args.preset,
        "smoke": bool(args.smoke),
        "dry_run": bool(args.dry_run),
        "base": args.base,
        "clean_variant": args.clean_variant,
        "folds_csv": args.folds_csv if args.folds_csv and Path(args.folds_csv).exists() else "",
        "n_splits": 2 if args.smoke else args.n_splits,
        "fold_seed": args.fold_seed,
        "fold_indices": fold_indices,
        "seed": args.seed,
        "extra_seeds": [int(x) for x in parse_int_csv(args.extra_seeds)] if args.extra_seeds.strip() else [],
        "extra_seed_target": args.extra_seed_target,
        "device": args.device,
        "num_workers": args.num_workers,
        "num_experiments": len(experiments),
        "num_runs_planned": len(experiments) * len(fold_indices),
    }
    (out_root / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== NIGHT MODEL ZOO AUTOPILOT ===", flush=True)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)

    records: List[Dict] = []
    n_splits = 2 if args.smoke else args.n_splits

    run_items = []
    for fold_idx in fold_indices:
        for exp in experiments:
            run_items.append((exp, int(fold_idx)))

    for idx, (exp, fold_idx) in enumerate(run_items, start=1):
        run_dir = runs_root / f"{idx:03d}_{exp.name}_f{fold_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = experiment_to_cmd(exp, args=args, out_dir=run_dir, n_splits=n_splits, fold_idx=fold_idx)
        run_seed = exp.run_seed if exp.run_seed is not None else args.seed

        cmd_txt = " ".join(cmd)
        (run_dir / "cmd.txt").write_text(cmd_txt + "\n", encoding="utf-8")
        print(f"\n[{idx}/{len(run_items)}] {exp.name} (seed={run_seed}, fold={fold_idx})", flush=True)
        print(cmd_txt, flush=True)

        t0 = time.time()
        status = "ok"
        error = ""
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                status = "failed"
                error = f"returncode={e.returncode}"
                if not args.continue_on_error:
                    dt = time.time() - t0
                    records.append(
                        {
                            "name": exp.name,
                            "group": exp.group,
                            "model_name": exp.model_name,
                            "status": status,
                            "error": error,
                            "seed": int(run_seed),
                            "fold_idx": int(fold_idx),
                            "seconds": float(dt),
                            "run_dir": str(run_dir),
                        }
                    )
                    break

        dt = time.time() - t0
        row = {
            "name": exp.name,
            "group": exp.group,
            "model_name": exp.model_name,
            "status": status,
            "error": error,
            "seed": int(run_seed),
            "fold_idx": int(fold_idx),
            "seconds": float(dt),
            "run_dir": str(run_dir),
        }

        summary_path = run_dir / "summary.json"
        if status == "ok" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            fm = summary.get("final_metrics", {})
            row["val_acc"] = float(fm.get("val_acc", np.nan))
            row["val_f1_macro"] = float(fm.get("val_f1_macro", np.nan))
            row["val_loss"] = float(fm.get("val_loss", np.nan))
            row["val_errors"] = int(fm.get("val_errors", -1))
            row["val_size"] = int(fm.get("val_size", -1))
            row["final_model_selected"] = str(summary.get("final_model_selected", ""))
        records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        rank_df = df.copy()
        for c in ["val_acc", "val_f1_macro", "val_loss"]:
            if c not in rank_df.columns:
                rank_df[c] = np.nan
        rank_df = rank_df.sort_values(["status", "val_acc", "val_f1_macro"], ascending=[True, False, False])
        rank_df.to_csv(out_root / "run_ranking.csv", index=False)

        # Cross-validation summary by model variant (name includes optional seed clone suffix).
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

    success_rows = []
    if not df.empty and "status" in df.columns:
        success_rows = df[df["status"] == "ok"].to_dict(orient="records")

    ensemble_summary = {"status": "skipped_dry_run"} if args.dry_run else build_ensemble_report(
        success_rows=success_rows,
        out_root=out_root,
        trials=args.ensemble_trials,
        max_models=args.ensemble_max_models,
        seed=args.seed,
    )

    final_summary = {
        "run_config": run_meta,
        "num_total": len(records),
        "num_success": int(sum(1 for r in records if r.get("status") == "ok")),
        "num_failed": int(sum(1 for r in records if r.get("status") == "failed")),
        "ranking_csv": str(out_root / "run_ranking.csv"),
        "cv_model_summary_csv": str(out_root / "cv_model_summary.csv") if (out_root / "cv_model_summary.csv").exists() else "",
        "ensemble": ensemble_summary,
    }
    (out_root / "autopilot_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== AUTOPILOT DONE ===", flush=True)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
