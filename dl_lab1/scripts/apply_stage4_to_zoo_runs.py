#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")
STAGE4_SCRIPT = ROOT / "scripts" / "finetune_stage4_confidence_mps.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply stage4 confidence-aware finetune to model zoo runs and refresh ranking/cv summaries."
    )
    p.add_argument("--zoo-root", type=str, required=True, help="Root containing run_ranking.csv and runs/")
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--clean-variant", type=str, default="raw", choices=["strict", "aggressive", "raw"])
    p.add_argument("--folds-csv", type=str, default="")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--python-bin", type=str, default="/opt/homebrew/bin/python3.11")
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    p.set_defaults(continue_on_error=True)

    p.add_argument("--folds", type=str, default="", help="Comma-separated folds filter (e.g. 0,1,2). Empty=all in ranking.")
    p.add_argument(
        "--include-names",
        type=str,
        default="",
        help="Comma-separated substrings; keep rows whose name/group/model_name contains any token.",
    )
    p.add_argument(
        "--exclude-names",
        type=str,
        default="",
        help="Comma-separated substrings; drop rows whose name/group/model_name contains any token.",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional limit after filtering, for quick tests.")

    p.add_argument(
        "--mode",
        type=str,
        default="inplace",
        choices=["inplace", "copy"],
        help="inplace overwrites run artifacts in each run_dir; copy writes mirrored run dirs under --out-root.",
    )
    p.add_argument("--out-root", type=str, default="", help="Required for --mode copy.")

    # Stage4 hyperparameters (applied uniformly unless overridden per run config for core model shape args).
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--lr-min-scale", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--no-weighted-sampler", action="store_true")
    p.add_argument("--hard-frac", type=float, default=0.25)
    p.add_argument("--exclude-misclassified-hard", action="store_true")
    p.add_argument("--hard-rotate-deg", type=float, default=10.0)
    p.add_argument("--easy-crop-scale-min", type=float, default=0.30)
    p.add_argument("--easy-crop-scale-max", type=float, default=0.92)
    p.add_argument("--easy-rotate-deg", type=float, default=30.0)
    p.add_argument("--easy-affine-p", type=float, default=0.55)
    p.add_argument("--easy-dropout-p", type=float, default=0.25)
    p.add_argument("--easy-degrade-p", type=float, default=0.20)
    return p.parse_args()


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_tokens(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def row_matches_tokens(row: pd.Series, tokens: List[str]) -> bool:
    if not tokens:
        return True
    hay = f"{row.get('name','')} {row.get('group','')} {row.get('model_name','')}".lower()
    return any(t in hay for t in tokens)


def select_init_checkpoint(run_dir: Path) -> Path:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final_selected = str(summary.get("final_model_selected", "best_checkpoint"))
        if final_selected == "swa_model" and (run_dir / "swa_model.pt").exists():
            return run_dir / "swa_model.pt"
    return run_dir / "best_by_val_loss.pt"


def build_stage4_cmd(
    args: argparse.Namespace,
    row: pd.Series,
    run_dir_src: Path,
    run_dir_dst: Path,
) -> List[str]:
    cfg = json.loads((run_dir_src / "config.json").read_text(encoding="utf-8"))
    fold_idx = int(row["fold_idx"])
    seed = int(row["seed"])

    cmd = [
        args.python_bin,
        str(STAGE4_SCRIPT),
        "--base",
        args.base,
        "--clean-variant",
        args.clean_variant,
        "--fold-idx",
        str(fold_idx),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--checkpoint",
        str(select_init_checkpoint(run_dir_src)),
        "--model-name",
        str(cfg["model_name"]),
        "--img-size",
        str(int(cfg["img_size"])),
        "--batch-size",
        str(int(cfg.get("batch_size", 16))),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--lr-min-scale",
        str(args.lr_min_scale),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--label-smoothing",
        str(args.label_smoothing),
        "--hard-frac",
        str(args.hard_frac),
        "--hard-rotate-deg",
        str(args.hard_rotate_deg),
        "--easy-crop-scale-min",
        str(args.easy_crop_scale_min),
        "--easy-crop-scale-max",
        str(args.easy_crop_scale_max),
        "--easy-rotate-deg",
        str(args.easy_rotate_deg),
        "--easy-affine-p",
        str(args.easy_affine_p),
        "--easy-dropout-p",
        str(args.easy_dropout_p),
        "--easy-degrade-p",
        str(args.easy_degrade_p),
        "--out-dir",
        str(run_dir_dst),
    ]

    if args.folds_csv and Path(args.folds_csv).exists():
        cmd.extend(["--folds-csv", args.folds_csv])
    else:
        cmd.extend(["--n-splits", str(args.n_splits), "--fold-seed", str(args.fold_seed)])

    if bool(cfg.get("use_channels_last", False)):
        cmd.append("--use-channels-last")
    if args.no_weighted_sampler:
        cmd.append("--no-weighted-sampler")
    if args.exclude_misclassified_hard:
        cmd.append("--exclude-misclassified-hard")

    return cmd


def load_updated_metrics(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fm = summary.get("final_metrics", {})
    return {
        "final_model_selected": str(summary.get("final_model_selected", "best_checkpoint")),
        "val_acc": float(fm.get("val_acc", float("nan"))),
        "val_f1_macro": float(fm.get("val_f1_macro", float("nan"))),
        "val_loss": float(fm.get("val_loss", float("nan"))),
        "val_errors": int(fm.get("val_errors", -1)),
        "val_size": int(fm.get("val_size", -1)),
    }


def refresh_ranking_and_cv(zoo_root: Path, ranking_df: pd.DataFrame, out_ranking_name: str = "run_ranking.csv") -> None:
    out_ranking = zoo_root / out_ranking_name
    ranking_df.to_csv(out_ranking, index=False)

    ok = ranking_df[ranking_df["status"] == "ok"].copy()
    if ok.empty:
        return

    gcols = ["name", "group", "model_name", "seed"]
    rows: List[Dict[str, object]] = []
    for keys, g in ok.groupby(gcols, dropna=False):
        name, group, model_name, seed = keys
        rows.append(
            {
                "name": str(name),
                "group": str(group),
                "model_name": str(model_name),
                "seed": int(seed),
                "folds_run": int(g["fold_idx"].nunique()),
                "val_acc_mean": float(g["val_acc"].mean()),
                "val_acc_std": float(g["val_acc"].std(ddof=0)) if len(g) > 1 else 0.0,
                "val_f1_macro_mean": float(g["val_f1_macro"].mean()),
                "val_f1_macro_std": float(g["val_f1_macro"].std(ddof=0)) if len(g) > 1 else 0.0,
                "val_loss_mean": float(g["val_loss"].mean()),
                "val_loss_std": float(g["val_loss"].std(ddof=0)) if len(g) > 1 else 0.0,
                "total_seconds": float(g["seconds"].fillna(0).sum()),
            }
        )
    cv_df = pd.DataFrame(rows).sort_values(
        ["val_acc_mean", "val_f1_macro_mean", "val_loss_mean"],
        ascending=[False, False, True],
    )
    cv_df.to_csv(zoo_root / "cv_model_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.mode == "copy" and not args.out_root:
        raise ValueError("--out-root is required for --mode copy")

    zoo_root = Path(args.zoo_root)
    ranking_path = zoo_root / "run_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing {ranking_path}")
    ranking = pd.read_csv(ranking_path)
    if "status" not in ranking.columns or "run_dir" not in ranking.columns:
        raise RuntimeError("run_ranking.csv missing required columns: status, run_dir")

    ranking["fold_idx"] = ranking["fold_idx"].astype(int)
    ranking["seed"] = ranking["seed"].astype(int)

    base_ok = ranking[ranking["status"] == "ok"].copy()
    if base_ok.empty:
        raise RuntimeError("No successful runs in run_ranking.csv")

    include_tokens = parse_tokens(args.include_names)
    exclude_tokens = parse_tokens(args.exclude_names)
    fold_filter = sorted(set(parse_int_csv(args.folds))) if args.folds.strip() else []

    selected_idx: List[int] = []
    for idx, row in base_ok.iterrows():
        if fold_filter and int(row["fold_idx"]) not in fold_filter:
            continue
        if include_tokens and not row_matches_tokens(row, include_tokens):
            continue
        if exclude_tokens and row_matches_tokens(row, exclude_tokens):
            continue
        selected_idx.append(int(idx))

    selected = base_ok.loc[selected_idx].copy() if selected_idx else base_ok.iloc[0:0].copy()
    selected = selected.sort_values(["fold_idx", "name", "run_dir"]).reset_index(drop=True)
    if args.limit > 0:
        selected = selected.head(args.limit).copy()

    if selected.empty:
        raise RuntimeError("No runs matched filters.")

    out_root = Path(args.out_root) if args.out_root else zoo_root
    if args.mode == "copy":
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "runs").mkdir(parents=True, exist_ok=True)

    print(f"zoo_root={zoo_root}", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"selected_runs={len(selected)}", flush=True)
    print(selected[["fold_idx", "name", "run_dir"]].to_string(index=False), flush=True)

    if args.mode == "copy":
        # Seed outputs with source ranking/cv summaries for compatibility.
        for fname in ["run_ranking.csv", "cv_model_summary.csv", "autopilot_summary.json", "run_config.json"]:
            src = zoo_root / fname
            if src.exists():
                dst = out_root / fname
                dst.write_bytes(src.read_bytes())

    # Work on a ranking copy that we will refresh with updated metrics for processed rows.
    ranking_updated = ranking.copy()
    if "stage4_applied" not in ranking_updated.columns:
        ranking_updated["stage4_applied"] = 0
    if "stage4_seconds" not in ranking_updated.columns:
        ranking_updated["stage4_seconds"] = 0.0

    failures: List[Dict[str, object]] = []
    processed_rows: List[Dict[str, object]] = []

    for i, row in enumerate(selected.to_dict("records"), start=1):
        run_dir_src = Path(str(row["run_dir"]))
        run_dir_dst = (
            run_dir_src
            if args.mode == "inplace"
            else (out_root / "runs" / run_dir_src.name)
        )
        if args.mode == "copy":
            run_dir_dst.mkdir(parents=True, exist_ok=True)

        cmd = build_stage4_cmd(args, pd.Series(row), run_dir_src=run_dir_src, run_dir_dst=run_dir_dst)
        print(f"\n[{i}/{len(selected)}] stage4 {row['name']} fold={row['fold_idx']} -> {run_dir_dst}", flush=True)
        print(" ".join(cmd), flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, check=False)
            seconds = float(time.time() - t0)
            if proc.returncode != 0:
                raise RuntimeError(f"stage4 command failed with code {proc.returncode}")
            m = load_updated_metrics(run_dir_dst)

            # Update ranking row by exact source row match.
            mask = (
                (ranking_updated["run_dir"].astype(str) == str(run_dir_src))
                & (ranking_updated["name"].astype(str) == str(row["name"]))
                & (ranking_updated["fold_idx"].astype(int) == int(row["fold_idx"]))
                & (ranking_updated["seed"].astype(int) == int(row["seed"]))
            )
            if mask.sum() != 1:
                # Fallback to first match on run_dir
                mask = ranking_updated["run_dir"].astype(str) == str(run_dir_src)
            ranking_updated.loc[mask, "status"] = "ok"
            ranking_updated.loc[mask, "error"] = ""
            ranking_updated.loc[mask, "seconds"] = seconds if args.mode == "copy" else ranking_updated.loc[mask, "seconds"]
            ranking_updated.loc[mask, "val_acc"] = float(m["val_acc"])
            ranking_updated.loc[mask, "val_f1_macro"] = float(m["val_f1_macro"])
            ranking_updated.loc[mask, "val_loss"] = float(m["val_loss"])
            ranking_updated.loc[mask, "val_errors"] = int(m["val_errors"])
            ranking_updated.loc[mask, "val_size"] = int(m["val_size"])
            ranking_updated.loc[mask, "final_model_selected"] = str(m["final_model_selected"])
            ranking_updated.loc[mask, "stage4_applied"] = 1
            ranking_updated.loc[mask, "stage4_seconds"] = seconds
            if args.mode == "copy":
                ranking_updated.loc[mask, "run_dir"] = str(run_dir_dst)

            processed_rows.append(
                {
                    "name": str(row["name"]),
                    "fold_idx": int(row["fold_idx"]),
                    "seed": int(row["seed"]),
                    "run_dir_src": str(run_dir_src),
                    "run_dir_dst": str(run_dir_dst),
                    "seconds": seconds,
                    "val_acc": float(m["val_acc"]),
                    "val_f1_macro": float(m["val_f1_macro"]),
                    "val_loss": float(m["val_loss"]),
                }
            )
        except Exception as exc:
            seconds = float(time.time() - t0)
            failures.append(
                {
                    "name": str(row["name"]),
                    "fold_idx": int(row["fold_idx"]),
                    "seed": int(row["seed"]),
                    "run_dir_src": str(run_dir_src),
                    "run_dir_dst": str(run_dir_dst),
                    "seconds": seconds,
                    "error": str(exc),
                }
            )
            print(f"[error] {exc}", flush=True)
            if not args.continue_on_error:
                break

    refresh_ranking_and_cv(out_root, ranking_updated)

    summary = {
        "timestamp": int(time.time()),
        "mode": args.mode,
        "zoo_root": str(zoo_root),
        "out_root": str(out_root),
        "base": args.base,
        "clean_variant": args.clean_variant,
        "folds_csv": args.folds_csv if args.folds_csv else "",
        "filters": {
            "folds": fold_filter,
            "include_tokens": include_tokens,
            "exclude_tokens": exclude_tokens,
            "limit": int(args.limit),
        },
        "stage4_params": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "lr_min_scale": float(args.lr_min_scale),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "label_smoothing": float(args.label_smoothing),
            "weighted_sampler": bool(not args.no_weighted_sampler),
            "hard_frac": float(args.hard_frac),
            "include_misclassified_hard": bool(not args.exclude_misclassified_hard),
            "hard_rotate_deg": float(args.hard_rotate_deg),
            "easy_crop_scale_min": float(args.easy_crop_scale_min),
            "easy_crop_scale_max": float(args.easy_crop_scale_max),
            "easy_rotate_deg": float(args.easy_rotate_deg),
            "easy_affine_p": float(args.easy_affine_p),
            "easy_dropout_p": float(args.easy_dropout_p),
            "easy_degrade_p": float(args.easy_degrade_p),
        },
        "selected_runs": int(len(selected)),
        "processed_ok": int(len(processed_rows)),
        "failed": int(len(failures)),
        "processed_rows": processed_rows,
        "failures": failures,
        "artifacts": {
            "run_ranking_csv": str(out_root / "run_ranking.csv"),
            "cv_model_summary_csv": str(out_root / "cv_model_summary.csv"),
        },
    }
    (out_root / "stage4_apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== STAGE4 APPLY DONE ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
