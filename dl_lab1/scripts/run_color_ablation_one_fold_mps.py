#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Allow importing train_top1_mps.py from dl_lab1 root.
THIS_DIR = Path(__file__).resolve().parent
DL_LAB1_DIR = THIS_DIR.parent


def _load_train_top1_module():
    module_path = DL_LAB1_DIR / "train_top1_mps.py"
    spec = importlib.util.spec_from_file_location("train_top1_mps", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_top1_mps"] = module
    spec.loader.exec_module(module)
    return module


_train_top1 = _load_train_top1_module()
Config = _train_top1.Config
FruitDataset = _train_top1.FruitDataset
build_paths = _train_top1.build_paths
build_valid_tfms = _train_top1.build_valid_tfms
create_model = _train_top1.create_model
evaluate_logits = _train_top1.evaluate_logits
load_train_df = _train_top1.load_train_df
make_folds = _train_top1.make_folds
seed_everything = _train_top1.seed_everything
train_one_fold = _train_top1.train_one_fold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast A/B: full vs no_color on the same single fold.")
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--clean-variant", type=str, default="strict")
    p.add_argument("--folds-csv", type=str, default="", help="Optional fixed fold assignment csv with columns: image_id,label,fold.")
    p.add_argument("--split-folds", type=int, default=5, help="Used only if --folds-csv is not provided.")
    p.add_argument("--fold-seed", type=int, default=42, help="Used only if --folds-csv is not provided.")
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-alias", type=str, default="convnext_small")
    p.add_argument("--profile", type=str, default="both", choices=["both", "full", "no_color"])
    p.add_argument(
        "--out-root",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_color_ablation_onefold_mps",
    )
    return p.parse_args()


def get_model_cfg(alias: str, cfg: Config) -> Dict:
    for m in cfg.model_configs:
        if m["alias"] == alias:
            return dict(m)
    raise ValueError(f"Unknown model alias: {alias}")


def load_fixed_or_build_folds(train_df: pd.DataFrame, folds_csv: Path | None, split_folds: int, fold_seed: int) -> pd.DataFrame:
    if folds_csv and folds_csv.exists():
        fdf = pd.read_csv(folds_csv)
        req = {"image_id", "label", "fold"}
        if not req.issubset(set(fdf.columns)):
            raise RuntimeError(f"{folds_csv} must contain columns: {sorted(req)}")
        merged = train_df[["image_id", "label", "class_name", "plu"]].merge(
            fdf[["image_id", "label", "fold"]], on=["image_id", "label"], how="left"
        )
        if merged["fold"].isna().any():
            miss = int(merged["fold"].isna().sum())
            raise RuntimeError(f"Fold csv mismatch: {miss} rows without fold assignment")
        merged["fold"] = merged["fold"].astype(int)
        merged["strat_key"] = merged["label"].astype(str) + "_" + merged["plu"].astype(str)
        return merged
    return make_folds(train_df, n_splits=split_folds, seed=fold_seed)


def eval_on_fold(
    model_cfg: Dict,
    best_path: Path,
    fold: int,
    train_df: pd.DataFrame,
    paths: Dict[str, Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, float]:
    use_channels_last = bool(model_cfg.get("use_channels_last", True))
    img_size = int(model_cfg["img_size"])

    va_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
    va_ds = FruitDataset(va_df, paths["train_images_dir"], transform=build_valid_tfms(img_size), is_test=False)
    dl_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)

    num_classes = int(train_df["label"].nunique())
    model = create_model(model_cfg, num_classes=num_classes).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y in va_loader:
            x = x.to(device)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()
            logits = model(x)
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())

    logits_np = torch.cat(logits_all).numpy()
    y_np = torch.cat(y_all).numpy()
    m = evaluate_logits(y_np, logits_np)
    return {
        "acc": float(m["acc"]),
        "f1_macro": float(m["f1_macro"]),
        "n_val": int(len(y_np)),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss_from_training": float(ckpt.get("val_loss", np.nan)),
        "best_val_acc_from_training": float(ckpt.get("val_acc", np.nan)),
    }


def run_profile(
    aug_profile: str,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    paths: Dict[str, Path],
    model_cfg: Dict,
    class_weights: Dict[int, float],
    class_weights_tensor: torch.Tensor,
    device: torch.device,
    out_root: Path,
) -> Dict:
    out_dir = out_root / aug_profile
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        seed=args.seed,
        clean_variant=args.clean_variant,
        n_folds=args.split_folds,
        fold_seed=args.fold_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        aug_profile=aug_profile,
        out_dir=str(out_dir),
    )
    cfg.model_configs = (model_cfg,)

    best_path = train_one_fold(
        model_cfg=model_cfg,
        fold=args.fold_idx,
        df=train_df,
        cfg=cfg,
        paths=paths,
        class_weights=class_weights,
        class_weights_tensor=class_weights_tensor,
        num_classes=int(train_df["label"].nunique()),
        device=device,
    )
    eval_metrics = eval_on_fold(
        model_cfg=model_cfg,
        best_path=best_path,
        fold=args.fold_idx,
        train_df=train_df,
        paths=paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    result = {
        "aug_profile": aug_profile,
        "best_checkpoint": str(best_path),
        "metrics_on_same_fold": eval_metrics,
        "config": asdict(cfg),
    }
    (out_dir / "one_fold_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    os_env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    for k, v in os_env.items():
        if k not in os.environ:
            os.environ[k] = v

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    device = torch.device("mps")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    paths = build_paths(Path(args.base), args.clean_variant)
    train_df = load_train_df(paths)

    folds_csv = Path(args.folds_csv) if args.folds_csv else None
    train_df = load_fixed_or_build_folds(train_df, folds_csv, args.split_folds, args.fold_seed)
    if args.fold_idx < 0 or args.fold_idx >= int(train_df["fold"].max()) + 1:
        raise ValueError(f"fold-idx out of range: {args.fold_idx}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_root / "folds_used.csv", index=False)

    # Class weights identical for both profiles.
    label_counts = train_df["label"].value_counts().sort_index()
    num_classes = int(label_counts.shape[0])
    n = len(train_df)
    class_weights = {k: n / (num_classes * v) for k, v in label_counts.to_dict().items()}
    max_w = max(class_weights.values())
    class_weights = {k: v / max_w for k, v in class_weights.items()}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32)

    cfg_ref = Config()
    model_cfg = get_model_cfg(args.model_alias, cfg_ref)

    if args.profile == "both":
        profiles = ["full", "no_color"]
    else:
        profiles = [args.profile]

    all_results = {}
    for p in profiles:
        print(f"\n=== RUN PROFILE: {p} | fold={args.fold_idx} | model={args.model_alias} ===", flush=True)
        all_results[p] = run_profile(
            aug_profile=p,
            args=args,
            train_df=train_df,
            paths=paths,
            model_cfg=model_cfg,
            class_weights=class_weights,
            class_weights_tensor=class_weights_tensor,
            device=device,
            out_root=out_root,
        )

    summary = {
        "model_alias": args.model_alias,
        "fold_idx": args.fold_idx,
        "split_folds": args.split_folds,
        "seed": args.seed,
        "fold_seed": args.fold_seed,
        "folds_csv": str(folds_csv) if folds_csv else "",
        "results": all_results,
    }

    if "full" in all_results and "no_color" in all_results:
        full = all_results["full"]["metrics_on_same_fold"]
        nc = all_results["no_color"]["metrics_on_same_fold"]
        summary["delta_no_color_minus_full"] = {
            "acc": float(nc["acc"] - full["acc"]),
            "f1_macro": float(nc["f1_macro"] - full["f1_macro"]),
            "val_loss_best_ckpt": float(nc["best_val_loss_from_training"] - full["best_val_loss_from_training"]),
        }

    summary_path = out_root / "one_fold_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ONE-FOLD COLOR ABLATION DONE ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
