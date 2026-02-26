#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


ROOT = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")
BASE_TRAINER = ROOT / "scripts" / "train_onefold_no_color_innov_mps.py"


def _load_base_trainer():
    spec = importlib.util.spec_from_file_location("train_onefold_no_color_innov_mps", str(BASE_TRAINER))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {BASE_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_onefold_no_color_innov_mps"] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_trainer()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage4 confidence-aware finetune from an existing checkpoint (hard=rotate-only, easy=strong crop)."
    )
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--clean-variant", type=str, default="raw", choices=["strict", "aggressive", "raw"])
    p.add_argument("--folds-csv", type=str, default="")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--fold-idx", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model-name", type=str, default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--use-channels-last", action="store_true")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3, help="Stage4 finetune epochs.")
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--lr-min-scale", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.02)

    p.add_argument("--no-weighted-sampler", action="store_true")
    p.add_argument("--hard-frac", type=float, default=0.25, help="Lowest-confidence train fraction treated as hard.")
    p.add_argument(
        "--include-misclassified-hard",
        dest="include_misclassified_hard",
        action="store_true",
        help="Always include train samples misclassified by the checkpoint in hard split.",
    )
    p.add_argument(
        "--exclude-misclassified-hard",
        dest="include_misclassified_hard",
        action="store_false",
        help="Do not force misclassified samples into hard split.",
    )
    p.set_defaults(include_misclassified_hard=True)

    p.add_argument("--hard-rotate-deg", type=float, default=10.0)
    p.add_argument("--easy-crop-scale-min", type=float, default=0.30)
    p.add_argument("--easy-crop-scale-max", type=float, default=0.92)
    p.add_argument("--easy-rotate-deg", type=float, default=30.0)
    p.add_argument("--easy-affine-p", type=float, default=0.55)
    p.add_argument("--easy-dropout-p", type=float, default=0.25)
    p.add_argument("--easy-degrade-p", type=float, default=0.20)

    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def build_hard_rotate_only_tfms(img_size: int, rotate_deg: float) -> A.Compose:
    # Preserve content for low-confidence samples: deterministic crop + slight rotations only.
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Rotate(limit=(-rotate_deg, rotate_deg), border_mode=0, p=0.95),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_easy_strong_no_color_tfms(
    img_size: int,
    crop_scale_min: float,
    crop_scale_max: float,
    rotate_deg: float,
    affine_p: float,
    dropout_p: float,
    degrade_p: float,
) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(crop_scale_min, crop_scale_max),
                ratio=(0.65, 1.45),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.10),
            A.Affine(
                scale=(0.80, 1.18),
                translate_percent=(-0.08, 0.08),
                rotate=(-rotate_deg, rotate_deg),
                shear=(-10, 10),
                border_mode=0,
                p=affine_p,
            ),
            A.RandomRotate90(p=0.20),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.ImageCompression(quality_range=(70, 98), p=1.0),
                ],
                p=degrade_p,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.20),
                hole_width_range=(0.05, 0.20),
                fill=0,
                p=dropout_p,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class ConfidencePolicyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        hard_ids: Set[str],
        hard_transform: A.Compose,
        easy_transform: A.Compose,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.hard_ids = set(hard_ids)
        self.hard_transform = hard_transform
        self.easy_transform = easy_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        image = np.array(Image.open(self.img_dir / image_id).convert("RGB"))
        if image_id in self.hard_ids:
            image = self.hard_transform(image=image)["image"]
            is_hard = 1
        else:
            image = self.easy_transform(image=image)["image"]
            is_hard = 0
        return image, int(row["label"]), image_id, is_hard


@torch.no_grad()
def infer_confidence_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_channels_last: bool,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, object]] = []
    for x, y, image_ids in tqdm(loader, desc="infer train conf", leave=False):
        x = x.to(device)
        y = y.to(device)
        if use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        for image_id, yt, yp, cf in zip(image_ids, y.detach().cpu(), pred.detach().cpu(), conf.detach().cpu()):
            rows.append(
                {
                    "image_id": str(image_id),
                    "label_true": int(yt.item()),
                    "label_pred": int(yp.item()),
                    "conf": float(cf.item()),
                    "is_error": int(int(yt.item()) != int(yp.item())),
                }
            )
    return pd.DataFrame(rows)


def cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, base_lr: float, min_scale: float) -> float:
    if total_epochs <= 1:
        lr = float(base_lr)
    else:
        t = epoch / float(total_epochs - 1)
        lr_min = base_lr * min_scale
        lr = lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return float(lr)


def make_class_weights(df_all: pd.DataFrame) -> Dict[int, float]:
    label_counts = df_all["label"].value_counts().sort_index()
    num_classes = int(label_counts.shape[0])
    n = len(df_all)
    class_weights = {k: n / (num_classes * v) for k, v in label_counts.to_dict().items()}
    max_w = max(class_weights.values())
    return {k: v / max_w for k, v in class_weights.items()}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if not (0.0 < args.hard_frac < 1.0):
        raise ValueError("--hard-frac must be in (0,1)")
    if not (0.0 < args.easy_crop_scale_min <= args.easy_crop_scale_max <= 1.0):
        raise ValueError("easy crop scale bounds must satisfy 0 < min <= max <= 1")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = base.resolve_device(args.device)
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
    base.seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cmd.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    cfg_dump = {
        "base": args.base,
        "clean_variant": args.clean_variant,
        "folds_csv": args.folds_csv,
        "n_splits": args.n_splits,
        "fold_seed": args.fold_seed,
        "fold_idx": args.fold_idx,
        "seed": args.seed,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "img_size": args.img_size,
        "use_channels_last": args.use_channels_last,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_min_scale": args.lr_min_scale,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "label_smoothing": args.label_smoothing,
        "use_weighted_sampler": not args.no_weighted_sampler,
        "hard_frac": args.hard_frac,
        "include_misclassified_hard": args.include_misclassified_hard,
        "hard_rotate_deg": args.hard_rotate_deg,
        "easy_crop_scale_min": args.easy_crop_scale_min,
        "easy_crop_scale_max": args.easy_crop_scale_max,
        "easy_rotate_deg": args.easy_rotate_deg,
        "easy_affine_p": args.easy_affine_p,
        "easy_dropout_p": args.easy_dropout_p,
        "easy_degrade_p": args.easy_degrade_p,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Stage4 config:", json.dumps(cfg_dump, ensure_ascii=False, indent=2), flush=True)
    print("Device:", device, flush=True)

    paths = base.build_paths(Path(args.base), args.clean_variant)
    for k, v in paths.items():
        print(f"{k}: {v} exists={v.exists()}", flush=True)

    df = base.load_train_df(paths)
    df = base.make_or_load_folds(df, n_splits=args.n_splits, seed=args.fold_seed, folds_csv=args.folds_csv)
    if args.fold_idx >= df["fold"].nunique():
        raise ValueError(f"fold_idx={args.fold_idx} out of range for n_folds={df['fold'].nunique()}")
    df.to_csv(out_dir / "folds_used.csv", index=False)

    tr_df = df[df["fold"] != args.fold_idx].reset_index(drop=True)
    va_df = df[df["fold"] == args.fold_idx].reset_index(drop=True)
    print(f"Train size={len(tr_df)}, Val size={len(va_df)}, Fold={args.fold_idx}", flush=True)

    label_counts = df["label"].value_counts().sort_index()
    num_classes = int(label_counts.shape[0])
    class_weights = make_class_weights(df)
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    print("Class counts:", label_counts.to_dict(), flush=True)
    print("Class weights:", class_weights, flush=True)

    dl_kwargs = {"num_workers": args.num_workers, "persistent_workers": args.num_workers > 0}
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    val_ds = base.FruitDataset(va_df, paths["train_images_dir"], transform=base.build_valid_tfms(args.img_size))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    train_conf_ds = base.FruitDataset(tr_df, paths["train_images_dir"], transform=base.build_valid_tfms(args.img_size))
    train_conf_loader = DataLoader(train_conf_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    hard_tfm = build_hard_rotate_only_tfms(args.img_size, args.hard_rotate_deg)
    easy_tfm = build_easy_strong_no_color_tfms(
        img_size=args.img_size,
        crop_scale_min=args.easy_crop_scale_min,
        crop_scale_max=args.easy_crop_scale_max,
        rotate_deg=args.easy_rotate_deg,
        affine_p=args.easy_affine_p,
        dropout_p=args.easy_dropout_p,
        degrade_p=args.easy_degrade_p,
    )
    train_stage4_ds = ConfidencePolicyDataset(tr_df, paths["train_images_dir"], set(), hard_tfm, easy_tfm)

    if not args.no_weighted_sampler:
        sample_weights = tr_df["label"].map(class_weights).values.astype(np.float32)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_stage4_loader = DataLoader(train_stage4_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True, **dl_kwargs)
    else:
        train_stage4_loader = DataLoader(train_stage4_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kwargs)

    model = base.create_model(args.model_name, num_classes=num_classes).to(device)
    if args.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Baseline eval before stage4 (same protocol as stage4 eval).
    base_eval = base.evaluate(model, val_loader, criterion, device, args.use_channels_last)
    print(
        "[base ckpt eval] "
        f"val_loss={base_eval['val_loss']:.4f} val_acc={base_eval['acc']:.4f} val_f1={base_eval['f1_macro']:.4f}",
        flush=True,
    )

    train_conf_df = infer_confidence_on_loader(model, train_conf_loader, device, args.use_channels_last)
    q_thr = float(train_conf_df["conf"].quantile(args.hard_frac))
    hard_mask = train_conf_df["conf"] <= q_thr
    if args.include_misclassified_hard:
        hard_mask = hard_mask | (train_conf_df["is_error"] == 1)
    hard_ids = set(train_conf_df.loc[hard_mask, "image_id"].astype(str).tolist())
    train_stage4_ds.hard_ids = hard_ids

    train_conf_df["stage4_bucket"] = np.where(train_conf_df["image_id"].astype(str).isin(hard_ids), "hard", "easy")
    train_conf_df.to_csv(out_dir / "train_confidence_before_stage4.csv", index=False)

    split_summary = {
        "hard_frac_requested": float(args.hard_frac),
        "hard_quantile_threshold": q_thr,
        "include_misclassified_hard": bool(args.include_misclassified_hard),
        "n_train": int(len(train_conf_df)),
        "n_hard": int((train_conf_df["stage4_bucket"] == "hard").sum()),
        "n_easy": int((train_conf_df["stage4_bucket"] == "easy").sum()),
        "n_train_errors_checkpoint": int(train_conf_df["is_error"].sum()),
        "hard_errors": int(train_conf_df[(train_conf_df["stage4_bucket"] == "hard") & (train_conf_df["is_error"] == 1)].shape[0]),
        "easy_errors": int(train_conf_df[(train_conf_df["stage4_bucket"] == "easy") & (train_conf_df["is_error"] == 1)].shape[0]),
        "hard_conf_mean": float(train_conf_df.loc[train_conf_df["stage4_bucket"] == "hard", "conf"].mean()),
        "easy_conf_mean": float(train_conf_df.loc[train_conf_df["stage4_bucket"] == "easy", "conf"].mean()),
    }
    (out_dir / "stage4_split_summary.json").write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage4 split:", json.dumps(split_summary, ensure_ascii=False, indent=2), flush=True)

    best_val_loss = float(base_eval["val_loss"])
    best_epoch = 0
    best_model_path = out_dir / "best_by_val_loss.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "val_loss": float(base_eval["val_loss"]),
            "val_acc": float(base_eval["acc"]),
            "val_f1_macro": float(base_eval["f1_macro"]),
            "source_checkpoint": str(args.checkpoint),
            "stage4": False,
        },
        best_model_path,
    )

    epoch_rows: List[Dict[str, object]] = []
    for epoch in range(args.epochs):
        lr_now = cosine_lr(optimizer, epoch, args.epochs, args.lr, args.lr_min_scale)
        model.train()
        tr_losses: List[float] = []
        n_hard_batches = 0
        n_easy_batches = 0

        pbar = tqdm(train_stage4_loader, desc=f"stage4 e{epoch+1}/{args.epochs}", leave=False)
        for x, y, _, is_hard in pbar:
            x = x.to(device)
            y = y.to(device)
            if args.use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()

            if int(is_hard.sum().item()) > 0:
                n_hard_batches += 1
            if int((is_hard == 0).sum().item()) > 0:
                n_easy_batches += 1

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            tr_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(tr_losses):.4f}", lr=f"{lr_now:.2e}")

        val_info = base.evaluate(model, val_loader, criterion, device, args.use_channels_last)
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        val_loss = float(val_info["val_loss"])
        row = {
            "epoch": epoch + 1,
            "phase": "stage4",
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": float(val_info["acc"]),
            "val_f1_macro": float(val_info["f1_macro"]),
            "lr": lr_now,
            "hard_batches_seen": int(n_hard_batches),
            "easy_batches_seen": int(n_easy_batches),
        }
        epoch_rows.append(row)
        print(
            f"[stage4 {epoch+1:02d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_info['acc']:.4f} val_f1={val_info['f1_macro']:.4f} "
            f"lr={lr_now:.2e} hard_batches={n_hard_batches} easy_batches={n_easy_batches}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_loss": float(val_info["val_loss"]),
                    "val_acc": float(val_info["acc"]),
                    "val_f1_macro": float(val_info["f1_macro"]),
                    "source_checkpoint": str(args.checkpoint),
                    "stage4": True,
                    "stage4_config": cfg_dump,
                    "stage4_split": split_summary,
                },
                best_model_path,
            )

    pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_log_stage4.csv", index=False)

    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_eval = base.evaluate(model, val_loader, criterion, device, args.use_channels_last)

    label_to_name = df.groupby("label")["class_name"].agg(lambda s: s.mode().iat[0]).to_dict()
    val_pred_df = pd.DataFrame(
        {
            "image_id": final_eval["image_ids"],
            "label_true": final_eval["y_true"],
            "label_pred": final_eval["y_pred"],
            "conf": final_eval["confidence"],
        }
    )
    val_pred_df["class_true"] = val_pred_df["label_true"].map(label_to_name)
    val_pred_df["class_pred"] = val_pred_df["label_pred"].map(label_to_name)
    val_pred_df["is_error"] = (val_pred_df["label_true"] != val_pred_df["label_pred"]).astype(int)
    val_pred_df.to_csv(out_dir / "val_predictions.csv", index=False)

    class_df = base.class_report_df(final_eval["y_true"], final_eval["y_pred"], label_to_name)
    class_df.to_csv(out_dir / "val_class_report.csv", index=False)
    conf_df = base.top_confusions_df(final_eval["y_true"], final_eval["y_pred"], label_to_name, topk=40)
    conf_df.to_csv(out_dir / "val_top_confusions.csv", index=False)
    val_logits = final_eval["logits"].astype(np.float32)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy().astype(np.float32)
    np.save(out_dir / "val_logits.npy", val_logits)
    np.save(out_dir / "val_probs.npy", val_probs)
    np.save(out_dir / "val_labels.npy", final_eval["y_true"].astype(np.int64))

    final_metrics = {
        "val_loss": float(final_eval["val_loss"]),
        "val_acc": float(final_eval["acc"]),
        "val_f1_macro": float(final_eval["f1_macro"]),
        "val_errors": int((final_eval["y_true"] != final_eval["y_pred"]).sum()),
        "val_size": int(len(final_eval["y_true"])),
    }
    base_metrics = {
        "val_loss": float(base_eval["val_loss"]),
        "val_acc": float(base_eval["acc"]),
        "val_f1_macro": float(base_eval["f1_macro"]),
        "val_errors": int((base_eval["y_true"] != base_eval["y_pred"]).sum()),
        "val_size": int(len(base_eval["y_true"])),
    }
    best_checkpoint_metrics = {
        "val_loss": float(final_metrics["val_loss"]) if bool(best_ckpt.get("stage4", False)) else float(base_metrics["val_loss"]),
        "val_acc": float(final_metrics["val_acc"]) if bool(best_ckpt.get("stage4", False)) else float(base_metrics["val_acc"]),
        "val_f1_macro": float(final_metrics["val_f1_macro"]) if bool(best_ckpt.get("stage4", False)) else float(base_metrics["val_f1_macro"]),
        "best_ckpt_epoch": int(best_ckpt.get("epoch", 0)),
    }

    summary = {
        "device": str(device),
        "final_model_selected": "best_checkpoint",
        "checkpoint_init": str(args.checkpoint),
        "best_stage4_epoch": int(best_epoch),
        "best_model_is_stage4": bool(best_ckpt.get("stage4", False)),
        "best_checkpoint_metrics": best_checkpoint_metrics,
        "swa_metrics": None,
        "base_checkpoint_metrics_recomputed": base_metrics,
        "final_metrics": final_metrics,
        "delta_final_minus_base": {
            "val_loss": float(final_metrics["val_loss"] - base_metrics["val_loss"]),
            "val_acc": float(final_metrics["val_acc"] - base_metrics["val_acc"]),
            "val_f1_macro": float(final_metrics["val_f1_macro"] - base_metrics["val_f1_macro"]),
            "val_errors": int(final_metrics["val_errors"] - base_metrics["val_errors"]),
        },
        "stage4_split": split_summary,
        "artifacts": {
            "config": str(out_dir / "config.json"),
            "epoch_log_stage4": str(out_dir / "epoch_log_stage4.csv"),
            "stage4_split_summary": str(out_dir / "stage4_split_summary.json"),
            "train_confidence_before_stage4": str(out_dir / "train_confidence_before_stage4.csv"),
            "best_checkpoint": str(best_model_path),
            "val_predictions": str(out_dir / "val_predictions.csv"),
            "val_logits_npy": str(out_dir / "val_logits.npy"),
            "val_probs_npy": str(out_dir / "val_probs.npy"),
            "val_labels_npy": str(out_dir / "val_labels.npy"),
            "val_class_report": str(out_dir / "val_class_report.csv"),
            "val_top_confusions": str(out_dir / "val_top_confusions.csv"),
            "folds_used": str(out_dir / "folds_used.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== STAGE4 FINETUNE DONE ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
