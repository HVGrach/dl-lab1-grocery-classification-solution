#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    base: str = "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped"
    clean_variant: str = "strict"
    folds_csv: str = ""
    n_splits: int = 5
    fold_seed: int = 42
    fold_idx: int = 0
    seed: int = 42

    model_name: str = "convnext_small.fb_in22k_ft_in1k"
    img_size: int = 224
    use_channels_last: bool = True

    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 20
    stage1_epochs: int = 12
    warmup_epochs: int = 2
    lr: float = 3e-4
    lr_drop_factor: float = 4.0
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    label_smoothing: float = 0.05
    use_weighted_sampler: bool = True

    use_mixup: bool = True
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.20

    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.20

    # Optional experimental last-stage class-aware hardening:
    # stronger no-color aug on "easy" classes, lighter rotate-style aug on others.
    class_aware_harden_last_epochs: int = 0
    class_aware_easy_topk: int = 5
    class_aware_easy_labels_csv: str = ""

    use_sam: bool = True
    sam_rho: float = 0.05
    sam_adaptive: bool = False

    use_swa: bool = True
    swa_start_epoch: int = 15

    out_dir: str = "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_no_color_innov_mps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-fold no-color trainer with advanced regularization.")
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--clean-variant", type=str, default="strict", choices=["strict", "aggressive", "raw"])
    p.add_argument("--folds-csv", type=str, default="")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])

    p.add_argument("--model-name", type=str, default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--use-channels-last", action="store_true")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--stage1-epochs", type=int, default=12)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-drop-factor", type=float, default=4.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)

    p.add_argument("--no-weighted-sampler", action="store_true")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.2)
    p.add_argument("--no-cutmix", action="store_true")
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--cutmix-prob", type=float, default=0.2)
    p.add_argument(
        "--class-aware-harden-last-epochs",
        type=int,
        default=0,
        help="If >0, enable class-aware end-stage hardening for the last N epochs.",
    )
    p.add_argument(
        "--class-aware-easy-topk",
        type=int,
        default=5,
        help="When class-aware hardening is enabled and easy labels are not provided, treat top-K frequent train classes as easy.",
    )
    p.add_argument(
        "--class-aware-easy-labels",
        type=str,
        default="",
        help="Optional comma-separated easy class labels (overrides top-k selection). Example: '0,4,13'.",
    )

    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--sam-rho", type=float, default=0.05)
    p.add_argument("--sam-adaptive", action="store_true")

    p.add_argument("--no-swa", action="store_true")
    p.add_argument("--swa-start-epoch", type=int, default=15)

    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_no_color_innov_mps",
    )
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but CUDA is not available on this machine.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested, but MPS is not available on this machine.")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("[warn] No CUDA/MPS available, falling back to CPU.", flush=True)
    return torch.device("cpu")


def build_paths(base: Path, clean_variant: str) -> Dict[str, Path]:
    if clean_variant == "strict":
        train_csv = base / "cleaning" / "train_clean_strict.csv"
    elif clean_variant == "aggressive":
        train_csv = base / "cleaning" / "train_clean_aggressive.csv"
    else:
        train_csv = base / "train.csv"
    return {
        "base": base,
        "train_csv": train_csv,
        "train_images_dir": base / "train" / "train",
    }


def load_train_df(paths: Dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(paths["train_csv"])
    df["label"] = df["label"].astype(int)
    split = df["image_id"].str.split("/", expand=True)
    df["class_name"] = split[0]
    df["plu"] = split[1]
    return df


def make_or_load_folds(df: pd.DataFrame, n_splits: int, seed: int, folds_csv: str) -> pd.DataFrame:
    if folds_csv:
        fdf = pd.read_csv(folds_csv)
        required = {"image_id", "label", "fold"}
        if not required.issubset(set(fdf.columns)):
            raise RuntimeError(f"{folds_csv} must contain columns: {sorted(required)}")
        merged = df[["image_id", "label", "class_name", "plu"]].merge(
            fdf[["image_id", "label", "fold"]],
            on=["image_id", "label"],
            how="left",
        )
        if merged["fold"].isna().any():
            n_missing = int(merged["fold"].isna().sum())
            raise RuntimeError(f"Fold mapping mismatch: {n_missing} rows have no fold")
        merged["fold"] = merged["fold"].astype(int)
        merged["strat_key"] = merged["label"].astype(str) + "_" + merged["plu"].astype(str)
        return merged

    out = df.copy()
    out["fold"] = -1
    out["strat_key"] = out["label"].astype(str) + "_" + out["plu"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, va_idx) in enumerate(skf.split(out, out["strat_key"])):
        out.loc[out.index[va_idx], "fold"] = fold
    return out


def build_train_tfms_no_color(img_size: int) -> A.Compose:
    # No hue/saturation/brightness transforms: keep color semantics intact.
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.70, 1.0), ratio=(0.80, 1.25), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.08),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.06, 0.06),
                rotate=(-20, 20),
                shear=(-8, 8),
                border_mode=0,
                p=0.45,
            ),
            A.RandomRotate90(p=0.15),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.ImageCompression(quality_range=(75, 100), p=1.0),
                ],
                p=0.25,
            ),
            A.Sharpen(alpha=(0.05, 0.2), lightness=(0.9, 1.1), p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(0.04, 0.12),
                hole_width_range=(0.04, 0.12),
                fill=0,
                p=0.25,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_train_tfms_no_color_easy_strong(img_size: int) -> A.Compose:
    # Stronger no-color recipe used only in the optional end-stage class-aware hardening.
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.62, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.10),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.09, 0.09),
                rotate=(-28, 28),
                shear=(-12, 12),
                border_mode=0,
                p=0.60,
            ),
            A.RandomRotate90(p=0.20),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
                    A.ImageCompression(quality_range=(60, 95), p=1.0),
                ],
                p=0.40,
            ),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(0.05, 0.18),
                hole_width_range=(0.05, 0.18),
                fill=0,
                p=0.35,
            ),
            A.Sharpen(alpha=(0.03, 0.15), lightness=(0.9, 1.1), p=0.12),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_train_tfms_no_color_hard_light(img_size: int) -> A.Compose:
    # Lighter transforms for harder classes: mostly geometry/rotation, minimal corruption.
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.80, 1.0), ratio=(0.85, 1.20), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-12, 12),
                shear=(-4, 4),
                border_mode=0,
                p=0.50,
            ),
            A.RandomRotate90(p=0.08),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_valid_tfms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class FruitDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform: A.Compose,
        *,
        easy_strong_transform: Optional[A.Compose] = None,
        hard_light_transform: Optional[A.Compose] = None,
        easy_labels: Optional[Set[int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.easy_strong_transform = easy_strong_transform
        self.hard_light_transform = hard_light_transform
        self.easy_labels = set(easy_labels or set())
        self.mode = "base"

    def set_mode(self, mode: str) -> None:
        if mode not in {"base", "class_aware_harden"}:
            raise ValueError(f"Unknown dataset mode: {mode}")
        self.mode = mode

    def _pick_transform(self, label: int) -> A.Compose:
        if self.mode != "class_aware_harden":
            return self.transform
        if self.easy_strong_transform is None or self.hard_light_transform is None:
            return self.transform
        if label in self.easy_labels:
            return self.easy_strong_transform
        return self.hard_light_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label"])
        image = np.array(Image.open(self.img_dir / row["image_id"]).convert("RGB"))
        tfm = self._pick_transform(label)
        image = tfm(image=image)["image"]
        return image, label, row["image_id"]


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], lam


def rand_bbox(w: int, h: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return int(x1), int(y1), int(x2), int(y2)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx].clone()
    _, _, h, w = x.shape
    x1, y1, x2b, y2b = rand_bbox(w=w, h=h, lam=lam)
    x[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]
    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / float(w * h))
    return x, y, y[idx], float(lam_adj)


def mixed_criterion(criterion, pred, y_a, y_b, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("Use first_step and second_step explicitly for SAM.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((scale * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def set_epoch_lr(optimizer, epoch: int, cfg: Config) -> float:
    if epoch < cfg.warmup_epochs:
        lr = cfg.lr * float(epoch + 1) / float(max(1, cfg.warmup_epochs))
    elif epoch < cfg.stage1_epochs:
        t = (epoch - cfg.warmup_epochs + 1) / float(max(1, cfg.stage1_epochs - cfg.warmup_epochs))
        lr_min = cfg.lr * 0.20
        lr = lr_min + (cfg.lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        stage2_lr = cfg.lr / max(1.0, cfg.lr_drop_factor)
        stage2_len = max(1, cfg.epochs - cfg.stage1_epochs)
        t2 = (epoch - cfg.stage1_epochs + 1) / float(stage2_len)
        lr2_min = stage2_lr * 0.10
        lr = lr2_min + (stage2_lr - lr2_min) * 0.5 * (1.0 + math.cos(math.pi * t2))

    for g in optimizer.param_groups:
        g["lr"] = lr
    return float(lr)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_channels_last: bool,
) -> Dict:
    model.eval()
    losses = []
    logits_all = []
    y_all = []
    ids_all: List[str] = []
    for x, y, image_ids in loader:
        x = x.to(device)
        y = y.to(device)
        if use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
        ids_all.extend(list(image_ids))

    logits_np = torch.cat(logits_all).numpy()
    y_np = torch.cat(y_all).numpy()
    y_pred = logits_np.argmax(1)
    probs = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    conf = probs.max(axis=1)
    return {
        "val_loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_np, y_pred)),
        "f1_macro": float(f1_score(y_np, y_pred, average="macro")),
        "y_true": y_np,
        "y_pred": y_pred,
        "logits": logits_np,
        "confidence": conf,
        "image_ids": ids_all,
    }


def class_report_df(y_true: np.ndarray, y_pred: np.ndarray, label_to_name: Dict[int, str]) -> pd.DataFrame:
    labels = sorted(label_to_name.keys())
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[label_to_name[x] for x in labels],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for lbl in labels:
        name = label_to_name[lbl]
        r = rep[name]
        rows.append(
            {
                "label": lbl,
                "class_name": name,
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1-score"]),
                "support": int(r["support"]),
            }
        )
    return pd.DataFrame(rows).sort_values("f1")


def top_confusions_df(y_true: np.ndarray, y_pred: np.ndarray, label_to_name: Dict[int, str], topk: int = 30) -> pd.DataFrame:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df[df["true"] != df["pred"]]
    if df.empty:
        return pd.DataFrame(columns=["count", "true_label", "pred_label", "true_name", "pred_name"])
    g = df.value_counts(["true", "pred"]).reset_index(name="count").sort_values("count", ascending=False).head(topk)
    g["true_name"] = g["true"].map(label_to_name)
    g["pred_name"] = g["pred"].map(label_to_name)
    g = g.rename(columns={"true": "true_label", "pred": "pred_label"})
    return g


def create_model(name: str, num_classes: int) -> nn.Module:
    fallback = {
        "convnext_small.fb_in22k_ft_in1k": "convnext_small",
        "tf_efficientnetv2_s.in21k_ft_in1k": "tf_efficientnetv2_s",
        "resnet50.a1_in1k": "resnet50",
    }
    try:
        return timm.create_model(name, pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)
    except Exception:
        return timm.create_model(
            fallback.get(name, name),
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.2,
            drop_path_rate=0.2,
        )


def main() -> None:
    args = parse_args()
    cfg = Config(
        base=args.base,
        clean_variant=args.clean_variant,
        folds_csv=args.folds_csv,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
        fold_idx=args.fold_idx,
        seed=args.seed,
        model_name=args.model_name,
        img_size=args.img_size,
        use_channels_last=args.use_channels_last,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        stage1_epochs=args.stage1_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        lr_drop_factor=args.lr_drop_factor,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        use_cutmix=not args.no_cutmix,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        class_aware_harden_last_epochs=args.class_aware_harden_last_epochs,
        class_aware_easy_topk=args.class_aware_easy_topk,
        class_aware_easy_labels_csv=args.class_aware_easy_labels,
        use_sam=not args.no_sam,
        sam_rho=args.sam_rho,
        sam_adaptive=args.sam_adaptive,
        use_swa=not args.no_swa,
        swa_start_epoch=args.swa_start_epoch,
        out_dir=args.out_dir,
    )

    if cfg.stage1_epochs >= cfg.epochs:
        raise ValueError("stage1_epochs must be < epochs")
    if cfg.fold_idx < 0:
        raise ValueError("fold_idx must be >= 0")
    if cfg.class_aware_harden_last_epochs < 0:
        raise ValueError("class_aware_harden_last_epochs must be >= 0")
    if cfg.class_aware_harden_last_epochs >= cfg.epochs:
        raise ValueError("class_aware_harden_last_epochs must be < epochs")
    if cfg.class_aware_easy_topk < 0:
        raise ValueError("class_aware_easy_topk must be >= 0")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = resolve_device(args.device)
    if device.type in {"mps", "cuda"}:
        torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    print("Config:", json.dumps(asdict(cfg), ensure_ascii=False, indent=2), flush=True)
    print("Device:", device, flush=True)

    paths = build_paths(Path(cfg.base), cfg.clean_variant)
    for k, v in paths.items():
        print(f"{k}: {v} exists={v.exists()}", flush=True)

    df = load_train_df(paths)
    df = make_or_load_folds(df, n_splits=cfg.n_splits, seed=cfg.fold_seed, folds_csv=cfg.folds_csv)
    if cfg.fold_idx >= df["fold"].nunique():
        raise ValueError(f"fold_idx={cfg.fold_idx} out of range for n_folds={df['fold'].nunique()}")
    df.to_csv(out_dir / "folds_used.csv", index=False)

    tr_df = df[df["fold"] != cfg.fold_idx].reset_index(drop=True)
    va_df = df[df["fold"] == cfg.fold_idx].reset_index(drop=True)
    print(f"Train size={len(tr_df)}, Val size={len(va_df)}, Fold={cfg.fold_idx}", flush=True)

    label_to_name = df.groupby("label")["class_name"].agg(lambda s: s.mode().iat[0]).to_dict()

    label_counts = df["label"].value_counts().sort_index()
    num_classes = int(label_counts.shape[0])
    n = len(df)
    class_weights = {k: n / (num_classes * v) for k, v in label_counts.to_dict().items()}
    max_w = max(class_weights.values())
    class_weights = {k: v / max_w for k, v in class_weights.items()}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    print("Class counts:", label_counts.to_dict(), flush=True)
    print("Class weights:", class_weights, flush=True)

    easy_labels: Set[int] = set()
    class_aware_hardening_enabled = cfg.class_aware_harden_last_epochs > 0
    if class_aware_hardening_enabled:
        raw_easy = [x.strip() for x in cfg.class_aware_easy_labels_csv.split(",") if x.strip()]
        if raw_easy:
            easy_labels = {int(x) for x in raw_easy}
        else:
            tr_counts = tr_df["label"].value_counts().sort_values(ascending=False)
            easy_labels = {int(x) for x in tr_counts.head(cfg.class_aware_easy_topk).index.tolist()}
        easy_labels = {x for x in easy_labels if 0 <= x < num_classes}
        easy_names = [label_to_name.get(x, str(x)) for x in sorted(easy_labels)]
        print(
            f"Class-aware hardening enabled: last_epochs={cfg.class_aware_harden_last_epochs}, "
            f"easy_labels={sorted(easy_labels)} ({easy_names})",
            flush=True,
        )

    tr_ds = FruitDataset(
        tr_df,
        paths["train_images_dir"],
        transform=build_train_tfms_no_color(cfg.img_size),
        easy_strong_transform=build_train_tfms_no_color_easy_strong(cfg.img_size) if class_aware_hardening_enabled else None,
        hard_light_transform=build_train_tfms_no_color_hard_light(cfg.img_size) if class_aware_hardening_enabled else None,
        easy_labels=easy_labels if class_aware_hardening_enabled else None,
    )
    va_ds = FruitDataset(va_df, paths["train_images_dir"], transform=build_valid_tfms(cfg.img_size))

    dl_kwargs = {"num_workers": cfg.num_workers, "persistent_workers": cfg.num_workers > 0}
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    if cfg.use_weighted_sampler:
        sample_weights = tr_df["label"].map(class_weights).values.astype(np.float32)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True, **dl_kwargs)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)

    model = create_model(cfg.model_name, num_classes=num_classes).to(device)
    if cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=cfg.label_smoothing)

    if cfg.use_sam:
        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            rho=cfg.sam_rho,
            adaptive=cfg.sam_adaptive,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        base_optimizer = optimizer.base_optimizer
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        base_optimizer = optimizer

    swa_model = None
    swa_updates = 0
    if cfg.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "best_by_val_loss.pt"
    epoch_rows = []

    for epoch in range(cfg.epochs):
        lr_now = set_epoch_lr(base_optimizer, epoch, cfg)
        model.train()

        class_aware_active = bool(
            class_aware_hardening_enabled and (epoch + 1) > (cfg.epochs - cfg.class_aware_harden_last_epochs)
        )
        tr_ds.set_mode("class_aware_harden" if class_aware_active else "base")

        tr_losses = []
        n_mixup = 0
        n_cutmix = 0
        n_plain = 0

        pbar = tqdm(tr_loader, desc=f"train e{epoch+1}/{cfg.epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            if cfg.use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()

            aug_mode = "plain"
            y_a = y
            y_b = y
            lam = 1.0

            r = random.random()
            if cfg.use_cutmix and r < cfg.cutmix_prob:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=cfg.cutmix_alpha)
                aug_mode = "cutmix"
                n_cutmix += 1
            elif cfg.use_mixup and r < (cfg.cutmix_prob + cfg.mixup_prob):
                x, y_a, y_b, lam = mixup_data(x, y, alpha=cfg.mixup_alpha)
                aug_mode = "mixup"
                n_mixup += 1
            else:
                n_plain += 1

            if not cfg.use_sam:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                if aug_mode == "plain":
                    loss = criterion(logits, y)
                else:
                    loss = mixed_criterion(criterion, logits, y_a, y_b, lam)
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                tr_losses.append(loss.item())
            else:
                optimizer.zero_grad(set_to_none=True)
                logits1 = model(x)
                if aug_mode == "plain":
                    loss1 = criterion(logits1, y)
                else:
                    loss1 = mixed_criterion(criterion, logits1, y_a, y_b, lam)
                loss1.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.first_step(zero_grad=True)

                logits2 = model(x)
                if aug_mode == "plain":
                    loss2 = criterion(logits2, y)
                else:
                    loss2 = mixed_criterion(criterion, logits2, y_a, y_b, lam)
                loss2.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.second_step(zero_grad=True)
                tr_losses.append(loss2.item())

            pbar.set_postfix(loss=f"{np.mean(tr_losses):.4f}", lr=f"{lr_now:.2e}", mode=aug_mode)

        val_info = evaluate(model, va_loader, criterion, device, cfg.use_channels_last)
        tr_loss = float(np.mean(tr_losses))
        val_loss = float(val_info["val_loss"])
        val_acc = float(val_info["acc"])
        val_f1 = float(val_info["f1_macro"])

        if cfg.use_swa and swa_model is not None and (epoch + 1) >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_updates += 1

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
            "lr": lr_now,
            "mixup_batches": n_mixup,
            "cutmix_batches": n_cutmix,
            "plain_batches": n_plain,
            "swa_active": int(cfg.use_swa and (epoch + 1) >= cfg.swa_start_epoch),
            "sam_enabled": int(cfg.use_sam),
            "class_aware_harden_active": int(class_aware_active),
        }
        epoch_rows.append(epoch_row)
        print(
            f"[epoch {epoch+1:02d}/{cfg.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"lr={lr_now:.2e} mixup={n_mixup} cutmix={n_cutmix} plain={n_plain} "
            f"swa_active={epoch_row['swa_active']} sam={epoch_row['sam_enabled']} "
            f"class_harden={epoch_row['class_aware_harden_active']}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1_macro": val_f1,
                    "cfg": asdict(cfg),
                },
                best_ckpt_path,
            )

    pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_log.csv", index=False)

    # Evaluate best checkpoint.
    best_model = create_model(cfg.model_name, num_classes=num_classes).to(device)
    if cfg.use_channels_last:
        best_model = best_model.to(memory_format=torch.channels_last)
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_eval = evaluate(best_model, va_loader, criterion, device, cfg.use_channels_last)

    # Evaluate SWA model if available.
    swa_eval = None
    swa_ckpt_path = out_dir / "swa_model.pt"
    if cfg.use_swa and swa_model is not None and swa_updates > 0:
        tr_ds.set_mode("base")
        print(f"Updating BN stats for SWA model (updates={swa_updates}) ...", flush=True)
        torch.optim.swa_utils.update_bn(tr_loader, swa_model, device=device)
        swa_eval = evaluate(swa_model, va_loader, criterion, device, cfg.use_channels_last)
        torch.save({"model_state_dict": swa_model.module.state_dict(), "cfg": asdict(cfg)}, swa_ckpt_path)
        print(
            f"[SWA] val_loss={swa_eval['val_loss']:.4f} val_acc={swa_eval['acc']:.4f} val_f1={swa_eval['f1_macro']:.4f}",
            flush=True,
        )

    # Pick final model by val_loss.
    final_name = "best_checkpoint"
    final_eval = best_eval
    if swa_eval is not None and swa_eval["val_loss"] < best_eval["val_loss"]:
        final_name = "swa_model"
        final_eval = swa_eval

    # Save detailed analysis artifacts for val fold.
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

    class_df = class_report_df(final_eval["y_true"], final_eval["y_pred"], label_to_name)
    class_df.to_csv(out_dir / "val_class_report.csv", index=False)
    conf_df = top_confusions_df(final_eval["y_true"], final_eval["y_pred"], label_to_name, topk=40)
    conf_df.to_csv(out_dir / "val_top_confusions.csv", index=False)
    val_logits = final_eval["logits"].astype(np.float32)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy().astype(np.float32)
    np.save(out_dir / "val_logits.npy", val_logits)
    np.save(out_dir / "val_probs.npy", val_probs)
    np.save(out_dir / "val_labels.npy", final_eval["y_true"].astype(np.int64))

    summary = {
        "device": str(device),
        "final_model_selected": final_name,
        "best_checkpoint_metrics": {
            "val_loss": float(best_eval["val_loss"]),
            "val_acc": float(best_eval["acc"]),
            "val_f1_macro": float(best_eval["f1_macro"]),
            "best_ckpt_epoch": int(best_ckpt.get("epoch", -1)),
        },
        "swa_metrics": None
        if swa_eval is None
        else {
            "val_loss": float(swa_eval["val_loss"]),
            "val_acc": float(swa_eval["acc"]),
            "val_f1_macro": float(swa_eval["f1_macro"]),
            "swa_updates": int(swa_updates),
        },
        "final_metrics": {
            "val_loss": float(final_eval["val_loss"]),
            "val_acc": float(final_eval["acc"]),
            "val_f1_macro": float(final_eval["f1_macro"]),
            "val_errors": int((final_eval["y_true"] != final_eval["y_pred"]).sum()),
            "val_size": int(len(final_eval["y_true"])),
        },
        "artifacts": {
            "config": str(out_dir / "config.json"),
            "epoch_log": str(out_dir / "epoch_log.csv"),
            "best_checkpoint": str(best_ckpt_path),
            "swa_checkpoint": str(swa_ckpt_path) if swa_ckpt_path.exists() else "",
            "val_predictions": str(out_dir / "val_predictions.csv"),
            "val_logits_npy": str(out_dir / "val_logits.npy"),
            "val_probs_npy": str(out_dir / "val_probs.npy"),
            "val_labels_npy": str(out_dir / "val_labels.npy"),
            "val_class_report": str(out_dir / "val_class_report.csv"),
            "val_top_confusions": str(out_dir / "val_top_confusions.csv"),
            "folds_used": str(out_dir / "folds_used.csv"),
        },
        "class_aware_hardening": {
            "enabled": bool(class_aware_hardening_enabled),
            "last_epochs": int(cfg.class_aware_harden_last_epochs),
            "easy_topk": int(cfg.class_aware_easy_topk),
            "easy_labels": sorted(int(x) for x in easy_labels) if class_aware_hardening_enabled else [],
            "easy_label_names": [label_to_name.get(int(x), str(x)) for x in sorted(easy_labels)]
            if class_aware_hardening_enabled
            else [],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ONE-FOLD TRAINING DONE ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
