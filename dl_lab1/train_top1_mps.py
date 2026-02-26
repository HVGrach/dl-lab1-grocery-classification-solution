#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

TQDM_OFF = os.getenv("TQDM_DISABLE", "0") == "1"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    seed: int = 42
    clean_variant: str = "strict"  # strict | aggressive | raw

    n_folds: int = 5
    fold_seed: int = 42

    batch_size: int = 16
    epochs: int = 14
    warmup_epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05

    num_workers: int = 4
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.3
    disable_mixup_last_epochs: int = 2
    grad_clip_norm: float = 1.0
    use_channels_last: bool = True
    aug_profile: str = "full"  # full | no_color

    ensemble_trials: int = 5000
    out_dir: str = "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps"

    model_configs: Tuple[Dict, ...] = (
        {
            "alias": "convnext_small",
            "timm_name": "convnext_small.fb_in22k_ft_in1k",
            "img_size": 224,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
            "use_channels_last": True,
        },
        {
            "alias": "effnetv2_s",
            "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
            "img_size": 224,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
            "use_channels_last": False,
        },
        {
            "alias": "resnet50",
            "timm_name": "resnet50.a1_in1k",
            "img_size": 224,
            "drop_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_channels_last": False,
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--clean-variant", type=str, default="strict")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-models", type=int, default=0)
    p.add_argument("--ensemble-trials", type=int, default=5000)
    p.add_argument("--out-dir", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aug-profile", type=str, default="full", choices=["full", "no_color"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


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
        "test_csv": base / "test.csv",
        "sample_submission": base / "sample_submission.csv",
        "train_images_dir": base / "train" / "train",
        "test_images_dir": base / "test_images" / "test_images",
    }


def load_train_df(paths: Dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(paths["train_csv"])
    df["label"] = df["label"].astype(int)
    split = df["image_id"].str.split("/", expand=True)
    df["class_name"] = split[0]
    df["plu"] = split[1]
    return df


def load_test_df(paths: Dict[str, Path]) -> pd.DataFrame:
    return pd.read_csv(paths["test_csv"])


def make_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1
    df["strat_key"] = df["label"].astype(str) + "_" + df["plu"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["strat_key"])):
        df.loc[df.index[val_idx], "fold"] = fold
    return df


def build_train_tfms(img_size: int, aug_profile: str = "full") -> A.Compose:
    tfms: List[A.BasicTransform] = [
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.65, 1.0), ratio=(0.8, 1.25), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, border_mode=0, p=0.35),
    ]
    if aug_profile == "full":
        tfms.append(
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                ],
                p=0.6,
            )
        )
    elif aug_profile != "no_color":
        raise ValueError(f"Unknown aug_profile: {aug_profile}")

    tfms.extend(
        [
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.ImageCompression(quality_range=(70, 100), p=1.0),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(0.04, 0.12),
                hole_width_range=(0.04, 0.12),
                fill=0,
                p=0.2,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(tfms)


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
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform: A.Compose, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = np.array(Image.open(self.img_dir / row["image_id"]).convert("RGB"))
        image = self.transform(image=image)["image"]
        if self.is_test:
            return image, row["image_id"]
        return image, int(row["label"])


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=2, min_lr=1e-6, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch + 1
        out = []
        for base_lr in self.base_lrs:
            if e <= self.warmup_epochs:
                lr = base_lr * e / max(1, self.warmup_epochs)
            else:
                p = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))
            out.append(lr)
        return out


def create_model(cfg: Dict, num_classes: int) -> nn.Module:
    fallback = {
        "convnext_small.fb_in22k_ft_in1k": "convnext_small",
        "tf_efficientnetv2_s.in21k_ft_in1k": "tf_efficientnetv2_s",
        "resnet50.a1_in1k": "resnet50",
    }
    name = cfg["timm_name"]
    try:
        return timm.create_model(
            name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=cfg.get("drop_rate", 0.2),
            drop_path_rate=cfg.get("drop_path_rate", 0.2),
        )
    except Exception:
        return timm.create_model(
            fallback.get(name, name),
            pretrained=True,
            num_classes=num_classes,
            drop_rate=cfg.get("drop_rate", 0.2),
            drop_path_rate=cfg.get("drop_path_rate", 0.2),
        )


def evaluate_logits(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    pred = logits.argmax(1)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
    }


@torch.no_grad()
def predict_logits(model, loader, device, tta: bool = True, use_channels_last: bool = False):
    model.eval()
    all_logits = []
    all_ids = []
    for x, ids in tqdm(loader, leave=False, disable=TQDM_OFF):
        x = x.to(device)
        if use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        logits = model(x)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
        all_logits.append(logits.detach().cpu())
        all_ids.extend(ids)
    return torch.cat(all_logits).numpy(), all_ids


def train_one_fold(
    model_cfg: Dict,
    fold: int,
    df: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    class_weights: Dict[int, float],
    class_weights_tensor: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> Path:
    tr = df[df["fold"] != fold].reset_index(drop=True)
    va = df[df["fold"] == fold].reset_index(drop=True)

    img_size = model_cfg["img_size"]
    tr_ds = FruitDataset(
        tr,
        paths["train_images_dir"],
        transform=build_train_tfms(img_size, aug_profile=cfg.aug_profile),
        is_test=False,
    )
    va_ds = FruitDataset(va, paths["train_images_dir"], transform=build_valid_tfms(img_size), is_test=False)

    sw = tr["label"].map(class_weights).values.astype(np.float32)
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)

    dl_kwargs = {
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        **dl_kwargs,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    use_channels_last = bool(model_cfg.get("use_channels_last", cfg.use_channels_last))
    model = create_model(model_cfg, num_classes=num_classes).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device), label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, total_epochs=cfg.epochs, warmup_epochs=cfg.warmup_epochs, min_lr=cfg.lr * 0.03)

    fold_dir = Path(cfg.out_dir) / model_cfg["alias"] / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_by_val_loss.pt"
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        pbar = tqdm(
            tr_loader,
            desc=f"{model_cfg['alias']} f{fold} e{epoch+1}/{cfg.epochs}",
            leave=False,
            disable=TQDM_OFF,
        )
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            use_mix = (
                cfg.use_mixup
                and (epoch < max(0, cfg.epochs - cfg.disable_mixup_last_epochs))
                and (random.random() < cfg.mixup_prob)
            )
            if use_mix:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=cfg.mixup_alpha)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()

            logits = model(x)
            if use_mix:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y)

            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            tr_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(tr_losses):.4f}")

        scheduler.step()

        model.eval()
        va_losses = []
        va_logits = []
        va_targets = []
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device)
                if use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                else:
                    x = x.contiguous()
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_losses.append(loss.item())
                va_logits.append(logits.detach().cpu())
                va_targets.append(y.detach().cpu())

        va_logits_np = torch.cat(va_logits).numpy()
        va_targets_np = torch.cat(va_targets).numpy()
        m = evaluate_logits(va_targets_np, va_logits_np)
        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        print(
            f"[{model_cfg['alias']}][fold={fold}] epoch={epoch+1}/{cfg.epochs} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={m['acc']:.4f} val_f1m={m['f1_macro']:.4f}",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": va_loss,
                    "val_acc": m["acc"],
                    "model_cfg": model_cfg,
                    "cfg": asdict(cfg),
                },
                best_path,
            )

    del model, tr_loader, va_loader, tr_ds, va_ds
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return best_path


def run_model_cv(
    model_cfg: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    class_weights: Dict[int, float],
    class_weights_tensor: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    alias = model_cfg["alias"]
    use_channels_last = bool(model_cfg.get("use_channels_last", cfg.use_channels_last))
    oof_logits = np.zeros((len(train_df), num_classes), dtype=np.float32)
    oof_targets = train_df["label"].values.astype(np.int64)

    test_ds = FruitDataset(test_df, paths["test_images_dir"], transform=build_valid_tfms(model_cfg["img_size"]), is_test=True)
    dl_kwargs = {
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)
    test_logits_folds = []

    for fold in range(cfg.n_folds):
        best_path = train_one_fold(
            model_cfg=model_cfg,
            fold=fold,
            df=train_df,
            cfg=cfg,
            paths=paths,
            class_weights=class_weights,
            class_weights_tensor=class_weights_tensor,
            num_classes=num_classes,
            device=device,
        )

        model = create_model(model_cfg, num_classes=num_classes).to(device)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        va_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
        va_ds = FruitDataset(va_df, paths["train_images_dir"], transform=build_valid_tfms(model_cfg["img_size"]), is_test=False)
        va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)

        logits_fold = []
        with torch.no_grad():
            for x, _ in tqdm(va_loader, desc=f"OOF {alias} fold{fold}", leave=False, disable=TQDM_OFF):
                x = x.to(device)
                if use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                else:
                    x = x.contiguous()
                logits_fold.append(model(x).detach().cpu())
        logits_fold = torch.cat(logits_fold).numpy()
        oof_logits[train_df["fold"].values == fold] = logits_fold

        test_logits, _ = predict_logits(model, test_loader, device, tta=True, use_channels_last=use_channels_last)
        test_logits_folds.append(test_logits)

        del model, va_ds, va_loader
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    test_logits = np.mean(np.stack(test_logits_folds, axis=0), axis=0)

    model_dir = Path(cfg.out_dir) / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "oof_logits.npy", oof_logits)
    np.save(model_dir / "oof_targets.npy", oof_targets)
    np.save(model_dir / "test_logits.npy", test_logits)

    metrics = evaluate_logits(oof_targets, oof_logits)
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"OOF [{alias}] acc={metrics['acc']:.5f} f1_macro={metrics['f1_macro']:.5f}", flush=True)
    return metrics


def search_best_weights(oof_list: List[np.ndarray], y_true: np.ndarray, trials: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(oof_list)
    best_w = np.ones(n) / n
    best_score = accuracy_score(y_true, np.mean(np.stack(oof_list, axis=0), axis=0).argmax(1))
    for _ in tqdm(range(trials), desc="ensemble weight search", disable=TQDM_OFF):
        w = rng.dirichlet(np.ones(n))
        blend = np.zeros_like(oof_list[0])
        for wi, lg in zip(w, oof_list):
            blend += wi * lg
        s = accuracy_score(y_true, blend.argmax(1))
        if s > best_score:
            best_score = s
            best_w = w
    return best_w, float(best_score)


def has_cached_outputs(out_dir: Path, alias: str) -> bool:
    model_dir = out_dir / alias
    required = [
        model_dir / "oof_logits.npy",
        model_dir / "oof_targets.npy",
        model_dir / "test_logits.npy",
    ]
    return all(p.exists() for p in required)


def load_or_build_metrics(out_dir: Path, alias: str) -> Dict[str, float]:
    model_dir = out_dir / alias
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    y_true = np.load(model_dir / "oof_targets.npy")
    oof_logits = np.load(model_dir / "oof_logits.npy")
    metrics = evaluate_logits(y_true, oof_logits)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        clean_variant=args.clean_variant,
        n_folds=2 if args.smoke else args.folds,
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ensemble_trials=300 if args.smoke else args.ensemble_trials,
        out_dir=args.out_dir,
        aug_profile=args.aug_profile,
    )
    if args.max_models > 0:
        cfg.model_configs = cfg.model_configs[: args.max_models]

    print("Config:")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Run this script outside sandbox with full system access.")

    device = torch.device("mps")
    print("Using device:", device, flush=True)
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    paths = build_paths(Path(args.base), cfg.clean_variant)
    for k, v in paths.items():
        print(f"{k}: {v} exists={v.exists()}")

    train_df = load_train_df(paths)
    test_df = load_test_df(paths)
    train_df = make_folds(train_df, n_splits=cfg.n_folds, seed=cfg.fold_seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / f"folds_{cfg.clean_variant}_{cfg.n_folds}f.csv", index=False)

    label_counts = train_df["label"].value_counts().sort_index()
    num_classes = label_counts.shape[0]
    n = len(train_df)
    class_weights = {k: n / (num_classes * v) for k, v in label_counts.to_dict().items()}
    max_w = max(class_weights.values())
    class_weights = {k: v / max_w for k, v in class_weights.items()}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32)
    print("Class counts:", label_counts.to_dict())
    print("Class weights:", class_weights)

    all_metrics = {}
    aliases = []
    # convnext_small already finished; keep its cached logits/metrics and skip retraining.
    skip_if_cached = {"convnext_small"}
    for model_cfg in cfg.model_configs:
        alias = model_cfg["alias"]
        aliases.append(alias)
        if alias in skip_if_cached and has_cached_outputs(out_dir, alias):
            print(f"\n=== SKIP MODEL: {alias} (using cached outputs) ===", flush=True)
            metrics = load_or_build_metrics(out_dir, alias)
            all_metrics[alias] = metrics
            print(f"OOF [{alias}] acc={metrics['acc']:.5f} f1_macro={metrics['f1_macro']:.5f}", flush=True)
            continue

        print(f"\n=== TRAIN MODEL: {alias} ===", flush=True)
        metrics = run_model_cv(
            model_cfg=model_cfg,
            train_df=train_df,
            test_df=test_df,
            cfg=cfg,
            paths=paths,
            class_weights=class_weights,
            class_weights_tensor=class_weights_tensor,
            num_classes=num_classes,
            device=device,
        )
        all_metrics[alias] = metrics

    with (out_dir / "all_model_oof_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # Ensemble
    oof_list = []
    test_list = []
    y_true = np.load(out_dir / aliases[0] / "oof_targets.npy")
    for alias in aliases:
        oof_list.append(np.load(out_dir / alias / "oof_logits.npy"))
        test_list.append(np.load(out_dir / alias / "test_logits.npy"))

    best_w, best_oof = search_best_weights(oof_list, y_true, trials=cfg.ensemble_trials, seed=cfg.seed)
    best_w = best_w / (best_w.sum() + 1e-12)
    blend_test = np.zeros_like(test_list[0])
    for wi, lg in zip(best_w, test_list):
        blend_test += wi * lg
    pred = blend_test.argmax(1)

    sub = pd.read_csv(paths["sample_submission"])
    sub["label"] = pred
    sub_path = out_dir / "submission_ensemble_oof_optimized.csv"
    sub.to_csv(sub_path, index=False)

    with (out_dir / "ensemble_weights.json").open("w", encoding="utf-8") as f:
        json.dump({a: float(w) for a, w in zip(aliases, best_w)}, f, ensure_ascii=False, indent=2)
    with (out_dir / "ensemble_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_oof_acc": best_oof, "aliases": aliases}, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("Model OOF metrics:", json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print("Ensemble best OOF acc:", best_oof)
    print("Submission:", sub_path)


if __name__ == "__main__":
    main()
