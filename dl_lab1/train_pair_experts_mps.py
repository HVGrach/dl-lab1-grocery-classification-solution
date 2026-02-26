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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
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
class PairModelCfg:
    alias: str
    pos_label: int
    neg_label: int
    timm_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    img_size: int = 224
    drop_rate: float = 0.2
    drop_path_rate: float = 0.2
    use_channels_last: bool = False
    pretrained: bool = True


@dataclass
class Config:
    seed: int = 42
    clean_variant: str = "strict"

    n_folds: int = 5
    fold_seed: int = 42

    batch_size: int = 24
    epochs: int = 14
    warmup_epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    num_workers: int = 0
    tta: bool = True

    out_dir: str = "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_pair_experts_mps"


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


class PairDataset(Dataset):
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
        # Keep binary targets in fp32 to avoid unsupported fp64 tensors on MPS.
        return image, np.float32(row["target"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--clean-variant", type=str, default="strict")
    p.add_argument("--pairs", type=str, default="6:5,14:12,9:0")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pair-model", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    p.add_argument("--pair-img-size", type=int, default=224)
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_pair_experts_mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def parse_pair_tokens(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split(":")
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("No pairs provided.")
    return out


def pair_alias(pos_label: int, neg_label: int) -> str:
    known = {
        (6, 5): "kiwi_vs_potato",
        (14, 12): "redapple_vs_tomato",
        (9, 0): "mandarin_vs_orange",
    }
    return known.get((pos_label, neg_label), f"pair_{pos_label}_vs_{neg_label}")


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


def build_train_tfms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), ratio=(0.9, 1.12), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(0.03, 0.03), rotate=(-12, 12), p=0.4),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(75, 100), p=1.0),
                ],
                p=0.2,
            ),
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


def load_train_df(paths: Dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(paths["train_csv"])
    df["label"] = df["label"].astype(int)
    split = df["image_id"].str.split("/", expand=True)
    df["class_name"] = split[0]
    df["plu"] = split[1]
    return df


def load_test_df(paths: Dict[str, Path]) -> pd.DataFrame:
    return pd.read_csv(paths["test_csv"])


def make_pair_df(df: pd.DataFrame, pos_label: int, neg_label: int) -> pd.DataFrame:
    out = df[df["label"].isin([pos_label, neg_label])].copy()
    out["target"] = (out["label"] == pos_label).astype(int)
    out = out.reset_index(drop=True)
    return out


def assign_pair_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1
    # Try label+plu first, fallback to label-only if classes become too sparse for StratifiedKFold.
    key = out["target"].astype(str) + "_" + out["plu"].astype(str)
    if key.value_counts().min() < n_splits:
        key = out["target"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, va_idx) in enumerate(skf.split(out, key)):
        out.loc[out.index[va_idx], "fold"] = fold
    return out


def create_model(cfg: PairModelCfg) -> nn.Module:
    fallback = {
        "convnext_small.fb_in22k_ft_in1k": "convnext_small",
        "tf_efficientnetv2_s.in21k_ft_in1k": "tf_efficientnetv2_s",
        "resnet50.a1_in1k": "resnet50",
    }
    name = cfg.timm_name
    try:
        model = timm.create_model(
            name,
            pretrained=cfg.pretrained,
            num_classes=1,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
        )
    except Exception:
        model = timm.create_model(fallback.get(name, name), pretrained=False, num_classes=1, drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate)
    return model


def predict_proba_binary(model, loader, device: torch.device, tta: bool, use_channels_last: bool) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    probs_all = []
    ids_all: List[str] = []
    with torch.no_grad():
        for x, ids in tqdm(loader, leave=False, disable=TQDM_OFF):
            x = x.to(device)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()
            logits = model(x)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            if tta:
                logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            probs_all.append(probs)
            ids_all.extend(ids)
    return np.concatenate(probs_all), ids_all


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = (prob >= thr).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    best = {"thr": 0.5, "f1": -1.0, "acc": -1.0}
    for thr in np.linspace(0.2, 0.8, 121):
        m = binary_metrics(y_true, prob, float(thr))
        if (m["f1"] > best["f1"] + 1e-12) or (abs(m["f1"] - best["f1"]) <= 1e-12 and m["acc"] > best["acc"]):
            best = {"thr": float(thr), "f1": float(m["f1"]), "acc": float(m["acc"])}
    return best


def train_one_fold(
    pair_cfg: PairModelCfg,
    fold: int,
    pair_df: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
) -> Path:
    tr = pair_df[pair_df["fold"] != fold].reset_index(drop=True)
    va = pair_df[pair_df["fold"] == fold].reset_index(drop=True)

    tr_ds = PairDataset(tr, paths["train_images_dir"], transform=build_train_tfms(pair_cfg.img_size), is_test=False)
    va_ds = PairDataset(va, paths["train_images_dir"], transform=build_valid_tfms(pair_cfg.img_size), is_test=False)

    dl_kwargs = {"num_workers": cfg.num_workers, "persistent_workers": cfg.num_workers > 0}
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)

    model = create_model(pair_cfg).to(device)
    if pair_cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    pos = float((tr["target"] == 1).sum())
    neg = float((tr["target"] == 0).sum())
    pos_weight_value = max(1e-6, neg / max(1.0, pos))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
        min_lr=cfg.lr * 0.03,
    )

    fold_dir = Path(cfg.out_dir) / pair_cfg.alias / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_by_val_loss.pt"
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses: List[float] = []
        pbar = tqdm(
            tr_loader,
            desc=f"{pair_cfg.alias} f{fold} e{epoch+1}/{cfg.epochs}",
            leave=False,
            disable=TQDM_OFF,
        )
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device=device, dtype=torch.float32).unsqueeze(1)
            if pair_cfg.use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            tr_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(tr_losses):.4f}")

        scheduler.step()

        model.eval()
        va_losses: List[float] = []
        va_probs: List[np.ndarray] = []
        va_targets: List[np.ndarray] = []
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device)
                if pair_cfg.use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                else:
                    x = x.contiguous()
                y = y.to(device=device, dtype=torch.float32).unsqueeze(1)
                logits = model(x)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                loss = criterion(logits, y)
                va_losses.append(float(loss.item()))
                va_probs.append(torch.sigmoid(logits).squeeze(1).detach().cpu().numpy())
                va_targets.append(y.squeeze(1).detach().cpu().numpy())

        va_prob_np = np.concatenate(va_probs)
        va_target_np = np.concatenate(va_targets).astype(int)
        va_m = binary_metrics(va_target_np, va_prob_np, thr=0.5)
        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        print(
            f"[{pair_cfg.alias}][fold={fold}] epoch={epoch+1}/{cfg.epochs} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_acc={va_m['acc']:.4f} val_f1={va_m['f1']:.4f} val_auc={va_m['roc_auc']:.4f}",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": va_loss,
                    "val_metrics": va_m,
                    "pair_cfg": asdict(pair_cfg),
                    "cfg": asdict(cfg),
                },
                best_path,
            )

    del model, tr_loader, va_loader, tr_ds, va_ds
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return best_path


def run_pair_cv(
    pair_cfg: PairModelCfg,
    pair_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, float]:
    alias = pair_cfg.alias
    oof_prob = np.zeros(len(pair_df), dtype=np.float32)
    oof_target = pair_df["target"].values.astype(np.int64)

    test_ds = PairDataset(test_df, paths["test_images_dir"], transform=build_valid_tfms(pair_cfg.img_size), is_test=True)
    dl_kwargs = {"num_workers": cfg.num_workers, "persistent_workers": cfg.num_workers > 0}
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)
    test_prob_folds: List[np.ndarray] = []

    for fold in range(cfg.n_folds):
        best_path = train_one_fold(pair_cfg=pair_cfg, fold=fold, pair_df=pair_df, cfg=cfg, paths=paths, device=device)

        model = create_model(pair_cfg).to(device)
        if pair_cfg.use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        va_df = pair_df[pair_df["fold"] == fold].reset_index(drop=True)
        va_ds = PairDataset(va_df, paths["train_images_dir"], transform=build_valid_tfms(pair_cfg.img_size), is_test=False)
        va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)

        va_fold_prob: List[np.ndarray] = []
        with torch.no_grad():
            for x, _ in tqdm(va_loader, desc=f"OOF {alias} fold{fold}", leave=False, disable=TQDM_OFF):
                x = x.to(device)
                if pair_cfg.use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                else:
                    x = x.contiguous()
                logits = model(x)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                if cfg.tta:
                    logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
                va_fold_prob.append(torch.sigmoid(logits).squeeze(1).detach().cpu().numpy())
        va_fold_prob_np = np.concatenate(va_fold_prob)
        oof_prob[pair_df["fold"].values == fold] = va_fold_prob_np

        fold_test_prob, _ = predict_proba_binary(
            model=model,
            loader=test_loader,
            device=device,
            tta=cfg.tta,
            use_channels_last=pair_cfg.use_channels_last,
        )
        test_prob_folds.append(fold_test_prob)

        del model, va_loader, va_ds
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    test_prob = np.mean(np.stack(test_prob_folds, axis=0), axis=0)

    pair_dir = Path(cfg.out_dir) / alias
    pair_dir.mkdir(parents=True, exist_ok=True)
    np.save(pair_dir / "oof_prob.npy", oof_prob)
    np.save(pair_dir / "oof_target.npy", oof_target)
    np.save(pair_dir / "test_prob.npy", test_prob)
    pair_df[["image_id", "label", "target", "fold"]].to_csv(pair_dir / "pair_train_split.csv", index=False)

    m05 = binary_metrics(oof_target, oof_prob, thr=0.5)
    bt = best_threshold(oof_target, oof_prob)
    mb = binary_metrics(oof_target, oof_prob, thr=bt["thr"])
    metrics = {
        "acc@0.5": m05["acc"],
        "f1@0.5": m05["f1"],
        "roc_auc": m05["roc_auc"],
        "best_thr": bt["thr"],
        "acc@best_thr": mb["acc"],
        "f1@best_thr": mb["f1"],
    }
    with (pair_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"OOF [{alias}] acc@0.5={metrics['acc@0.5']:.5f} f1@0.5={metrics['f1@0.5']:.5f} "
        f"auc={metrics['roc_auc']:.5f} best_thr={metrics['best_thr']:.3f} "
        f"f1@best_thr={metrics['f1@best_thr']:.5f}",
        flush=True,
    )
    return metrics


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        clean_variant=args.clean_variant,
        n_folds=2 if args.smoke else args.folds,
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        out_dir=args.out_dir,
    )
    pair_tokens = parse_pair_tokens(args.pairs)
    pair_cfgs = [
        PairModelCfg(
            alias=pair_alias(p, n),
            pos_label=p,
            neg_label=n,
            timm_name=args.pair_model,
            img_size=args.pair_img_size,
            use_channels_last=args.channels_last,
            pretrained=not args.no_pretrained,
        )
        for p, n in pair_tokens
    ]
    if args.max_pairs > 0:
        pair_cfgs = pair_cfgs[: args.max_pairs]

    print("Config:")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("Pairs:", [asdict(p) for p in pair_cfgs])
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = choose_device(args.device)
    print("Using device:", device, flush=True)
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    paths = build_paths(Path(args.base), cfg.clean_variant)
    for k, v in paths.items():
        print(f"{k}: {v} exists={v.exists()}")

    train_df = load_train_df(paths)
    test_df = load_test_df(paths)
    label_to_name = (
        train_df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )
    print("Label mapping:", label_to_name)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, float]] = {}
    all_summary: List[Dict] = []
    for p in pair_cfgs:
        print(
            f"\n=== TRAIN PAIR EXPERT: {p.alias} ({label_to_name.get(p.pos_label, p.pos_label)} vs "
            f"{label_to_name.get(p.neg_label, p.neg_label)}) ===",
            flush=True,
        )
        pair_df = make_pair_df(train_df, p.pos_label, p.neg_label)
        pair_df = assign_pair_folds(pair_df, n_splits=cfg.n_folds, seed=cfg.fold_seed)
        pair_dir = out_dir / p.alias
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_df.to_csv(pair_dir / f"folds_{cfg.n_folds}f.csv", index=False)

        metrics = run_pair_cv(
            pair_cfg=p,
            pair_df=pair_df,
            test_df=test_df,
            cfg=cfg,
            paths=paths,
            device=device,
        )
        all_metrics[p.alias] = metrics
        all_summary.append(
            {
                "alias": p.alias,
                "pos_label": p.pos_label,
                "neg_label": p.neg_label,
                "pos_name": label_to_name.get(p.pos_label, str(p.pos_label)),
                "neg_name": label_to_name.get(p.neg_label, str(p.neg_label)),
                "num_samples": int(len(pair_df)),
                "num_pos": int((pair_df["target"] == 1).sum()),
                "num_neg": int((pair_df["target"] == 0).sum()),
                "metrics": metrics,
            }
        )

    with (out_dir / "pair_experts_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    with (out_dir / "pair_experts_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print("\n=== DONE PAIR EXPERTS ===")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
