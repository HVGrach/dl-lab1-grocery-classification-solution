#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced stacking over CV5 model-zoo OOF probs: logistic, catboost, attention, classwise-ridge."
    )
    p.add_argument(
        "--base",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1",
    )
    p.add_argument(
        "--zoo-root",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/meta_stack_cv5",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16, help="Base-model test inference batch size.")
    p.add_argument("--tta-mode", type=str, default="geo8", choices=["none", "flip", "geo4", "geo8"])
    p.add_argument("--tta-views", type=int, default=0)
    p.add_argument(
        "--methods",
        type=str,
        default="logreg,catboost,attn",
        help="Comma-separated: logreg,catboost,attn,cwr",
    )
    p.add_argument("--skip-test-infer", action="store_true", help="Only evaluate OOF and fit meta models, no test submissions.")
    p.add_argument("--reuse-test-cache", action="store_true", help="Reuse cached per-run test probs if present.")
    p.add_argument("--save-test-cache", action="store_true", help="Save per-run test probs cache.")
    p.add_argument(
        "--fold-aggregation",
        type=str,
        default="equal",
        choices=["equal", "oof_acc"],
        help="Average fold-level meta predictions equally or weight by per-fold OOF meta accuracy.",
    )

    # Logistic regression params
    p.add_argument("--logreg-c", type=float, default=2.0)
    p.add_argument("--logreg-max-iter", type=int, default=4000)

    # CatBoost params
    p.add_argument("--catboost-iterations", type=int, default=500)
    p.add_argument("--catboost-depth", type=int, default=6)
    p.add_argument("--catboost-lr", type=float, default=0.03)
    p.add_argument("--catboost-l2", type=float, default=3.0)

    # Class-wise ridge (notebook-inspired) params
    p.add_argument("--cwr-ridge-alpha", type=float, default=1.0)
    p.add_argument("--cwr-classwise-alpha", type=float, default=0.2)

    # Attention meta params
    p.add_argument("--attn-d-model", type=int, default=64)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-layers", type=int, default=2)
    p.add_argument("--attn-dropout", type=float, default=0.15)
    p.add_argument("--attn-epochs", type=int, default=18)
    p.add_argument("--attn-batch-size", type=int, default=128)
    p.add_argument("--attn-lr", type=float, default=2e-3)
    p.add_argument("--attn-weight-decay", type=float, default=1e-4)
    p.add_argument("--attn-label-smoothing", type=float, default=0.05)
    p.add_argument("--attn-patience", type=int, default=4)
    p.add_argument("--attn-final-epochs", type=int, default=12, help="Epochs when fitting final attention model on all OOF.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def clip_and_norm(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-12
    return probs.astype(np.float32)


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(probs.shape[1])))),
    }


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        image = np.array(Image.open(self.img_dir / image_id).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, image_id


def build_valid_tfms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def create_model_safe(name: str, num_classes: int, drop_rate: float = 0.2, drop_path_rate: float = 0.2) -> torch.nn.Module:
    fallback = {
        "convnext_small.fb_in22k_ft_in1k": "convnext_small",
        "tf_efficientnetv2_s.in21k_ft_in1k": "tf_efficientnetv2_s",
        "deit3_small_patch16_224.fb_in22k_ft_in1k": "deit3_small_patch16_224",
        "vit_base_patch16_224.augreg_in21k_ft_in1k": "vit_base_patch16_224",
    }
    try:
        return timm.create_model(
            name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
    except Exception:
        return timm.create_model(
            fallback.get(name, name),
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )


def build_tta_views(x: torch.Tensor, mode: str, max_views: int = 0) -> List[torch.Tensor]:
    views: List[torch.Tensor] = [x]
    if mode == "none":
        pass
    elif mode == "flip":
        views.append(torch.flip(x, dims=[3]))
    elif mode == "geo4":
        views.extend(
            [
                torch.flip(x, dims=[3]),
                torch.flip(x, dims=[2]),
                torch.rot90(x, k=2, dims=[2, 3]),
            ]
        )
    elif mode == "geo8":
        t = x.transpose(2, 3)
        views.extend(
            [
                torch.flip(x, dims=[3]),
                torch.flip(x, dims=[2]),
                torch.rot90(x, k=1, dims=[2, 3]),
                torch.rot90(x, k=2, dims=[2, 3]),
                torch.rot90(x, k=3, dims=[2, 3]),
                t,
                torch.flip(t, dims=[3]),
            ]
        )
    else:
        raise ValueError(f"Unknown tta mode: {mode}")
    if max_views and max_views > 0:
        views = views[:max_views]
    return views


def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_channels_last: bool,
    tta_mode: str,
    tta_views: int,
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    all_logits = []
    all_ids: List[str] = []
    with torch.no_grad():
        for x, image_ids in tqdm(loader, desc="infer", leave=False):
            x = x.to(device)
            x = x.contiguous(memory_format=torch.channels_last) if use_channels_last else x.contiguous()
            views = build_tta_views(x, mode=tta_mode, max_views=tta_views)
            logits_sum = None
            for xv in views:
                xv = xv.contiguous(memory_format=torch.channels_last) if use_channels_last else xv.contiguous()
                lv = model(xv)
                logits_sum = lv if logits_sum is None else (logits_sum + lv)
            logits = logits_sum / float(len(views))
            all_logits.append(logits.detach().cpu())
            all_ids.extend(list(image_ids))
    logits_np = torch.cat(all_logits).numpy()
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy().astype(np.float32)
    return probs, all_ids


@dataclass
class FoldBundle:
    fold_idx: int
    model_names: List[str]
    image_ids: List[str]
    y_true: np.ndarray
    probs_stack: np.ndarray  # [N, M, C]
    run_dirs: List[Path]
    configs: List[Dict]
    checkpoints: List[Path]
    final_selected: List[str]


def load_cv5_bundles(zoo_root: Path) -> Tuple[List[FoldBundle], int, int]:
    rr = pd.read_csv(zoo_root / "run_ranking.csv")
    rr = rr[rr["status"] == "ok"].copy()
    rr["fold_idx"] = rr["fold_idx"].astype(int)
    if rr.empty:
        raise RuntimeError("No successful runs in run_ranking.csv")

    bundles: List[FoldBundle] = []
    num_classes = None
    num_models = None
    for fold_idx in sorted(rr["fold_idx"].unique().tolist()):
        fr = rr[rr["fold_idx"] == fold_idx].sort_values("name")
        model_names = fr["name"].astype(str).tolist()
        run_dirs = [Path(p) for p in fr["run_dir"].astype(str).tolist()]
        probs_list = []
        y_ref = None
        ids_ref = None
        cfgs = []
        ckpts = []
        final_names = []
        for run_dir in run_dirs:
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            probs = np.load(run_dir / "val_probs.npy").astype(np.float32)
            y = np.load(run_dir / "val_labels.npy").astype(np.int64)
            pred_df = pd.read_csv(run_dir / "val_predictions.csv")
            ids = pred_df["image_id"].astype(str).tolist()
            if y_ref is None:
                y_ref = y
                ids_ref = ids
            else:
                if not np.array_equal(y_ref, y):
                    raise RuntimeError(f"val_labels mismatch in fold {fold_idx}: {run_dir}")
                if ids_ref != ids:
                    raise RuntimeError(f"val image_id order mismatch in fold {fold_idx}: {run_dir}")

            final_selected = str(summ.get("final_model_selected", "best_checkpoint"))
            if final_selected == "swa_model" and (run_dir / "swa_model.pt").exists():
                ckpt = run_dir / "swa_model.pt"
            else:
                ckpt = run_dir / "best_by_val_loss.pt"

            probs_list.append(probs)
            cfgs.append(cfg)
            ckpts.append(ckpt)
            final_names.append(final_selected)

        probs_stack = np.stack(probs_list, axis=1)  # [N,M,C]
        num_classes = probs_stack.shape[2]
        num_models = probs_stack.shape[1]
        bundles.append(
            FoldBundle(
                fold_idx=int(fold_idx),
                model_names=model_names,
                image_ids=ids_ref or [],
                y_true=y_ref if y_ref is not None else np.empty(0, dtype=np.int64),
                probs_stack=probs_stack,
                run_dirs=run_dirs,
                configs=cfgs,
                checkpoints=ckpts,
                final_selected=final_names,
            )
        )
    if num_classes is None or num_models is None:
        raise RuntimeError("No folds loaded")
    # Validate consistent model order across folds.
    ref_order = bundles[0].model_names
    for b in bundles[1:]:
        if b.model_names != ref_order:
            raise RuntimeError(f"Model order mismatch across folds: {b.fold_idx}")
    return bundles, int(num_models), int(num_classes)


def make_features_from_probs_stack(probs_stack: np.ndarray) -> np.ndarray:
    # probs_stack: [N, M, C]
    n, m, c = probs_stack.shape
    flat = probs_stack.reshape(n, m * c)
    maxp = probs_stack.max(axis=2)  # [N,M]
    p_sorted = np.sort(probs_stack, axis=2)
    margin = p_sorted[:, :, -1] - p_sorted[:, :, -2]  # [N,M]
    entropy = -(probs_stack * np.log(np.clip(probs_stack, 1e-12, 1.0))).sum(axis=2)  # [N,M]
    mean_probs = probs_stack.mean(axis=1)  # [N,C]
    feat = np.concatenate([flat, maxp, margin, entropy, mean_probs], axis=1)
    return feat.astype(np.float32)


def fit_classwise_ridge_from_stack(
    probs_stack: np.ndarray, y: np.ndarray, classwise_alpha: float, ridge_alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    # probs_stack: [N, M, C]
    n, m, c = probs_stack.shape
    y_fit = np.asarray(y, dtype=np.int64)
    class_freq = np.bincount(y_fit, minlength=c).astype(np.float64)
    class_freq = np.maximum(class_freq, 1.0)
    class_freq /= class_freq.sum()

    # Reorder to [N, C, M] so each class has an M-dimensional feature vector across models.
    x = np.transpose(probs_stack.astype(np.float64), (0, 2, 1))
    W = np.zeros((c, m), dtype=np.float64)
    for cls in range(c):
        Xc = x[:, cls, :]
        yc = (y_fit == cls).astype(np.float64)
        Xc_adj = Xc / (class_freq[cls] ** classwise_alpha)
        reg = Ridge(alpha=ridge_alpha, fit_intercept=False)
        reg.fit(Xc_adj, yc)
        w = np.clip(np.asarray(reg.coef_, dtype=np.float64), 0.0, None)
        s = float(w.sum())
        if s <= 0 or (not np.all(np.isfinite(w))):
            W[cls] = np.ones(m, dtype=np.float64) / float(m)
        else:
            W[cls] = w / s
    return W.astype(np.float32), class_freq.astype(np.float32)


def predict_classwise_ridge_from_stack(
    probs_stack: np.ndarray, W: np.ndarray, class_freq: np.ndarray, classwise_alpha: float
) -> np.ndarray:
    # probs_stack: [N, M, C] -> [N, C, M]
    x = np.transpose(probs_stack.astype(np.float64), (0, 2, 1))
    x_adj = x / (np.asarray(class_freq, dtype=np.float64)[None, :, None] ** float(classwise_alpha))
    blended = np.einsum("ncm,cm->nc", x_adj, np.asarray(W, dtype=np.float64))
    return clip_and_norm(blended)


def build_oof_meta_dataset(bundles: List[FoldBundle]) -> Dict[str, np.ndarray]:
    x_stack = []
    x_flat = []
    y = []
    fold_ids = []
    image_ids = []
    for b in bundles:
        x_stack.append(b.probs_stack.astype(np.float32))
        x_flat.append(make_features_from_probs_stack(b.probs_stack))
        y.append(b.y_true.astype(np.int64))
        fold_ids.append(np.full(len(b.y_true), b.fold_idx, dtype=np.int64))
        image_ids.extend(b.image_ids)
    X_stack = np.concatenate(x_stack, axis=0)
    X_flat = np.concatenate(x_flat, axis=0)
    y_all = np.concatenate(y, axis=0)
    f_all = np.concatenate(fold_ids, axis=0)
    return {
        "X_stack": X_stack,
        "X_flat": X_flat,
        "y": y_all,
        "fold_ids": f_all,
        "image_ids": np.array(image_ids, dtype=object),
    }


def build_test_probs_per_fold(
    bundles: List[FoldBundle],
    base: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_mode: str,
    tta_views: int,
    reuse_cache: bool,
    save_cache: bool,
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    test_df = pd.read_csv(base / "test.csv")
    test_img_dir = base / "test_images" / "test_images"
    ids_ref: List[str] | None = None
    loader_cache: Dict[int, DataLoader] = {}

    def get_loader(img_size: int) -> DataLoader:
        if img_size not in loader_cache:
            tfm = build_valid_tfms(img_size)
            ds = TestDataset(test_df, test_img_dir, tfm)
            loader_cache[img_size] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
        return loader_cache[img_size]

    out: Dict[int, np.ndarray] = {}
    cache_tag = f"test_probs_tta_{tta_mode}{'' if tta_views <= 0 else f'_v{tta_views}'}"

    for b in bundles:
        fold_probs_list = []
        print(f"\n=== TEST INFER fold={b.fold_idx} ({len(b.model_names)} models) ===", flush=True)
        for name, run_dir, cfg, ckpt in zip(b.model_names, b.run_dirs, b.configs, b.checkpoints):
            cache_path = run_dir / f"{cache_tag}.npy"
            if reuse_cache and cache_path.exists():
                probs = np.load(cache_path).astype(np.float32)
                # reconstruct ids_ref from first on-demand inference only; cache assumes standard test order.
                print(f"[cache] {run_dir.name}/{cache_path.name}", flush=True)
            else:
                print(f"[infer] {b.fold_idx}:{name} <- {ckpt.name}", flush=True)
                model = create_model_safe(
                    name=cfg["model_name"],
                    num_classes=15,
                    drop_rate=0.2,
                    drop_path_rate=0.2,
                ).to(device)
                use_channels_last = bool(cfg.get("use_channels_last", False))
                if use_channels_last:
                    model = model.to(memory_format=torch.channels_last)
                state = torch.load(ckpt, map_location=device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)

                loader = get_loader(int(cfg["img_size"]))
                probs, ids = predict_probs(
                    model=model,
                    loader=loader,
                    device=device,
                    use_channels_last=use_channels_last,
                    tta_mode=tta_mode,
                    tta_views=tta_views,
                )
                if ids_ref is None:
                    ids_ref = ids
                elif ids_ref != ids:
                    raise RuntimeError("Test image order mismatch across model runs")
                if save_cache:
                    np.save(cache_path, probs.astype(np.float32))
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
            fold_probs_list.append(probs)
        out[b.fold_idx] = np.stack(fold_probs_list, axis=1)  # [N_test, M, C]
    if ids_ref is None:
        # If everything came from cache, test order = test.csv order.
        ids_ref = pd.read_csv(base / "test.csv")["image_id"].astype(str).tolist()
    return out, ids_ref


def fit_logreg_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    c: float,
    max_iter: int,
) -> np.ndarray:
    lr_kwargs = {
        "C": c,
        "solver": "lbfgs",
        "max_iter": max_iter,
        "random_state": seed,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        lr_kwargs["multi_class"] = "multinomial"
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_kwargs)),
        ]
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_eval).astype(np.float32)


def fit_logreg_full(X: np.ndarray, y: np.ndarray, seed: int, c: float, max_iter: int):
    lr_kwargs = {
        "C": c,
        "solver": "lbfgs",
        "max_iter": max_iter,
        "random_state": seed,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        lr_kwargs["multi_class"] = "multinomial"
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_kwargs)),
        ]
    )
    clf.fit(X, y)
    return clf


def fit_catboost_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    iterations: int,
    depth: int,
    lr: float,
    l2_leaf_reg: float,
):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval).astype(np.float32)


def fit_catboost_full(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    iterations: int,
    depth: int,
    lr: float,
    l2_leaf_reg: float,
):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    return model


class AttentionMetaEnsemble(nn.Module):
    def __init__(self, in_dim: int, num_models: int, num_classes: int, d_model: int, nhead: int, layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.model_embed = nn.Parameter(torch.randn(1, num_models, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, C]
        b, m, _ = x.shape
        z = self.proj(x) + self.model_embed[:, :m, :]
        cls = self.cls_token.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.encoder(z)
        return self.head(z[:, 0, :])


def _torch_predict_probs(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            out.append(probs.detach().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def train_attention_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    device: torch.device,
    seed: int,
    num_classes: int,
    d_model: int,
    nhead: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    patience: int,
) -> nn.Module:
    seed_everything(seed)
    model = AttentionMetaEnsemble(
        in_dim=X_train.shape[2],
        num_models=X_train.shape[1],
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        dropout=dropout,
    ).to(device)
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_state = None
    best_score = (-1.0, -1.0)  # acc, f1
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()

        if val_loader is not None:
            probs_val = []
            y_val_ref = []
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    probs_val.append(torch.softmax(logits, dim=1).cpu())
                    y_val_ref.append(yb)
            pv = torch.cat(probs_val).numpy()
            yv = torch.cat(y_val_ref).numpy()
            m = evaluate_probs(yv, pv)
            score = (m["acc"], m["f1_macro"])
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(
                f"[attn] e{epoch:02d}/{epochs} train_loss={np.mean(losses):.4f} "
                f"val_acc={m['acc']:.5f} val_f1={m['f1_macro']:.5f}",
                flush=True,
            )
            if patience > 0 and no_improve >= patience:
                break
        else:
            print(f"[attn] e{epoch:02d}/{epochs} train_loss={np.mean(losses):.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def meta_cv_eval_logreg(X_flat: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float], Dict[int, Dict[str, float]]]:
    n_classes = int(np.max(y) + 1)
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold = {}
    for fid in sorted(np.unique(fold_ids).tolist()):
        tr = fold_ids != fid
        va = fold_ids == fid
        prob = fit_logreg_and_predict(
            X_train=X_flat[tr],
            y_train=y[tr],
            X_eval=X_flat[va],
            seed=args.seed + int(fid),
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        oof_prob[va] = prob
        per_fold[int(fid)] = evaluate_probs(y[va], prob)
        print(f"[logreg] fold={fid} acc={per_fold[int(fid)]['acc']:.5f} f1={per_fold[int(fid)]['f1_macro']:.5f}", flush=True)
    return oof_prob, evaluate_probs(y, oof_prob), per_fold


def meta_cv_eval_catboost(X_flat: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float], Dict[int, Dict[str, float]]]:
    n_classes = int(np.max(y) + 1)
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold = {}
    for fid in sorted(np.unique(fold_ids).tolist()):
        tr = fold_ids != fid
        va = fold_ids == fid
        prob = fit_catboost_and_predict(
            X_train=X_flat[tr],
            y_train=y[tr],
            X_eval=X_flat[va],
            seed=args.seed + int(fid),
            iterations=args.catboost_iterations,
            depth=args.catboost_depth,
            lr=args.catboost_lr,
            l2_leaf_reg=args.catboost_l2,
        )
        oof_prob[va] = prob
        per_fold[int(fid)] = evaluate_probs(y[va], prob)
        print(f"[catboost] fold={fid} acc={per_fold[int(fid)]['acc']:.5f} f1={per_fold[int(fid)]['f1_macro']:.5f}", flush=True)
    return oof_prob, evaluate_probs(y, oof_prob), per_fold


def meta_cv_eval_attn(X_stack: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, args: argparse.Namespace, device: torch.device) -> Tuple[np.ndarray, Dict[str, float], Dict[int, Dict[str, float]]]:
    n_classes = int(np.max(y) + 1)
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold = {}
    for fid in sorted(np.unique(fold_ids).tolist()):
        tr = fold_ids != fid
        va = fold_ids == fid
        model = train_attention_model(
            X_train=X_stack[tr],
            y_train=y[tr],
            X_val=X_stack[va],
            y_val=y[va],
            device=device,
            seed=args.seed + int(fid),
            num_classes=n_classes,
            d_model=args.attn_d_model,
            nhead=args.attn_heads,
            layers=args.attn_layers,
            dropout=args.attn_dropout,
            epochs=args.attn_epochs,
            batch_size=args.attn_batch_size,
            lr=args.attn_lr,
            weight_decay=args.attn_weight_decay,
            label_smoothing=args.attn_label_smoothing,
            patience=args.attn_patience,
        )
        prob = _torch_predict_probs(model, X_stack[va], device=device, batch_size=args.attn_batch_size)
        oof_prob[va] = prob
        per_fold[int(fid)] = evaluate_probs(y[va], prob)
        print(f"[attn] fold={fid} acc={per_fold[int(fid)]['acc']:.5f} f1={per_fold[int(fid)]['f1_macro']:.5f}", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    return oof_prob, evaluate_probs(y, oof_prob), per_fold


def meta_cv_eval_cwr(
    X_stack: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, args: argparse.Namespace
) -> Tuple[np.ndarray, Dict[str, float], Dict[int, Dict[str, float]]]:
    n_classes = int(np.max(y) + 1)
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold = {}
    for fid in sorted(np.unique(fold_ids).tolist()):
        tr = fold_ids != fid
        va = fold_ids == fid
        W, class_freq = fit_classwise_ridge_from_stack(
            X_stack[tr],
            y[tr],
            classwise_alpha=args.cwr_classwise_alpha,
            ridge_alpha=args.cwr_ridge_alpha,
        )
        prob = predict_classwise_ridge_from_stack(
            X_stack[va],
            W=W,
            class_freq=class_freq,
            classwise_alpha=args.cwr_classwise_alpha,
        )
        oof_prob[va] = prob
        per_fold[int(fid)] = evaluate_probs(y[va], prob)
        print(f"[cwr] fold={fid} acc={per_fold[int(fid)]['acc']:.5f} f1={per_fold[int(fid)]['f1_macro']:.5f}", flush=True)
    return oof_prob, evaluate_probs(y, oof_prob), per_fold


def fit_final_attention(X_stack: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device) -> nn.Module:
    # Fit on all OOF without validation (shorter fixed schedule to avoid extra split leakage).
    return train_attention_model(
        X_train=X_stack,
        y_train=y,
        X_val=None,
        y_val=None,
        device=device,
        seed=args.seed,
        num_classes=int(np.max(y) + 1),
        d_model=args.attn_d_model,
        nhead=args.attn_heads,
        layers=args.attn_layers,
        dropout=args.attn_dropout,
        epochs=args.attn_final_epochs,
        batch_size=args.attn_batch_size,
        lr=args.attn_lr,
        weight_decay=args.attn_weight_decay,
        label_smoothing=args.attn_label_smoothing,
        patience=0,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    seed_everything(args.seed)

    device = resolve_device(args.device)
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
    print("Device:", device, flush=True)

    base = Path(args.base)
    zoo_root = Path(args.zoo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods_raw = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    alias = {"classwise_ridge": "cwr"}
    methods = []
    for m in methods_raw:
        mm = alias.get(m, m)
        if mm in {"logreg", "catboost", "attn", "cwr"} and mm not in methods:
            methods.append(mm)
    if not methods:
        raise ValueError("No valid methods selected")

    bundles, num_models, num_classes = load_cv5_bundles(zoo_root)
    print(f"Loaded {len(bundles)} folds, {num_models} models/fold, classes={num_classes}", flush=True)
    print("Model order:", bundles[0].model_names, flush=True)

    oof = build_oof_meta_dataset(bundles)
    X_stack = oof["X_stack"]  # [N,M,C]
    X_flat = oof["X_flat"]
    y = oof["y"]
    fold_ids = oof["fold_ids"]

    # Save dataset manifest for reproducibility.
    dataset_manifest = {
        "num_samples": int(len(y)),
        "num_classes": int(num_classes),
        "num_models_per_fold": int(num_models),
        "feature_dim_flat": int(X_flat.shape[1]),
        "fold_counts": {str(int(fid)): int((fold_ids == fid).sum()) for fid in sorted(np.unique(fold_ids))},
        "model_order": bundles[0].model_names,
        "zoo_root": str(zoo_root),
    }
    (out_dir / "meta_dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    results = {}
    final_models = {}
    meta_oof_probs = {}

    # Method: Logistic Regression
    if "logreg" in methods:
        print("\n=== META: LOGREG ===", flush=True)
        oof_prob, metrics_all, per_fold = meta_cv_eval_logreg(X_flat, y, fold_ids, args)
        results["logreg"] = {"oof_metrics": metrics_all, "per_fold": per_fold}
        meta_oof_probs["logreg"] = oof_prob
        final_models["logreg"] = fit_logreg_full(X_flat, y, seed=args.seed, c=args.logreg_c, max_iter=args.logreg_max_iter)
        np.save(out_dir / "oof_meta_logreg_probs.npy", oof_prob.astype(np.float32))

    # Method: CatBoost (optional)
    if "catboost" in methods:
        print("\n=== META: CATBOOST ===", flush=True)
        try:
            oof_prob, metrics_all, per_fold = meta_cv_eval_catboost(X_flat, y, fold_ids, args)
            results["catboost"] = {"oof_metrics": metrics_all, "per_fold": per_fold}
            meta_oof_probs["catboost"] = oof_prob
            final_models["catboost"] = fit_catboost_full(
                X_flat,
                y,
                seed=args.seed,
                iterations=args.catboost_iterations,
                depth=args.catboost_depth,
                lr=args.catboost_lr,
                l2_leaf_reg=args.catboost_l2,
            )
            np.save(out_dir / "oof_meta_catboost_probs.npy", oof_prob.astype(np.float32))
        except ImportError as e:
            results["catboost"] = {"status": "skipped_import_error", "error": str(e)}
            print("[catboost] not installed -> skipped", flush=True)

    # Method: Attention-on-logits
    if "attn" in methods:
        print("\n=== META: ATTENTION (on logits/probs tokens) ===", flush=True)
        oof_prob, metrics_all, per_fold = meta_cv_eval_attn(X_stack, y, fold_ids, args, device=device)
        results["attn"] = {"oof_metrics": metrics_all, "per_fold": per_fold}
        meta_oof_probs["attn"] = oof_prob
        final_models["attn"] = fit_final_attention(X_stack, y, args=args, device=device)
        np.save(out_dir / "oof_meta_attn_probs.npy", oof_prob.astype(np.float32))
        torch.save(final_models["attn"].state_dict(), out_dir / "meta_attn_final_state.pt")

    # Method: Class-wise Ridge (notebook-inspired)
    if "cwr" in methods:
        print("\n=== META: CLASSWISE RIDGE ===", flush=True)
        oof_prob, metrics_all, per_fold = meta_cv_eval_cwr(X_stack, y, fold_ids, args)
        results["cwr"] = {"oof_metrics": metrics_all, "per_fold": per_fold}
        meta_oof_probs["cwr"] = oof_prob
        W_full, class_freq_full = fit_classwise_ridge_from_stack(
            X_stack,
            y,
            classwise_alpha=args.cwr_classwise_alpha,
            ridge_alpha=args.cwr_ridge_alpha,
        )
        final_models["cwr"] = {"W": W_full, "class_freq": class_freq_full}
        np.save(out_dir / "oof_meta_cwr_probs.npy", oof_prob.astype(np.float32))
        np.save(out_dir / "meta_cwr_W.npy", W_full.astype(np.float32))
        np.save(out_dir / "meta_cwr_class_freq.npy", class_freq_full.astype(np.float32))

    # Ranking of methods by OOF metrics
    ranking_rows = []
    for name, rec in results.items():
        if "oof_metrics" not in rec:
            continue
        row = {"method": name, **rec["oof_metrics"]}
        ranking_rows.append(row)
    if ranking_rows:
        ranking_df = pd.DataFrame(ranking_rows).sort_values(["acc", "f1_macro"], ascending=False)
        ranking_df.to_csv(out_dir / "meta_methods_oof_ranking.csv", index=False)
        print("\nOOF ranking:")
        print(ranking_df.to_string(index=False), flush=True)

    # Optional test inference and submissions
    if args.skip_test_infer:
        summary = {
            "status": "oof_only",
            "methods": methods,
            "results": results,
            "out_dir": str(out_dir),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[done] OOF-only mode", flush=True)
        return

    print("\n=== BUILD TEST META FEATURES (base models + TTA) ===", flush=True)
    test_probs_by_fold, test_ids = build_test_probs_per_fold(
        bundles=bundles,
        base=base,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tta_mode=args.tta_mode,
        tta_views=args.tta_views,
        reuse_cache=args.reuse_test_cache,
        save_cache=args.save_test_cache,
    )
    test_stack_by_fold = {fid: arr.astype(np.float32) for fid, arr in test_probs_by_fold.items()}  # [N,M,C]
    test_flat_by_fold = {fid: make_features_from_probs_stack(arr) for fid, arr in test_stack_by_fold.items()}

    sample_sub = pd.read_csv(base / "sample_submission.csv")
    if sample_sub["image_id"].astype(str).tolist() != test_ids:
        # Keep safety check visible but not fatal if only order differs in sample_sub? Here should match test.csv.
        print("[warn] sample_submission image_id order differs from inferred test_ids; using inferred order then mapping", flush=True)
        sub_template = pd.DataFrame({"image_id": test_ids})
    else:
        sub_template = sample_sub.copy()

    submissions_meta = {}

    for method_name, model in final_models.items():
        if method_name not in results or "oof_metrics" not in results[method_name]:
            continue
        print(f"\n=== TEST PRED: {method_name} ===", flush=True)
        fold_pred_probs = []
        fold_eval_weight_scores = []
        for b in bundles:
            fid = b.fold_idx
            if method_name == "logreg":
                prob = model.predict_proba(test_flat_by_fold[fid]).astype(np.float32)
            elif method_name == "catboost":
                prob = model.predict_proba(test_flat_by_fold[fid]).astype(np.float32)
            elif method_name == "attn":
                prob = _torch_predict_probs(model, test_stack_by_fold[fid], device=device, batch_size=args.attn_batch_size)
            elif method_name == "cwr":
                prob = predict_classwise_ridge_from_stack(
                    test_stack_by_fold[fid],
                    W=model["W"],
                    class_freq=model["class_freq"],
                    classwise_alpha=args.cwr_classwise_alpha,
                )
            else:
                raise ValueError(method_name)
            fold_pred_probs.append(prob.astype(np.float64))
            fold_eval_weight_scores.append(float(results[method_name]["per_fold"][fid]["acc"]))

        if args.fold_aggregation == "equal":
            fw = np.ones(len(fold_pred_probs), dtype=np.float64) / len(fold_pred_probs)
        else:
            fw = np.array(fold_eval_weight_scores, dtype=np.float64)
            fw = fw / (fw.sum() + 1e-12)

        blend = np.zeros_like(fold_pred_probs[0], dtype=np.float64)
        for w, p in zip(fw, fold_pred_probs):
            blend += w * p
        pred = blend.argmax(1).astype(int)

        sub = sub_template.copy()
        sub["label"] = pred
        out_csv = out_dir / f"submission_meta_{method_name}_{args.tta_mode}_{args.fold_aggregation}.csv"
        sub.to_csv(out_csv, index=False)
        if method_name == "attn":
            np.save(out_dir / f"submission_meta_{method_name}_{args.tta_mode}_{args.fold_aggregation}_probs.npy", blend.astype(np.float32))

        meta = {
            "method": method_name,
            "oof_metrics": results[method_name]["oof_metrics"],
            "per_fold_metrics": results[method_name]["per_fold"],
            "tta_mode": args.tta_mode,
            "tta_views": int(args.tta_views),
            "fold_aggregation": args.fold_aggregation,
            "fold_weights": [float(x) for x in fw],
            "zoo_root": str(zoo_root),
            "out_csv": str(out_csv),
            "label_distribution": {str(int(k)): int(v) for k, v in pd.Series(pred).value_counts().sort_index().items()},
        }
        (out_dir / f"submission_meta_{method_name}_{args.tta_mode}_{args.fold_aggregation}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        submissions_meta[method_name] = meta
        print(f"[saved] {out_csv}", flush=True)

    summary = {
        "status": "ok",
        "methods_requested": methods,
        "methods_finished": list(submissions_meta.keys()),
        "results": results,
        "submissions": {k: v["out_csv"] for k, v in submissions_meta.items()},
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== DONE META STACKING ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
