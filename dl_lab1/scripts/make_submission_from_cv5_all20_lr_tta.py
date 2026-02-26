#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build test submission from CV5 model zoo (all 20 models) with per-fold linear-regression weights + TTA."
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--tta-mode", type=str, default="geo8", choices=["none", "flip", "geo4", "geo8"])
    p.add_argument("--tta-views", type=int, default=0, help="Optional cap on TTA views (0 = all views for mode).")
    p.add_argument(
        "--fold-aggregation",
        type=str,
        default="equal",
        choices=["equal", "val_acc"],
        help="How to combine fold ensembles on test.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8.csv",
    )
    p.add_argument("--save-test-probs", action="store_true", help="Save blended test probs (.npy) next to csv.")
    return p.parse_args()


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
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


def create_model_safe(name: str, num_classes: int, drop_rate: float = 0.2, drop_path_rate: float = 0.2) -> torch.nn.Module:
    fallback = {
        "convnext_small.fb_in22k_ft_in1k": "convnext_small",
        "tf_efficientnetv2_s.in21k_ft_in1k": "tf_efficientnetv2_s",
        "resnet50.a1_in1k": "resnet50",
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
) -> Tuple[np.ndarray, List[str], int]:
    model.eval()
    all_logits = []
    all_ids: List[str] = []
    used_views = 1
    with torch.no_grad():
        for x, image_ids in tqdm(loader, desc="infer", leave=False):
            x = x.to(device)
            x = x.contiguous(memory_format=torch.channels_last) if use_channels_last else x.contiguous()
            views = build_tta_views(x, mode=tta_mode, max_views=tta_views)
            used_views = len(views)
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
    return probs, all_ids, int(used_views)


def fit_lr_weights(prob_list: List[np.ndarray], y_true: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    n_models = len(prob_list)
    prob_stack = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list], axis=-1)  # [N,C,M]
    n_samples, n_classes, _ = prob_stack.shape
    y_idx = np.asarray(y_true, dtype=np.int64)
    y_onehot = np.eye(n_classes, dtype=np.float64)[y_idx]
    X = prob_stack.reshape(n_samples * n_classes, n_models)
    y = y_onehot.reshape(n_samples * n_classes)

    try:
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y)
        w = np.asarray(reg.coef_, dtype=np.float64).reshape(-1)
    except TypeError:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        w = np.clip(np.asarray(reg.coef_, dtype=np.float64).reshape(-1), 0.0, None)

    w = np.clip(w, 0.0, None)
    if (not np.all(np.isfinite(w))) or float(w.sum()) <= 0:
        w = np.ones(n_models, dtype=np.float64) / n_models
    else:
        w /= float(w.sum())

    blend = np.tensordot(prob_stack, w, axes=([2], [0]))
    pred = blend.argmax(1)
    acc = float((pred == y_idx).mean())
    # Macro-F1 optional for metadata; compute without sklearn to avoid extra import here? sklearn already installed.
    from sklearn.metrics import f1_score  # local import keeps startup simple

    f1m = float(f1_score(y_idx, pred, average="macro"))
    return w, {"acc": acc, "f1_macro": f1m}


def select_checkpoint(run_dir: Path) -> Tuple[Path, str]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    final_selected = summary.get("final_model_selected", "best_checkpoint")
    if final_selected == "swa_model" and (run_dir / "swa_model.pt").exists():
        return run_dir / "swa_model.pt", final_selected
    return run_dir / "best_by_val_loss.pt", final_selected


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = resolve_device(args.device)
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
    print("Device:", device, flush=True)

    base = Path(args.base)
    zoo_root = Path(args.zoo_root)
    runs_root = zoo_root / "runs"
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ranking_path = zoo_root / "run_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing {ranking_path}")
    ranking = pd.read_csv(ranking_path)
    ranking = ranking[ranking["status"] == "ok"].copy()
    ranking["fold_idx"] = ranking["fold_idx"].astype(int)

    if ranking.empty:
        raise RuntimeError("No successful runs in run_ranking.csv")

    test_df = pd.read_csv(base / "test.csv")
    sample_sub = pd.read_csv(base / "sample_submission.csv")
    test_img_dir = base / "test_images" / "test_images"

    # Pre-create dataset/loader cache by image size.
    loader_cache: Dict[int, Tuple[TestDataset, DataLoader]] = {}

    def get_loader(img_size: int) -> DataLoader:
        if img_size not in loader_cache:
            tfm = build_valid_tfms(img_size)
            ds = TestDataset(test_df, test_img_dir, tfm)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
            )
            loader_cache[img_size] = (ds, loader)
        return loader_cache[img_size][1]

    fold_blends: List[np.ndarray] = []
    fold_meta: Dict[str, Dict] = {}
    ids_ref: List[str] | None = None
    fold_scores_for_agg: List[float] = []

    # Infer in deterministic order: fold -> model rows sorted by name
    for fold_idx in sorted(ranking["fold_idx"].unique().tolist()):
        fold_rows = ranking[ranking["fold_idx"] == fold_idx].sort_values("name").to_dict("records")
        print(f"\n=== FOLD {fold_idx}: {len(fold_rows)} models ===", flush=True)

        # Load OOF probs for LR weights.
        oof_probs: List[np.ndarray] = []
        y_ref = None
        oof_ids_ref = None
        model_names: List[str] = []
        run_dirs: List[Path] = []
        infer_cfgs: List[Dict] = []
        selected_ckpts: List[Tuple[Path, str]] = []

        for row in fold_rows:
            run_dir = Path(row["run_dir"])
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            val_probs = np.load(run_dir / "val_probs.npy")
            val_labels = np.load(run_dir / "val_labels.npy")
            val_pred_df = pd.read_csv(run_dir / "val_predictions.csv")
            oof_ids = val_pred_df["image_id"].astype(str).tolist()
            if y_ref is None:
                y_ref = val_labels
                oof_ids_ref = oof_ids
            else:
                if not np.array_equal(y_ref, val_labels):
                    raise RuntimeError(f"OOF labels mismatch in fold {fold_idx}: {run_dir}")
                if oof_ids_ref != oof_ids:
                    raise RuntimeError(f"OOF image order mismatch in fold {fold_idx}: {run_dir}")

            oof_probs.append(val_probs)
            model_names.append(str(row["name"]))
            run_dirs.append(run_dir)
            infer_cfgs.append(cfg)
            selected_ckpts.append(select_checkpoint(run_dir))

        assert y_ref is not None
        w_fold, fold_metric = fit_lr_weights(oof_probs, y_ref)
        print(f"[fold {fold_idx}] LR weights:", {n: round(float(w), 6) for n, w in zip(model_names, w_fold)}, flush=True)
        print(f"[fold {fold_idx}] OOF blend acc={fold_metric['acc']:.6f} f1={fold_metric['f1_macro']:.6f}", flush=True)

        # Test inference for each model in the fold
        probs_test_list: List[np.ndarray] = []
        fold_tta_views: Dict[str, int] = {}
        for name, run_dir, cfg, (ckpt_path, final_selected) in zip(model_names, run_dirs, infer_cfgs, selected_ckpts):
            print(f"Loading: fold={fold_idx} {name} from {ckpt_path.name} (selected={final_selected})", flush=True)
            model = create_model_safe(
                name=cfg["model_name"],
                num_classes=15,
                drop_rate=0.2,
                drop_path_rate=0.2,
            ).to(device)
            use_channels_last = bool(cfg.get("use_channels_last", False))
            if use_channels_last:
                model = model.to(memory_format=torch.channels_last)

            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)

            loader = get_loader(int(cfg["img_size"]))
            probs_test, ids, n_views = predict_probs(
                model=model,
                loader=loader,
                device=device,
                use_channels_last=use_channels_last,
                tta_mode=args.tta_mode,
                tta_views=args.tta_views,
            )
            probs_test_list.append(probs_test)
            fold_tta_views[name] = int(n_views)
            if ids_ref is None:
                ids_ref = ids
            elif ids_ref != ids:
                raise RuntimeError("Test image order mismatch across models")

            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        blend_fold = np.zeros_like(probs_test_list[0], dtype=np.float64)
        for w, p in zip(w_fold, probs_test_list):
            blend_fold += float(w) * p

        fold_blends.append(blend_fold)
        fold_scores_for_agg.append(float(fold_metric["acc"]))
        fold_meta[str(fold_idx)] = {
            "models": model_names,
            "weights_lr_mse": [float(x) for x in w_fold],
            "oof_blend_metrics": fold_metric,
            "tta_mode": args.tta_mode,
            "tta_views_limit": int(args.tta_views),
            "per_model_tta_views": fold_tta_views,
        }

    if not fold_blends:
        raise RuntimeError("No fold blends were created")

    if args.fold_aggregation == "equal":
        fold_weights = np.ones(len(fold_blends), dtype=np.float64) / len(fold_blends)
    else:
        fw = np.array(fold_scores_for_agg, dtype=np.float64)
        fold_weights = fw / (fw.sum() + 1e-12)

    final_blend = np.zeros_like(fold_blends[0], dtype=np.float64)
    for w, p in zip(fold_weights, fold_blends):
        final_blend += float(w) * p

    pred = final_blend.argmax(1)
    out = sample_sub.copy()
    out["label"] = pred
    out.to_csv(out_csv, index=False)
    if args.save_test_probs:
        np.save(out_csv.with_suffix(".probs.npy"), final_blend.astype(np.float32))

    meta = {
        "zoo_root": str(zoo_root),
        "run_ranking_csv": str(ranking_path),
        "num_models_total": int(len(ranking)),
        "num_folds": int(len(fold_blends)),
        "fold_aggregation": args.fold_aggregation,
        "fold_weights": [float(x) for x in fold_weights],
        "tta_mode": args.tta_mode,
        "tta_views_limit": int(args.tta_views),
        "device": str(device),
        "folds": fold_meta,
        "out_csv": str(out_csv),
        "label_distribution": {str(int(k)): int(v) for k, v in out["label"].value_counts().sort_index().items()},
    }
    (out_csv.parent / (out_csv.stem + "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== DONE ===", flush=True)
    print("Saved submission:", out_csv, flush=True)
    print("Fold weights:", {i: float(w) for i, w in enumerate(fold_weights)}, flush=True)
    print("Label distribution:", meta["label_distribution"], flush=True)


if __name__ == "__main__":
    main()
