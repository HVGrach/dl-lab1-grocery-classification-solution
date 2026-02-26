#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--outputs-dir", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps")
    p.add_argument("--folds-csv", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/folds_strict_5f.csv")
    p.add_argument(
        "--ckpt-fold1",
        type=str,
        default="/Users/fedorgracev/Downloads/convnext_tiny_fold1_best.pth",
    )
    p.add_argument(
        "--ckpt-fold4",
        type=str,
        default="/Users/fedorgracev/Downloads/convnext_tiny_fold4_best.pth",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--score-f1-weight", type=float, default=0.15)
    p.add_argument("--beta-grid", type=str, default="0.00,0.01,0.02,0.03,0.05,0.07,0.10,0.12,0.15,0.20,0.25,0.30")
    p.add_argument("--force-beta", type=float, default=None)
    p.add_argument("--out-submission-name", type=str, default="submission_base_plus_peer_convnext_tiny.csv")
    return p.parse_args()


def parse_grid(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def metric_bundle(y_true: np.ndarray, pred: np.ndarray, f1_w: float) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, pred))
    f1m = float(f1_score(y_true, pred, average="macro"))
    return {
        "acc": acc,
        "f1_macro": f1m,
        "score": acc + f1_w * f1m,
    }


def build_valid_tfms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class ImgDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform: A.Compose, is_test: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.array(Image.open(self.img_dir / row["image_id"]).convert("RGB"))
        x = self.transform(image=img)["image"]
        if self.is_test:
            return x, row["image_id"]
        return x, int(row["label"]), row["image_id"]


class PeerConvNextTiny(nn.Module):
    """Reconstructed from checkpoint keys: backbone + main/aux heads."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
        self.backbone.head = nn.Identity()
        self.main_classifier = nn.Linear(768, 15)
        self.auxiliary_classifier = nn.Linear(768, 43)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = feat.mean(dim=(-2, -1))
        return self.main_classifier(feat)


@torch.no_grad()
def infer_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_test: bool,
    tta: bool,
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_ids: List[str] = []
    for batch in tqdm(loader, leave=False):
        if is_test:
            x, ids = batch
        else:
            x, _, ids = batch
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        logits = model(x)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
        all_logits.append(logits.detach().cpu())
        all_ids.extend(ids)
    return torch.cat(all_logits).numpy(), all_ids


def load_base_ensemble_logits(outputs_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    weights = json.loads((outputs_dir / "ensemble_weights.json").read_text(encoding="utf-8"))
    aliases = list(weights.keys())
    oof_list = [np.load(outputs_dir / a / "oof_logits.npy") for a in aliases]
    test_list = [np.load(outputs_dir / a / "test_logits.npy") for a in aliases]

    blend_oof = np.zeros_like(oof_list[0], dtype=np.float64)
    blend_test = np.zeros_like(test_list[0], dtype=np.float64)
    for a, oof, tst in zip(aliases, oof_list, test_list):
        w = float(weights[a])
        blend_oof += w * oof
        blend_test += w * tst
    return blend_oof, blend_test


def main() -> None:
    args = parse_args()
    beta_grid = parse_grid(args.beta_grid)

    base = Path(args.base)
    outputs_dir = Path(args.outputs_dir)
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_csv = base / "cleaning" / "train_clean_strict.csv"
    test_csv = base / "test.csv"
    sample_submission = base / "sample_submission.csv"
    train_img_dir = base / "train" / "train"
    test_img_dir = base / "test_images" / "test_images"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    folds_df = pd.read_csv(args.folds_csv)[["image_id", "fold"]]
    train_df = train_df.merge(folds_df, on="image_id", how="left", validate="one_to_one")
    if train_df["fold"].isna().any():
        raise RuntimeError("Some train rows have no fold assignment.")

    y_true = train_df["label"].astype(int).to_numpy()
    tfm = build_valid_tfms(args.img_size)
    train_ds = ImgDataset(train_df, train_img_dir, tfm, is_test=False)
    test_ds = ImgDataset(test_df, test_img_dir, tfm, is_test=True)
    dl_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
    }
    train_loader = DataLoader(train_ds, **dl_kwargs)
    test_loader = DataLoader(test_ds, **dl_kwargs)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device, flush=True)

    ckpts = {
        "fold1": Path(args.ckpt_fold1),
        "fold4": Path(args.ckpt_fold4),
    }
    model_logits_train: Dict[str, np.ndarray] = {}
    model_logits_test: Dict[str, np.ndarray] = {}
    model_metrics: Dict[str, Dict] = {}

    for alias, ckpt_path in ckpts.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print(f"\n[peer] infer {alias} from {ckpt_path}", flush=True)
        model = PeerConvNextTiny().to(device)
        model = model.to(memory_format=torch.channels_last)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)

        tr_logits, tr_ids = infer_logits(model, train_loader, device, is_test=False, tta=args.tta)
        te_logits, te_ids = infer_logits(model, test_loader, device, is_test=True, tta=args.tta)

        if tr_ids != train_df["image_id"].tolist():
            raise RuntimeError(f"Train image order mismatch for {alias}.")
        if te_ids != test_df["image_id"].tolist():
            raise RuntimeError(f"Test image order mismatch for {alias}.")

        pred = tr_logits.argmax(1)
        overall = metric_bundle(y_true, pred, args.score_f1_weight)
        per_fold = {}
        for f in sorted(train_df["fold"].unique().tolist()):
            m = train_df["fold"].to_numpy() == f
            per_fold[str(int(f))] = metric_bundle(y_true[m], pred[m], args.score_f1_weight)

        model_metrics[alias] = {
            "overall_train_eval": overall,
            "per_fold_train_eval": per_fold,
            "checkpoint_path": str(ckpt_path),
        }
        model_logits_train[alias] = tr_logits.astype(np.float32)
        model_logits_test[alias] = te_logits.astype(np.float32)
        print(f"[peer] {alias} overall acc={overall['acc']:.6f} f1={overall['f1_macro']:.6f}", flush=True)

    # Construct mapped OOF-like predictions:
    # fold1 rows from fold1 model, fold4 rows from fold4 model, others from average.
    peer_logits_avg_train = 0.5 * (model_logits_train["fold1"] + model_logits_train["fold4"])
    peer_logits_avg_test = 0.5 * (model_logits_test["fold1"] + model_logits_test["fold4"])
    peer_logits_mapped = peer_logits_avg_train.copy()
    mask_f1 = train_df["fold"].to_numpy() == 1
    mask_f4 = train_df["fold"].to_numpy() == 4
    peer_logits_mapped[mask_f1] = model_logits_train["fold1"][mask_f1]
    peer_logits_mapped[mask_f4] = model_logits_train["fold4"][mask_f4]
    mask_unbiased = mask_f1 | mask_f4

    peer_mapped_metrics = metric_bundle(y_true[mask_unbiased], peer_logits_mapped[mask_unbiased].argmax(1), args.score_f1_weight)
    print(
        f"[peer] mapped(oof-ish folds 1&4) acc={peer_mapped_metrics['acc']:.6f} f1={peer_mapped_metrics['f1_macro']:.6f}",
        flush=True,
    )

    # Blend with current best public base ensemble (submission_ensemble_oof_optimized lineage).
    base_oof_logits, base_test_logits = load_base_ensemble_logits(outputs_dir)
    base_prob = softmax(base_oof_logits)
    base_pred = base_prob.argmax(1)
    base_metrics_mask = metric_bundle(y_true[mask_unbiased], base_pred[mask_unbiased], args.score_f1_weight)
    print(
        f"[blend] base on folds1&4 acc={base_metrics_mask['acc']:.6f} f1={base_metrics_mask['f1_macro']:.6f}",
        flush=True,
    )

    peer_prob_mapped = softmax(peer_logits_mapped)
    peer_prob_test = softmax(peer_logits_avg_test)
    base_prob_test = softmax(base_test_logits)

    if args.force_beta is not None:
        beta = float(np.clip(args.force_beta, 0.0, 1.0))
        p = (1.0 - beta) * base_prob + beta * peer_prob_mapped
        pred = p.argmax(1)
        m = metric_bundle(y_true[mask_unbiased], pred[mask_unbiased], args.score_f1_weight)
        best = {"beta": beta, **m}
    else:
        best = {
            "beta": 0.0,
            "score": base_metrics_mask["score"],
            "acc": base_metrics_mask["acc"],
            "f1_macro": base_metrics_mask["f1_macro"],
        }
        for beta in beta_grid:
            p = (1.0 - beta) * base_prob + beta * peer_prob_mapped
            pred = p.argmax(1)
            m = metric_bundle(y_true[mask_unbiased], pred[mask_unbiased], args.score_f1_weight)
            if m["score"] > best["score"] + 1e-12:
                best = {"beta": float(beta), **m}

    print(
        f"[blend] best beta={best['beta']:.3f} on folds1&4 acc={best['acc']:.6f} f1={best['f1_macro']:.6f}",
        flush=True,
    )

    final_test_prob = (1.0 - best["beta"]) * base_prob_test + best["beta"] * peer_prob_test
    final_test_pred = final_test_prob.argmax(1)

    sub = pd.read_csv(sample_submission)
    sub["label"] = final_test_pred
    sub_path = outputs_dir / args.out_submission_name
    sub.to_csv(sub_path, index=False)

    # Peer-only submission as diagnostic.
    sub_peer = pd.read_csv(sample_submission)
    sub_peer["label"] = peer_prob_test.argmax(1)
    sub_peer_path = outputs_dir / "submission_peer_convnext_tiny_only.csv"
    sub_peer.to_csv(sub_peer_path, index=False)

    # Save artifacts.
    peer_dir = outputs_dir / "external_peer_convnext_tiny"
    peer_dir.mkdir(parents=True, exist_ok=True)
    np.save(peer_dir / "peer_fold1_train_logits.npy", model_logits_train["fold1"])
    np.save(peer_dir / "peer_fold4_train_logits.npy", model_logits_train["fold4"])
    np.save(peer_dir / "peer_avg_test_logits.npy", peer_logits_avg_test)
    np.save(peer_dir / "peer_mapped_train_logits.npy", peer_logits_mapped)

    summary = {
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "tta": bool(args.tta),
        "model_metrics": model_metrics,
        "peer_mapped_oofish_folds_1_4": peer_mapped_metrics,
        "base_metrics_on_folds_1_4": base_metrics_mask,
        "best_blend_on_folds_1_4": best,
        "mask_unbiased_rows": int(mask_unbiased.sum()),
        "submission_blend_path": str(sub_path),
        "submission_peer_only_path": str(sub_peer_path),
    }
    summary_path = analysis_dir / "peer_convnext_tiny_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===", flush=True)
    print("summary:", summary_path, flush=True)
    print("submission_blend:", sub_path, flush=True)
    print("submission_peer_only:", sub_peer_path, flush=True)


if __name__ == "__main__":
    main()
