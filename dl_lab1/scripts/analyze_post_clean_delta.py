#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import importlib.util
import sys


def _load_train_top1_module():
    module_path = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/train_top1_mps.py")
    spec = importlib.util.spec_from_file_location("train_top1_mps", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_top1_mps"] = module
    spec.loader.exec_module(module)
    return module


_m = _load_train_top1_module()
FruitDataset = _m.FruitDataset
build_valid_tfms = _m.build_valid_tfms


IDX_TO_NAME = {
    0: "Апельсин",
    1: "Бананы",
    2: "Груши",
    3: "Кабачки",
    4: "Капуста",
    5: "Картофель",
    6: "Киви",
    7: "Лимон",
    8: "Лук",
    9: "Мандарины",
    10: "Морковь",
    11: "Огурцы",
    12: "Томаты",
    13: "Яблоки зелёные",
    14: "Яблоки красные",
}


def create_model_infer_only(model_cfg: Dict, num_classes: int):
    # For checkpoint comparison we do not need pretrained weights; avoid network lookups/retries.
    fallback = {
        "convnext_small.fb_in22k_ft_in1k": "convnext_small",
        "tf_efficientnetv2_s.in21k_ft_in1k": "tf_efficientnetv2_s",
        "resnet50.a1_in1k": "resnet50",
    }
    name = str(model_cfg["timm_name"])
    try:
        return timm.create_model(
            name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=float(model_cfg.get("drop_rate", 0.2)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.2)),
        )
    except Exception:
        return timm.create_model(
            fallback.get(name, name),
            pretrained=False,
            num_classes=num_classes,
            drop_rate=float(model_cfg.get("drop_rate", 0.2)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.2)),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare old vs new checkpoint on same post-clean fold.")
    p.add_argument(
        "--base",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1",
    )
    p.add_argument(
        "--folds-csv",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps/folds_used.csv",
    )
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument(
        "--old-ckpt",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_color_ablation_onefold_mps/no_color/convnext_small/fold_0/best_by_val_loss.pt",
    )
    p.add_argument(
        "--new-ckpt",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps/no_color/convnext_small/fold_0/best_by_val_loss.pt",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps/analysis_delta",
    )
    return p.parse_args()


def infer_checkpoint(
    ckpt_path: Path,
    model_cfg: Dict,
    df_val: pd.DataFrame,
    train_images_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = FruitDataset(
        df_val.reset_index(drop=True),
        train_images_dir,
        transform=build_valid_tfms(int(model_cfg["img_size"])),
        is_test=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    model = create_model_infer_only(model_cfg, num_classes=15).to(device)
    use_cl = bool(model_cfg.get("use_channels_last", True)) and device.type != "cpu"
    if use_cl:
        model = model.to(memory_format=torch.channels_last)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logits_all = []
    y_all = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            if use_cl:
                x = x.contiguous(memory_format=torch.channels_last)
            logits = model(x)
            logits_all.append(logits.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return logits, y


def class_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    labels = sorted(np.unique(y_true))
    for c in labels:
        mask = y_true == c
        support = int(mask.sum())
        correct = int((y_pred[mask] == c).sum())
        recall = float(correct / support) if support else 0.0
        pred_mask = y_pred == c
        pred_support = int(pred_mask.sum())
        precision = float((y_true[pred_mask] == c).sum() / pred_support) if pred_support else 0.0
        rows.append(
            {
                "label": int(c),
                "class_name": IDX_TO_NAME.get(int(c), str(c)),
                "support": support,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, topk: int = 20) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    labels = sorted(np.unique(y_true))
    rows = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if i == j:
                continue
            v = int(cm[i, j])
            if v > 0:
                rows.append(
                    {
                        "true_label": int(t),
                        "pred_label": int(p),
                        "true_name": IDX_TO_NAME.get(int(t), str(t)),
                        "pred_name": IDX_TO_NAME.get(int(p), str(p)),
                        "count": v,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["true_label", "pred_label", "true_name", "pred_name", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(topk).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    folds_df = pd.read_csv(args.folds_csv)
    val_df = folds_df[folds_df["fold"] == args.fold_idx].reset_index(drop=True)
    if val_df.empty:
        raise RuntimeError(f"No rows for fold={args.fold_idx} in {args.folds_csv}")

    model_cfg = {
        "alias": "convnext_small",
        "timm_name": "convnext_small.fb_in22k_ft_in1k",
        "img_size": 224,
        "drop_rate": 0.2,
        "drop_path_rate": 0.2,
        "use_channels_last": True,
    }
    train_images_dir = Path(args.base) / "train" / "train"

    old_logits, y_true_old = infer_checkpoint(
        Path(args.old_ckpt),
        model_cfg,
        val_df,
        train_images_dir,
        device,
        args.batch_size,
        args.num_workers,
    )
    new_logits, y_true_new = infer_checkpoint(
        Path(args.new_ckpt),
        model_cfg,
        val_df,
        train_images_dir,
        device,
        args.batch_size,
        args.num_workers,
    )
    if not np.array_equal(y_true_old, y_true_new):
        raise RuntimeError("Label mismatch between old/new eval.")
    y_true = y_true_old

    old_pred = old_logits.argmax(axis=1)
    new_pred = new_logits.argmax(axis=1)
    old_conf = torch.softmax(torch.tensor(old_logits), dim=1).max(dim=1).values.numpy()
    new_conf = torch.softmax(torch.tensor(new_logits), dim=1).max(dim=1).values.numpy()

    old_acc = float(accuracy_score(y_true, old_pred))
    old_f1 = float(f1_score(y_true, old_pred, average="macro"))
    new_acc = float(accuracy_score(y_true, new_pred))
    new_f1 = float(f1_score(y_true, new_pred, average="macro"))

    # Per-sample table
    cmp_df = val_df[["image_id", "label"]].copy()
    cmp_df["class_name"] = cmp_df["label"].map(IDX_TO_NAME)
    cmp_df["old_pred"] = old_pred
    cmp_df["old_pred_name"] = cmp_df["old_pred"].map(IDX_TO_NAME)
    cmp_df["old_conf"] = old_conf
    cmp_df["new_pred"] = new_pred
    cmp_df["new_pred_name"] = cmp_df["new_pred"].map(IDX_TO_NAME)
    cmp_df["new_conf"] = new_conf
    cmp_df["old_ok"] = (cmp_df["old_pred"] == cmp_df["label"]).astype(int)
    cmp_df["new_ok"] = (cmp_df["new_pred"] == cmp_df["label"]).astype(int)
    cmp_df["status"] = np.where(
        (cmp_df["old_ok"] == 0) & (cmp_df["new_ok"] == 1),
        "improved",
        np.where((cmp_df["old_ok"] == 1) & (cmp_df["new_ok"] == 0), "worsened", "unchanged"),
    )

    cmp_df.to_csv(out_dir / "per_sample_compare.csv", index=False)

    # Class-level delta
    old_class = class_table(y_true, old_pred).rename(columns={"precision": "old_precision", "recall": "old_recall"})
    new_class = class_table(y_true, new_pred).rename(columns={"precision": "new_precision", "recall": "new_recall"})
    cl = old_class.merge(new_class[["label", "new_precision", "new_recall"]], on="label", how="inner")
    cl["delta_precision"] = cl["new_precision"] - cl["old_precision"]
    cl["delta_recall"] = cl["new_recall"] - cl["old_recall"]
    cl.to_csv(out_dir / "class_delta.csv", index=False)

    # Confusions
    old_conf_df = top_confusions(y_true, old_pred, topk=30)
    new_conf_df = top_confusions(y_true, new_pred, topk=30)
    old_conf_df.to_csv(out_dir / "top_confusions_old_on_newfold.csv", index=False)
    new_conf_df.to_csv(out_dir / "top_confusions_new_on_newfold.csv", index=False)

    # Improved/worsened by true class
    iw = (
        cmp_df.groupby(["label", "class_name", "status"], as_index=False)
        .size()
        .pivot_table(index=["label", "class_name"], columns="status", values="size", fill_value=0)
        .reset_index()
    )
    for c in ("improved", "worsened", "unchanged"):
        if c not in iw.columns:
            iw[c] = 0
    iw["net"] = iw["improved"] - iw["worsened"]
    iw = iw.sort_values("net", ascending=False)
    iw.to_csv(out_dir / "improved_worsened_by_class.csv", index=False)

    summary = {
        "eval_protocol": "old and new checkpoints evaluated on same post-clean fold",
        "fold_idx": int(args.fold_idx),
        "n_val": int(len(y_true)),
        "old": {
            "acc": old_acc,
            "f1_macro": old_f1,
            "errors": int((old_pred != y_true).sum()),
            "mean_conf": float(old_conf.mean()),
            "ckpt": str(args.old_ckpt),
        },
        "new": {
            "acc": new_acc,
            "f1_macro": new_f1,
            "errors": int((new_pred != y_true).sum()),
            "mean_conf": float(new_conf.mean()),
            "ckpt": str(args.new_ckpt),
        },
        "delta_new_minus_old": {
            "acc": new_acc - old_acc,
            "f1_macro": new_f1 - old_f1,
            "errors": int((new_pred != y_true).sum() - (old_pred != y_true).sum()),
        },
        "status_counts": cmp_df["status"].value_counts().to_dict(),
        "artifacts": {
            "per_sample_compare": str(out_dir / "per_sample_compare.csv"),
            "class_delta": str(out_dir / "class_delta.csv"),
            "top_confusions_old_on_newfold": str(out_dir / "top_confusions_old_on_newfold.csv"),
            "top_confusions_new_on_newfold": str(out_dir / "top_confusions_new_on_newfold.csv"),
            "improved_worsened_by_class": str(out_dir / "improved_worsened_by_class.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
