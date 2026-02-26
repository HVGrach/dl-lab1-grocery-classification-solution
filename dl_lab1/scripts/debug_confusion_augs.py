#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
DL_LAB1_DIR = THIS_DIR.parent
if str(DL_LAB1_DIR) not in sys.path:
    sys.path.insert(0, str(DL_LAB1_DIR))

from train_top1_mps import build_train_tfms, build_valid_tfms, load_train_df, build_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    p.add_argument("--clean-variant", type=str, default="strict")
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/aug_debug",
    )
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--samples-per-pair", type=int, default=6)
    p.add_argument("--num-train-augs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def denorm_name_map(df: pd.DataFrame) -> Dict[int, str]:
    return df.groupby("label")["class_name"].agg(lambda s: s.mode().iat[0]).to_dict()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def load_base_oof_logits(outputs_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    weights = json.loads((outputs_dir / "ensemble_weights.json").read_text(encoding="utf-8"))
    aliases = ["convnext_small", "effnetv2_s", "resnet50"]
    oofs = [np.load(outputs_dir / a / "oof_logits.npy") for a in aliases]
    y = np.load(outputs_dir / "convnext_small" / "oof_targets.npy")
    blend = np.zeros_like(oofs[0], dtype=np.float64)
    for a, o in zip(aliases, oofs):
        blend += float(weights[a]) * o
    return blend, y


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def hue_stats_rgb(img_rgb: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    # Ignore gray/dark pixels so background affects less.
    mask = (s > 40) & (v > 40)
    if mask.sum() == 0:
        return {"h_mean": float(h.mean()), "h_std": float(h.std()), "sat_mean": float(s.mean()), "pix": 0}
    hh = h[mask]
    ss = s[mask]
    return {"h_mean": float(hh.mean()), "h_std": float(hh.std()), "sat_mean": float(ss.mean()), "pix": int(mask.sum())}


def as_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="RGB")


def draw_text(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 42), (245, 245, 245))
    canvas.paste(img, (0, 42))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    d.text((6, 6), text, fill=(20, 20, 20), font=font)
    return canvas


def stack_grid(rows: List[List[Image.Image]]) -> Image.Image:
    if not rows or not rows[0]:
        raise ValueError("empty grid")
    w = max(im.width for r in rows for im in r)
    h = max(im.height for r in rows for im in r)
    cols = max(len(r) for r in rows)
    out = Image.new("RGB", (w * cols, h * len(rows)), (255, 255, 255))
    for ri, row in enumerate(rows):
        for ci, im in enumerate(row):
            out.paste(im, (ci * w, ri * h))
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    root = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1")
    outputs_dir = root / "outputs_top1_mps"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = build_paths(Path(args.base), args.clean_variant)
    df = load_train_df(paths)
    folds_df = pd.read_csv(outputs_dir / "folds_strict_5f.csv")
    if len(df) != len(folds_df):
        raise RuntimeError("train dataframe and folds csv size mismatch")
    df = folds_df.copy()
    label_to_name = denorm_name_map(df)

    logits, y_true = load_base_oof_logits(outputs_dir)
    pred = logits.argmax(1)
    prob = softmax(logits)
    conf = prob.max(1)

    df["y_true"] = y_true
    df["y_pred"] = pred
    df["pred_conf"] = conf

    # Pairs of interest from user.
    pair_specs = [
        {"alias": "orange_confusions", "true_label": 0, "pred_labels": [7, 9]},
        {"alias": "greenapple_confusions", "true_label": 13, "pred_labels": [14, 3]},
    ]

    full_train_tfms = build_train_tfms(args.img_size)
    full_val_tfms = build_valid_tfms(args.img_size)
    train_vis_tfms = A.Compose(full_train_tfms.transforms[:-2])
    val_vis_tfms = A.Compose(full_val_tfms.transforms[:-2])

    summary = {
        "pairs": [],
        "train_color_aug_params": {
            "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
            "HueSaturationValue": {"hue_shift_limit": 12, "sat_shift_limit": 20, "val_shift_limit": 20},
            "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2},
            "color_block_probability": 0.6,
        },
    }

    for spec in pair_specs:
        part = df[(df["y_true"] == spec["true_label"]) & (df["y_pred"].isin(spec["pred_labels"]))].copy()
        part = part.sort_values("pred_conf", ascending=False).head(args.samples_per_pair)
        rows: List[List[Image.Image]] = []
        sample_logs = []
        for _, r in part.iterrows():
            rel = r["image_id"]
            true_lbl = int(r["y_true"])
            pred_lbl = int(r["y_pred"])
            true_name = label_to_name[true_lbl]
            pred_name = label_to_name[pred_lbl]
            p = float(r["pred_conf"])

            img_raw = load_rgb(paths["train_images_dir"] / rel)
            img_val = val_vis_tfms(image=img_raw)["image"]
            img_val = np.clip(img_val, 0, 255).astype(np.uint8)

            row_imgs: List[Image.Image] = []
            raw_stats = hue_stats_rgb(img_raw)
            val_stats = hue_stats_rgb(img_val)

            row_imgs.append(
                draw_text(
                    as_pil(img_raw),
                    f"RAW | true={true_name} pred={pred_name} p={p:.3f}\n{rel}",
                )
            )
            row_imgs.append(
                draw_text(
                    as_pil(img_val),
                    f"VAL_CROP | h_mean={val_stats['h_mean']:.1f} sat_mean={val_stats['sat_mean']:.1f}",
                )
            )

            aug_stats = []
            for ai in range(args.num_train_augs):
                img_aug = train_vis_tfms(image=img_raw)["image"]
                img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)
                st = hue_stats_rgb(img_aug)
                aug_stats.append(st)
                row_imgs.append(
                    draw_text(
                        as_pil(img_aug),
                        f"TRAIN_AUG_{ai+1} | h_mean={st['h_mean']:.1f} sat_mean={st['sat_mean']:.1f}",
                    )
                )

            rows.append(row_imgs)
            sample_logs.append(
                {
                    "image_id": rel,
                    "true_label": true_lbl,
                    "true_name": true_name,
                    "pred_label": pred_lbl,
                    "pred_name": pred_name,
                    "pred_conf": p,
                    "raw_hue_stats": raw_stats,
                    "val_hue_stats": val_stats,
                    "aug_hue_stats": aug_stats,
                }
            )

        if not rows:
            continue
        sheet = stack_grid(rows)
        out_sheet = out_dir / f"{spec['alias']}_raw_vs_aug_sheet.png"
        out_json = out_dir / f"{spec['alias']}_raw_vs_aug_meta.json"
        sheet.save(out_sheet)
        out_json.write_text(json.dumps(sample_logs, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["pairs"].append(
            {
                "alias": spec["alias"],
                "true_label": spec["true_label"],
                "pred_labels": spec["pred_labels"],
                "num_samples": int(len(sample_logs)),
                "sheet_path": str(out_sheet),
                "meta_path": str(out_json),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
