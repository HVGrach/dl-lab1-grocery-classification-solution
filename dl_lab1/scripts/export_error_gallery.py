#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export error images with true/pred labels as PNG gallery pages.")
    p.add_argument(
        "--pred-csv",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_no_color_innov_mps/val_predictions.csv",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/train/train",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_no_color_innov_mps/error_gallery",
    )
    p.add_argument("--thumb-w", type=int, default=220)
    p.add_argument("--thumb-h", type=int, default=220)
    p.add_argument("--text-h", type=int, default=62)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--sort", type=str, default="pair_then_conf", choices=["pair_then_conf", "conf_desc"])
    return p.parse_args()


def pick_font(size: int = 13) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for fp in candidates:
        p = Path(fp)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def safe_open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def tile_image(im: Image.Image, tw: int, th: int, text_h: int, text: str, font: ImageFont.ImageFont) -> Image.Image:
    # Fit image preserving aspect ratio.
    im_fit = ImageOps.fit(im, (tw, th), method=Image.Resampling.BICUBIC)
    tile = Image.new("RGB", (tw, th + text_h), (245, 245, 245))
    tile.paste(im_fit, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.multiline_text((6, th + 4), text, fill=(10, 10, 10), font=font, spacing=2)
    return tile


def save_pages(tiles: List[Image.Image], cols: int, rows: int, out_dir: Path) -> List[Path]:
    page_size = cols * rows
    page_paths: List[Path] = []
    if not tiles:
        return page_paths

    tw = tiles[0].width
    th = tiles[0].height
    for i in range(0, len(tiles), page_size):
        chunk = tiles[i : i + page_size]
        page_h = rows * th
        page_w = cols * tw
        canvas = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        for j, t in enumerate(chunk):
            rr = j // cols
            cc = j % cols
            canvas.paste(t, (cc * tw, rr * th))
        page_idx = i // page_size + 1
        out_path = out_dir / f"errors_page_{page_idx:02d}.png"
        canvas.save(out_path)
        page_paths.append(out_path)
    return page_paths


def main() -> None:
    args = parse_args()
    pred_csv = Path(args.pred_csv)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font = pick_font(size=13)

    df = pd.read_csv(pred_csv)
    # Support both naming conventions:
    # 1) class_true/class_pred, label_true/label_pred
    # 2) true_name/pred_name, true_label/pred_label
    if "class_true" not in df.columns and "true_name" in df.columns:
        df["class_true"] = df["true_name"]
    if "class_pred" not in df.columns and "pred_name" in df.columns:
        df["class_pred"] = df["pred_name"]
    if "label_true" not in df.columns and "true_label" in df.columns:
        df["label_true"] = df["true_label"]
    if "label_pred" not in df.columns and "pred_label" in df.columns:
        df["label_pred"] = df["pred_label"]

    required_cols = {"is_error", "image_id", "conf", "class_true", "class_pred", "label_true", "label_pred"}
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns in {pred_csv}: {miss}")

    err = df[df["is_error"] == 1].copy()
    if err.empty:
        print("No errors found in predictions.")
        return

    err["pair"] = err["class_true"].astype(str) + " -> " + err["class_pred"].astype(str)
    pair_count = err["pair"].value_counts().to_dict()
    err["pair_count"] = err["pair"].map(pair_count).astype(int)
    if args.sort == "pair_then_conf":
        err = err.sort_values(["pair_count", "conf"], ascending=[False, False]).reset_index(drop=True)
    else:
        err = err.sort_values(["conf"], ascending=[False]).reset_index(drop=True)

    err["abs_image_path"] = err["image_id"].map(lambda x: str((image_root / str(x)).resolve()))
    out_csv = out_dir / "errors_with_paths.csv"
    err.to_csv(out_csv, index=False)

    tiles: List[Image.Image] = []
    missing = []
    for idx, row in err.iterrows():
        path = image_root / str(row["image_id"])
        if not path.exists():
            missing.append(str(path))
            continue
        im = safe_open_rgb(path)
        text = (
            f"#{idx+1} conf={row['conf']:.3f}\n"
            f"TRUE[{row['label_true']}]: {row['class_true']}\n"
            f"PRED[{row['label_pred']}]: {row['class_pred']}"
        )
        tiles.append(tile_image(im, args.thumb_w, args.thumb_h, args.text_h, text, font))

    pages = save_pages(tiles, args.cols, args.rows, out_dir)

    # Also save one image per confusion pair (up to 20 examples per pair).
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for pair, part in err.groupby("pair", sort=False):
        part = part.head(20)
        pair_tiles: List[Image.Image] = []
        for _, row in part.iterrows():
            path = image_root / str(row["image_id"])
            if not path.exists():
                continue
            im = safe_open_rgb(path)
            txt = (
                f"conf={row['conf']:.3f}\n"
                f"TRUE[{row['label_true']}]: {row['class_true']}\n"
                f"PRED[{row['label_pred']}]: {row['class_pred']}"
            )
            pair_tiles.append(tile_image(im, args.thumb_w, args.thumb_h, args.text_h, txt, font))
        if not pair_tiles:
            continue
        pair_canvas = Image.new(
            "RGB",
            (min(len(pair_tiles), args.cols) * pair_tiles[0].width, ((len(pair_tiles) - 1) // args.cols + 1) * pair_tiles[0].height),
            (255, 255, 255),
        )
        tw = pair_tiles[0].width
        th = pair_tiles[0].height
        for j, t in enumerate(pair_tiles):
            rr = j // args.cols
            cc = j % args.cols
            pair_canvas.paste(t, (cc * tw, rr * th))
        safe_name = (
            pair.replace("/", "_")
            .replace(" ", "_")
            .replace("->", "to")
            .replace(":", "")
            .replace("|", "_")
        )
        pair_canvas.save(pair_dir / f"{safe_name}.png")

    missing_path = out_dir / "missing_images.txt"
    missing_path.write_text("\n".join(missing), encoding="utf-8")

    print("saved_csv:", out_csv)
    print("saved_pages:", len(pages))
    for p in pages:
        print("page:", p)
    print("pair_dir:", pair_dir)
    print("missing:", len(missing), "file:", missing_path)


if __name__ == "__main__":
    main()
