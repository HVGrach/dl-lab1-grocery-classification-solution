#!/usr/bin/env python3
"""Prepare cleaned train CSV files without touching test data.

Outputs are written under:
  dl_lab1/unzipped/cleaning/
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, UnidentifiedImageError


MIN_SIDE_DROP = 56
MANUAL_BAD_BASENAMES = {
    # visually confirmed non-target/noise samples
    "11473eb90c1143d7a7f1ba0a255b94ba.jpg",
    "c250a64e743746bb91d9009347566c3b.jpg",
    "376054ec56424fedb66f3a45791e85be.jpg",
    "45b44e99eeb4468387124d89c14662d1.jpg",
    "7bf115af344a487d9b9eb662567c4f9a.jpg",
}


@dataclass
class Row:
    image_id: str
    label: int
    class_name: str
    plu: str
    basename: str


def load_rows(train_csv: Path) -> List[Row]:
    rows: List[Row] = []
    with train_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            image_id = rec["image_id"]
            class_name, plu, basename = image_id.split("/")
            rows.append(
                Row(
                    image_id=image_id,
                    label=int(rec["label"]),
                    class_name=class_name,
                    plu=plu,
                    basename=basename,
                )
            )
    return rows


def hash_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    rank = np.zeros(len(values), dtype=np.float32)
    n = len(values)
    for pos, idx in enumerate(order):
        rank[idx] = (pos + 1) / n
    return rank


def main() -> None:
    root = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped")
    train_csv = root / "train.csv"
    train_root = root / "train" / "train"
    out_dir = root / "cleaning"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(train_csv)
    n = len(rows)

    width = np.zeros(n, dtype=np.int32)
    height = np.zeros(n, dtype=np.int32)
    min_side = np.zeros(n, dtype=np.int32)
    file_size = np.zeros(n, dtype=np.int64)
    blur_score = np.zeros(n, dtype=np.float32)
    is_corrupt = np.zeros(n, dtype=bool)

    # 32x32 RGB features for lightweight semantic checks.
    feats = np.zeros((n, 32 * 32 * 3), dtype=np.float32)
    labels = np.array([r.label for r in rows], dtype=np.int32)

    md5_to_indices: Dict[str, List[int]] = defaultdict(list)

    for i, row in enumerate(rows):
        path = train_root / row.image_id
        file_size[i] = path.stat().st_size
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                width[i] = w
                height[i] = h
                min_side[i] = min(w, h)

                arr = np.asarray(rgb, dtype=np.float32)
                gray = arr.mean(axis=2)
                gx = np.diff(gray, axis=1)
                gy = np.diff(gray, axis=0)
                blur_score[i] = float(np.var(gx) + np.var(gy))

                small = np.asarray(
                    rgb.resize((32, 32), Image.BILINEAR), dtype=np.float32
                )
                feats[i] = (small / 255.0).reshape(-1)
        except (UnidentifiedImageError, OSError, ValueError):
            is_corrupt[i] = True

        md5_to_indices[hash_file(path)].append(i)

    valid_idx = np.where(~is_corrupt)[0]
    x = feats[valid_idx]
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    valid_labels = labels[valid_idx]
    valid_global_to_local = {g: i for i, g in enumerate(valid_idx.tolist())}

    classes = sorted(np.unique(valid_labels).tolist())
    class_to_pos = {c: i for i, c in enumerate(classes)}
    centroids = np.zeros((len(classes), x_norm.shape[1]), dtype=np.float32)
    for c in classes:
        v = x_norm[valid_labels == c].mean(axis=0)
        v /= np.linalg.norm(v) + 1e-8
        centroids[class_to_pos[c]] = v

    mismatch_margin = np.zeros(n, dtype=np.float32)
    pred_label = np.full(n, -1, dtype=np.int32)
    for local_i, global_i in enumerate(valid_idx):
        sims = x_norm[local_i] @ centroids.T
        best_pos = int(np.argmax(sims))
        best_label = classes[best_pos]
        true_pos = class_to_pos[labels[global_i]]
        pred_label[global_i] = best_label
        if best_label != labels[global_i]:
            mismatch_margin[global_i] = float(sims[best_pos] - sims[true_pos])

    group_rank = np.ones(n, dtype=np.float32)
    grouped: Dict[tuple[int, str], List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[(row.label, row.plu)].append(i)

    for (_, _), idxs in grouped.items():
        valid_group = [j for j in idxs if not is_corrupt[j]]
        if len(valid_group) < 8:
            continue
        local = np.array([valid_global_to_local[j] for j in valid_group], dtype=int)
        m = x_norm[local]
        c = m.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-8
        sims = m @ c
        ranks = quantile_rank(sims)
        for pos, j in enumerate(valid_group):
            group_rank[j] = ranks[pos]

    blur_rank = np.ones(n, dtype=np.float32)
    if len(valid_idx):
        ranks = quantile_rank(blur_score[valid_idx])
        for pos, j in enumerate(valid_idx):
            blur_rank[j] = ranks[pos]

    duplicate_drop = set()
    for _, idxs in md5_to_indices.items():
        if len(idxs) > 1:
            for j in sorted(idxs)[1:]:
                duplicate_drop.add(j)

    drop_reasons: Dict[int, List[str]] = defaultdict(list)
    quarantine_reasons: Dict[int, List[str]] = defaultdict(list)

    for i, row in enumerate(rows):
        if is_corrupt[i]:
            drop_reasons[i].append("corrupt_image")
        if min_side[i] < MIN_SIDE_DROP:
            drop_reasons[i].append(f"too_small_min_side_lt_{MIN_SIDE_DROP}")
        if row.basename in MANUAL_BAD_BASENAMES:
            drop_reasons[i].append("manual_semantic_noise")
        if i in duplicate_drop:
            drop_reasons[i].append("exact_duplicate_md5")

        if drop_reasons[i]:
            continue

        if mismatch_margin[i] >= 0.02 and group_rank[i] <= 0.01:
            quarantine_reasons[i].append("semantic_outlier_group_and_class_mismatch")
        if blur_rank[i] <= 0.003 and min_side[i] < 72:
            quarantine_reasons[i].append("low_detail_low_resolution")
        if group_rank[i] <= 0.003 and min_side[i] < 96:
            quarantine_reasons[i].append("extreme_group_outlier_low_resolution")

    manifest_path = out_dir / "train_clean_manifest.csv"
    strict_csv_path = out_dir / "train_clean_strict.csv"
    aggressive_csv_path = out_dir / "train_clean_aggressive.csv"
    quarantine_path = out_dir / "train_quarantine.csv"
    summary_path = out_dir / "summary.json"

    keep_strict = []
    keep_aggressive = []
    quarantine_rows = []

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "image_id",
            "label",
            "class_name",
            "plu",
            "basename",
            "width",
            "height",
            "min_side",
            "file_size",
            "blur_score",
            "class_mismatch_margin",
            "pred_label",
            "group_rank",
            "blur_rank",
            "status",
            "drop_reasons",
            "quarantine_reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            if drop_reasons[i]:
                status = "drop"
            elif quarantine_reasons[i]:
                status = "quarantine"
            else:
                status = "keep"

            rec = {
                "image_id": row.image_id,
                "label": row.label,
                "class_name": row.class_name,
                "plu": row.plu,
                "basename": row.basename,
                "width": int(width[i]),
                "height": int(height[i]),
                "min_side": int(min_side[i]),
                "file_size": int(file_size[i]),
                "blur_score": float(blur_score[i]),
                "class_mismatch_margin": float(mismatch_margin[i]),
                "pred_label": int(pred_label[i]),
                "group_rank": float(group_rank[i]),
                "blur_rank": float(blur_rank[i]),
                "status": status,
                "drop_reasons": "|".join(drop_reasons[i]),
                "quarantine_reasons": "|".join(quarantine_reasons[i]),
            }
            writer.writerow(rec)

            base_rec = {"image_id": row.image_id, "label": row.label}
            if status != "drop":
                keep_strict.append(base_rec)
            if status == "keep":
                keep_aggressive.append(base_rec)
            if status == "quarantine":
                quarantine_rows.append(rec)

    for path, recs in (
        (strict_csv_path, keep_strict),
        (aggressive_csv_path, keep_aggressive),
    ):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
            writer.writeheader()
            writer.writerows(recs)

    with quarantine_path.open("w", encoding="utf-8-sig", newline="") as f:
        if quarantine_rows:
            writer = csv.DictWriter(f, fieldnames=list(quarantine_rows[0].keys()))
            writer.writeheader()
            writer.writerows(quarantine_rows)
        else:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "label",
                    "class_name",
                    "plu",
                    "basename",
                    "width",
                    "height",
                    "min_side",
                    "file_size",
                    "blur_score",
                    "class_mismatch_margin",
                    "pred_label",
                    "group_rank",
                    "blur_rank",
                    "status",
                    "drop_reasons",
                    "quarantine_reasons",
                ],
            )
            writer.writeheader()

    class_total = Counter(r.class_name for r in rows)
    class_kept_strict = Counter(r["image_id"].split("/")[0] for r in keep_strict)
    class_kept_aggressive = Counter(r["image_id"].split("/")[0] for r in keep_aggressive)

    summary = {
        "total_rows": n,
        "drop_count": int(sum(1 for i in range(n) if drop_reasons[i])),
        "quarantine_count": int(sum(1 for i in range(n) if quarantine_reasons[i] and not drop_reasons[i])),
        "keep_count": int(sum(1 for i in range(n) if not drop_reasons[i] and not quarantine_reasons[i])),
        "strict_count": len(keep_strict),
        "aggressive_count": len(keep_aggressive),
        "drop_reason_counts": Counter(
            reason for i in range(n) for reason in drop_reasons[i]
        ),
        "quarantine_reason_counts": Counter(
            reason
            for i in range(n)
            for reason in quarantine_reasons[i]
            if not drop_reasons[i]
        ),
        "class_total": class_total,
        "class_kept_strict": class_kept_strict,
        "class_kept_aggressive": class_kept_aggressive,
        "paths": {
            "manifest": str(manifest_path),
            "train_clean_strict": str(strict_csv_path),
            "train_clean_aggressive": str(aggressive_csv_path),
            "quarantine": str(quarantine_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"manifest: {manifest_path}")
    print(f"strict:   {strict_csv_path} ({len(keep_strict)} rows)")
    print(f"aggr:     {aggressive_csv_path} ({len(keep_aggressive)} rows)")
    print(f"summary:  {summary_path}")


if __name__ == "__main__":
    main()
