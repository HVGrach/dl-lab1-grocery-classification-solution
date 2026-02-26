#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DEFAULT_TRAIN_ROOT = Path(
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/train/train"
)
DEFAULT_BASE_TRAIN_CSV = Path(
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1/train.csv"
)
DEFAULT_ZOO_ROOT = Path(
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5"
)
DEFAULT_OUT_CSV = Path(
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/confidence_cache/top_new_dataset_ensemble_confidence.csv"
)
DEFAULT_OUT_META = DEFAULT_OUT_CSV.with_suffix(".meta.json")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".jfif",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class EnsembleEntry:
    fold_idx: int
    name: str
    weight: float
    run_dir: Path
    checkpoint: Path
    config: Path


class ImageListDataset(Dataset):
    def __init__(self, rel_paths: List[str], img_root: Path, transform: A.Compose):
        self.rel_paths = list(rel_paths)
        self.img_root = img_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        rel_path = self.rel_paths[idx]
        image = np.array(Image.open(self.img_root / rel_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, rel_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build confidence cache for Dataset Tinder review app using best CV5 ensemble: "
            "reuse OOF probs for known train images and infer missing/new images on MPS/CPU."
        )
    )
    p.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    p.add_argument("--base-train-csv", type=Path, default=DEFAULT_BASE_TRAIN_CSV)
    p.add_argument("--zoo-root", type=Path, default=DEFAULT_ZOO_ROOT)
    p.add_argument("--ensemble-meta", type=Path, default=None, help="Defaults to submission_*_meta.json in zoo root")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to ensemble_cv5_lr_geo8_nonzero_15models_manifest.json in zoo root",
    )
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-meta-json", type=Path, default=DEFAULT_OUT_META)
    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cpu", "none"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--force-infer-all", action="store_true", help="Ignore OOF cache and infer all images via ensemble")
    p.add_argument("--max-infer-images", type=int, default=0, help="Optional cap for debugging (0 = all)")
    return p.parse_args()


def resolve_device(requested: str) -> Optional[torch.device]:
    if requested == "none":
        return None
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_valid_tfms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def create_model_safe(
    name: str,
    num_classes: int,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.2,
) -> torch.nn.Module:
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
    all_logits: List[torch.Tensor] = []
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


def scan_target_images(train_root: Path) -> List[str]:
    paths = [
        p.relative_to(train_root).as_posix()
        for p in train_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    return paths


def build_label_maps(base_train_csv: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    df = pd.read_csv(base_train_csv)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise RuntimeError("base train csv must contain image_id and label columns")
    pairs = (
        df.assign(class_name=df["image_id"].astype(str).str.split("/").str[0])
        .loc[:, ["label", "class_name"]]
        .drop_duplicates()
    )
    by_label: Dict[int, str] = {}
    by_class: Dict[str, int] = {}
    for _, r in pairs.iterrows():
        label = int(r["label"])
        class_name = str(r["class_name"])
        if label in by_label and by_label[label] != class_name:
            raise RuntimeError(f"Conflicting class for label {label}: {by_label[label]} vs {class_name}")
        by_label[label] = class_name
        if class_name in by_class and by_class[class_name] != label:
            raise RuntimeError(f"Conflicting label for class {class_name}: {by_class[class_name]} vs {label}")
        by_class[class_name] = label
    return dict(sorted(by_label.items())), by_class


def find_default_ensemble_meta(zoo_root: Path) -> Path:
    preferred = zoo_root / "submission_cv5_all20_lr_geo8_equal_meta.json"
    if preferred.exists():
        return preferred
    candidates = sorted(zoo_root.glob("submission*_meta.json"))
    if not candidates:
        raise FileNotFoundError(f"No submission meta json found in {zoo_root}")
    return candidates[-1]


def find_default_manifest(zoo_root: Path) -> Path:
    preferred = zoo_root / "ensemble_cv5_lr_geo8_nonzero_15models_manifest.json"
    if preferred.exists():
        return preferred
    candidates = sorted(zoo_root.glob("*manifest*.json"))
    if not candidates:
        raise FileNotFoundError(f"No ensemble manifest json found in {zoo_root}")
    return candidates[-1]


def load_manifest_entries(manifest_path: Path) -> List[EnsembleEntry]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = []
    for e in obj.get("entries", []):
        entries.append(
            EnsembleEntry(
                fold_idx=int(e["fold_idx"]),
                name=str(e["name"]),
                weight=float(e["weight"]),
                run_dir=Path(e["run_dir"]),
                checkpoint=Path(e["checkpoint"]),
                config=Path(e["config"]),
            )
        )
    if not entries:
        raise RuntimeError(f"No manifest entries in {manifest_path}")
    return entries


def reconstruct_oof_confidence(
    meta: Dict,
    manifest_entries: List[EnsembleEntry],
    num_classes: int,
) -> Dict[str, np.ndarray]:
    by_fold_name = {(e.fold_idx, e.name): e for e in manifest_entries}
    oof_map: Dict[str, np.ndarray] = {}
    for fold_key, fold_info in sorted((meta.get("folds") or {}).items(), key=lambda kv: int(kv[0])):
        fold_idx = int(fold_key)
        names = [str(x) for x in fold_info.get("models", [])]
        weights = [float(x) for x in fold_info.get("weights_lr_mse", [])]
        if len(names) != len(weights):
            raise RuntimeError(f"Fold {fold_idx}: models/weights length mismatch")
        used = [(n, w) for n, w in zip(names, weights) if abs(float(w)) > 1e-12]
        if not used:
            raise RuntimeError(f"Fold {fold_idx}: no non-zero weights")

        ids_ref: Optional[List[str]] = None
        fold_blend: Optional[np.ndarray] = None
        for model_name, weight in used:
            entry = by_fold_name.get((fold_idx, model_name))
            if entry is None:
                raise RuntimeError(f"Fold {fold_idx}: manifest missing entry for model {model_name}")
            run_dir = entry.run_dir
            val_probs = np.load(run_dir / "val_probs.npy").astype(np.float32)
            if val_probs.shape[1] != num_classes:
                raise RuntimeError(f"{run_dir}: val_probs classes={val_probs.shape[1]} expected={num_classes}")
            val_pred_df = pd.read_csv(run_dir / "val_predictions.csv")
            ids = val_pred_df["image_id"].astype(str).tolist()
            if ids_ref is None:
                ids_ref = ids
                fold_blend = np.zeros_like(val_probs, dtype=np.float64)
            else:
                if ids_ref != ids:
                    raise RuntimeError(f"Fold {fold_idx}: val_predictions order mismatch in {run_dir}")
            fold_blend += float(weight) * val_probs.astype(np.float64)

        assert ids_ref is not None and fold_blend is not None
        for rel_path, probs in zip(ids_ref, fold_blend):
            if rel_path in oof_map:
                raise RuntimeError(f"Duplicate OOF image_id across folds: {rel_path}")
            oof_map[rel_path] = probs.astype(np.float32, copy=False)
    return oof_map


def build_loader_cache(
    rel_paths: List[str],
    train_root: Path,
    batch_size: int,
    num_workers: int,
) -> Dict[int, DataLoader]:
    cache: Dict[int, DataLoader] = {}

    def get_loader(img_size: int) -> DataLoader:
        if img_size not in cache:
            ds = ImageListDataset(rel_paths, train_root, build_valid_tfms(img_size))
            cache[img_size] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
        return cache[img_size]

    return {"_getter": get_loader}  # type: ignore[return-value]


def infer_confidence_for_paths(
    rel_paths: List[str],
    train_root: Path,
    meta: Dict,
    manifest_entries: List[EnsembleEntry],
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    if not rel_paths:
        return {}

    tta_mode = str(meta.get("tta_mode", "geo8"))
    tta_views = int(meta.get("tta_views_limit", 0))
    fold_weights_list = [float(x) for x in meta.get("fold_weights", [])]
    if not fold_weights_list:
        unique_folds = sorted({e.fold_idx for e in manifest_entries})
        fold_weights = {f: 1.0 / len(unique_folds) for f in unique_folds}
    else:
        fold_weights = {i: float(w) for i, w in enumerate(fold_weights_list)}

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    loader_cache: Dict[int, DataLoader] = {}

    def get_loader(img_size: int) -> DataLoader:
        if img_size not in loader_cache:
            ds = ImageListDataset(rel_paths, train_root, build_valid_tfms(img_size))
            loader_cache[img_size] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
        return loader_cache[img_size]

    fold_blends: Dict[int, np.ndarray] = {}
    ids_ref: Optional[List[str]] = None
    total = len(manifest_entries)
    for idx_entry, entry in enumerate(sorted(manifest_entries, key=lambda e: (e.fold_idx, e.name)), start=1):
        print(
            f"[infer {idx_entry}/{total}] fold={entry.fold_idx} {entry.name} "
            f"w={entry.weight:.6f} ckpt={entry.checkpoint.name}",
            flush=True,
        )
        cfg = json.loads(entry.config.read_text(encoding="utf-8"))
        model = create_model_safe(
            name=str(cfg["model_name"]),
            num_classes=num_classes,
            drop_rate=0.2,
            drop_path_rate=0.2,
        ).to(device)
        use_channels_last = bool(cfg.get("use_channels_last", False))
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)

        ckpt = torch.load(entry.checkpoint, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)

        loader = get_loader(int(cfg["img_size"]))
        probs, ids, n_views = predict_probs(
            model=model,
            loader=loader,
            device=device,
            use_channels_last=use_channels_last,
            tta_mode=tta_mode,
            tta_views=tta_views,
        )
        print(f"  -> views={n_views}, batch={batch_size}, imgs={len(ids)}", flush=True)
        if ids_ref is None:
            ids_ref = ids
        elif ids_ref != ids:
            raise RuntimeError("Image order mismatch across models during inference")

        if entry.fold_idx not in fold_blends:
            fold_blends[entry.fold_idx] = np.zeros_like(probs, dtype=np.float64)
        fold_blends[entry.fold_idx] += float(entry.weight) * probs.astype(np.float64)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    if ids_ref is None:
        return {}

    final = np.zeros((len(ids_ref), num_classes), dtype=np.float64)
    fold_weight_sum = 0.0
    for fold_idx, blend in sorted(fold_blends.items()):
        fw = float(fold_weights.get(fold_idx, 0.0))
        final += fw * blend
        fold_weight_sum += fw
    if fold_weight_sum <= 0:
        raise RuntimeError("Fold weights sum to 0 during inference blend")
    final /= fold_weight_sum

    return {rel: final[i].astype(np.float32, copy=False) for i, rel in enumerate(ids_ref)}


def write_cache(
    out_csv: Path,
    out_meta: Path,
    target_rel_paths: List[str],
    merged: Dict[str, Tuple[str, np.ndarray]],
    label_to_class: Dict[int, str],
    class_to_label: Dict[str, int],
    source_counts: Dict[str, int],
    extra_meta: Dict,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    prob_cols = [f"prob_{i}" for i in range(len(label_to_class))]
    fieldnames = ["relative_path", "source", "pred_label", "pred_class"] + prob_cols
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rel in target_rel_paths:
            if rel not in merged:
                continue
            source, probs = merged[rel]
            pred_label = int(np.argmax(probs))
            row = {
                "relative_path": rel,
                "source": source,
                "pred_label": pred_label,
                "pred_class": label_to_class.get(pred_label, f"label_{pred_label}"),
            }
            for i, p in enumerate(probs.tolist()):
                row[f"prob_{i}"] = f"{float(p):.8f}"
            writer.writerow(row)

    meta = {
        "cache_type": "dataset_tinder_confidence_cache",
        "label_to_class": {str(k): v for k, v in label_to_class.items()},
        "class_to_label": {k: int(v) for k, v in class_to_label.items()},
        "num_rows": sum(source_counts.values()),
        "source_counts": source_counts,
        "out_csv": str(out_csv),
        **extra_meta,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    train_root = args.train_root.expanduser().resolve()
    base_train_csv = args.base_train_csv.expanduser().resolve()
    zoo_root = args.zoo_root.expanduser().resolve()
    ensemble_meta_path = (
        args.ensemble_meta.expanduser().resolve()
        if args.ensemble_meta is not None
        else find_default_ensemble_meta(zoo_root)
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else find_default_manifest(zoo_root)
    )
    out_csv = args.out_csv.expanduser().resolve()
    out_meta = args.out_meta_json.expanduser().resolve()

    if not train_root.is_dir():
        raise FileNotFoundError(f"train root not found: {train_root}")
    if not base_train_csv.is_file():
        raise FileNotFoundError(f"base train csv not found: {base_train_csv}")
    if not ensemble_meta_path.is_file():
        raise FileNotFoundError(f"ensemble meta not found: {ensemble_meta_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    label_to_class, class_to_label = build_label_maps(base_train_csv)
    num_classes = len(label_to_class)
    print(f"classes: {num_classes}", flush=True)
    print("label map:", label_to_class, flush=True)

    target_rel_paths = scan_target_images(train_root)
    print(f"target images: {len(target_rel_paths)}", flush=True)

    meta = json.loads(ensemble_meta_path.read_text(encoding="utf-8"))
    manifest_entries = load_manifest_entries(manifest_path)
    print(f"manifest entries: {len(manifest_entries)}", flush=True)

    oof_map: Dict[str, np.ndarray] = {}
    if not args.force_infer_all:
        print("reconstructing OOF ensemble confidence from saved val_probs...", flush=True)
        oof_map = reconstruct_oof_confidence(meta, manifest_entries, num_classes=num_classes)
        print(f"OOF map size: {len(oof_map)}", flush=True)

    if args.force_infer_all:
        need_infer = list(target_rel_paths)
    else:
        need_infer = [p for p in target_rel_paths if p not in oof_map]

    if args.max_infer_images and args.max_infer_images > 0:
        need_infer = need_infer[: int(args.max_infer_images)]
        print(f"[debug] capped infer set to {len(need_infer)} images", flush=True)

    print(
        f"coverage from OOF: {sum(1 for p in target_rel_paths if p in oof_map)}/{len(target_rel_paths)}; "
        f"need infer: {len(need_infer)}",
        flush=True,
    )

    infer_map: Dict[str, np.ndarray] = {}
    if need_infer:
        device = resolve_device(args.device)
        if device is None:
            raise RuntimeError(
                f"{len(need_infer)} images require inference but --device=none was specified"
            )
        print(f"infer device: {device}", flush=True)
        infer_map = infer_confidence_for_paths(
            rel_paths=need_infer,
            train_root=train_root,
            meta=meta,
            manifest_entries=manifest_entries,
            num_classes=num_classes,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        print(f"inferred images: {len(infer_map)}", flush=True)

    merged: Dict[str, Tuple[str, np.ndarray]] = {}
    source_counts = {"oof": 0, "infer": 0}
    for rel in target_rel_paths:
        if rel in infer_map:
            merged[rel] = ("infer", infer_map[rel])
            source_counts["infer"] += 1
        elif rel in oof_map:
            merged[rel] = ("oof", oof_map[rel])
            source_counts["oof"] += 1

    missing_after = [p for p in target_rel_paths if p not in merged]
    if missing_after:
        print(f"[warn] unresolved images: {len(missing_after)}", flush=True)
        print("sample unresolved:", missing_after[:10], flush=True)

    write_cache(
        out_csv=out_csv,
        out_meta=out_meta,
        target_rel_paths=target_rel_paths,
        merged=merged,
        label_to_class=label_to_class,
        class_to_label=class_to_label,
        source_counts=source_counts,
        extra_meta={
            "train_root": str(train_root),
            "base_train_csv": str(base_train_csv),
            "zoo_root": str(zoo_root),
            "ensemble_meta": str(ensemble_meta_path),
            "manifest": str(manifest_path),
            "force_infer_all": bool(args.force_infer_all),
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "missing_unresolved": len(missing_after),
        },
    )

    print("\n=== DONE ===", flush=True)
    print("out_csv:", out_csv, flush=True)
    print("out_meta:", out_meta, flush=True)
    print("source_counts:", source_counts, flush=True)
    print("resolved:", len(merged), "/", len(target_rel_paths), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
