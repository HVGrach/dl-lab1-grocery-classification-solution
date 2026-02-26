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
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Kaggle submission from night autopilot best ensemble.")
    p.add_argument(
        "--base",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1",
    )
    p.add_argument(
        "--autopilot-summary",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/autopilot_summary.json",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tta-mode", type=str, default="none", choices=["none", "flip", "geo4", "geo8"])
    p.add_argument(
        "--tta-views",
        type=int,
        default=0,
        help="Optional cap on number of TTA views (0 = use all views from selected mode).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/submission_night_best3_weighted.csv",
    )
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
    # Color is intentionally untouched; only geometric TTA.
    views: List[torch.Tensor] = [x]
    if mode == "none":
        pass
    elif mode == "flip":
        views.append(torch.flip(x, dims=[3]))  # horizontal
    elif mode == "geo4":
        views.extend(
            [
                torch.flip(x, dims=[3]),  # horizontal
                torch.flip(x, dims=[2]),  # vertical
                torch.rot90(x, k=2, dims=[2, 3]),  # 180
            ]
        )
    elif mode == "geo8":
        t = x.transpose(2, 3)
        views.extend(
            [
                torch.flip(x, dims=[3]),  # horizontal
                torch.flip(x, dims=[2]),  # vertical
                torch.rot90(x, k=1, dims=[2, 3]),  # 90
                torch.rot90(x, k=2, dims=[2, 3]),  # 180
                torch.rot90(x, k=3, dims=[2, 3]),  # 270
                t,  # transpose
                torch.flip(t, dims=[3]),  # transpose + horizontal
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
    views_count = 1
    with torch.no_grad():
        for x, image_ids in tqdm(loader, desc="infer", leave=False):
            x = x.to(device)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()
            views = build_tta_views(x, mode=tta_mode, max_views=tta_views)
            views_count = len(views)
            logits = None
            for xv in views:
                if use_channels_last:
                    xv = xv.contiguous(memory_format=torch.channels_last)
                else:
                    xv = xv.contiguous()
                lv = model(xv)
                logits = lv if logits is None else (logits + lv)
            logits = logits / float(len(views))
            all_logits.append(logits.detach().cpu())
            all_ids.extend(list(image_ids))
    logits_np = torch.cat(all_logits).numpy()
    probs = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    return probs.astype(np.float32), all_ids, int(views_count)


def main() -> None:
    args = parse_args()
    os_env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    for k, v in os_env.items():
        if k not in __import__("os").environ:
            __import__("os").environ[k] = v

    device = resolve_device(args.device)
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
    print("Device:", device)

    base = Path(args.base)
    summary_path = Path(args.autopilot_summary)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["ensemble"]["best"]
    model_names: List[str] = best["models"]
    weights = np.array(best["weights"], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    tta_mode = args.tta_mode
    if args.tta and tta_mode == "none":
        # Backward compatibility: old flag --tta means flip TTA.
        tta_mode = "flip"

    runs_root = summary_path.parent / "runs"
    run_dirs = {p.name.split("_", 1)[1]: p for p in runs_root.iterdir() if p.is_dir() and "_" in p.name}

    test_df = pd.read_csv(base / "test.csv")
    sample_sub = pd.read_csv(base / "sample_submission.csv")
    test_img_dir = base / "test_images" / "test_images"

    num_classes = 15
    probs_list = []
    ids_ref: List[str] | None = None
    per_model_tta_views: Dict[str, int] = {}

    for alias in model_names:
        run_dir = run_dirs.get(alias)
        if run_dir is None:
            raise FileNotFoundError(f"Run dir for alias '{alias}' not found in {runs_root}")

        run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

        final_selected = run_summary.get("final_model_selected", "best_checkpoint")
        if final_selected == "swa_model" and (run_dir / "swa_model.pt").exists():
            ckpt_path = run_dir / "swa_model.pt"
        else:
            ckpt_path = run_dir / "best_by_val_loss.pt"

        print(f"Loading: {alias} from {ckpt_path.name} (selected={final_selected})")
        model = create_model_safe(
            name=cfg["model_name"],
            num_classes=num_classes,
            drop_rate=0.2,
            drop_path_rate=0.2,
        ).to(device)

        use_channels_last = bool(cfg.get("use_channels_last", False))
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)

        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)

        tfm = build_valid_tfms(int(cfg["img_size"]))
        ds = TestDataset(test_df, test_img_dir, tfm)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

        probs, ids, n_views = predict_probs(
            model,
            loader,
            device,
            use_channels_last,
            tta_mode=tta_mode,
            tta_views=args.tta_views,
        )
        probs_list.append(probs)
        per_model_tta_views[alias] = int(n_views)

        if ids_ref is None:
            ids_ref = ids
        elif ids_ref != ids:
            raise RuntimeError("Image order mismatch across models")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    blend = np.zeros_like(probs_list[0], dtype=np.float64)
    for w, p in zip(weights, probs_list):
        blend += w * p
    pred = blend.argmax(1)

    out = sample_sub.copy()
    out["label"] = pred
    out.to_csv(out_csv, index=False)

    meta = {
        "autopilot_summary": str(summary_path),
        "models": model_names,
        "weights": [float(x) for x in weights],
        "tta_mode": tta_mode,
        "tta_views_limit": int(args.tta_views),
        "per_model_tta_views": per_model_tta_views,
        "device": str(device),
        "out_csv": str(out_csv),
    }
    (out_csv.parent / (out_csv.stem + "_meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved submission:", out_csv)
    print("Label distribution:", out["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
