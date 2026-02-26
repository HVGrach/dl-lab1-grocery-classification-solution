#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


LABEL_TO_CLASS: Dict[int, str] = {
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
CLASS_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_TO_CLASS.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply Tinder review actions to top_new_dataset (files + train.csv + manual_review_queue manifest)."
    )
    p.add_argument("--actions-csv", type=Path, required=True)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset"),
    )
    p.add_argument(
        "--batch-name",
        type=str,
        default=f"tinder_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for logs/summary (default: sibling folder near actions csv)",
    )
    p.add_argument("--apply", action="store_true", help="Actually move files and overwrite CSVs")
    return p.parse_args()


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 1
    while True:
        c = p.with_name(f"{stem}__dup{i}{suffix}")
        if not c.exists():
            return c
        i += 1


def rel_from_root(abs_path: Path, root: Path) -> str:
    return abs_path.resolve().relative_to(root.resolve()).as_posix()


def relabel_rel_path(old_rel: str, target_class: str) -> str:
    old = Path(old_rel)
    if len(old.parts) < 3:
        raise ValueError(f"Unexpected image_id format: {old_rel}")
    plu = old.parts[1]
    basename = old.parts[-1]
    return (Path(target_class) / plu / basename).as_posix()


def scan_active_image_files(train_root: Path) -> List[Path]:
    out: List[Path] = []
    for class_dir in [p for p in train_root.iterdir() if p.is_dir()]:
        if class_dir.name.startswith("_"):
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                out.append(p)
    return out


def main() -> int:
    args = parse_args()
    actions_csv = args.actions_csv.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    batch_name = args.batch_name
    run_apply = bool(args.apply)

    train_root = dataset_root / "train" / "train"
    train_csv = dataset_root / "train.csv"
    queue_root = dataset_root / "cleaning" / "manual_review_queue"
    queue_manifest_csv = queue_root / "manifest.csv"
    queue_summary_json = queue_root / "queue_summary.json"
    quarantine_root = dataset_root / "_manual_quarantine_batches" / batch_name

    if not actions_csv.is_file():
        raise FileNotFoundError(f"actions csv not found: {actions_csv}")
    if not train_root.is_dir():
        raise FileNotFoundError(f"train root not found: {train_root}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not queue_manifest_csv.is_file():
        raise FileNotFoundError(f"queue manifest not found: {queue_manifest_csv}")

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        out_dir = actions_csv.parent / f"apply_top_new_dataset_{batch_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_actions = pd.read_csv(actions_csv)
    need_cols = {"abs_path", "action"}
    miss = need_cols - set(df_actions.columns)
    if miss:
        raise ValueError(f"actions csv missing columns: {sorted(miss)}")

    train_df = pd.read_csv(train_csv)
    manifest_df = pd.read_csv(queue_manifest_csv)
    for col in ["image_id"]:
        if col not in train_df.columns:
            raise ValueError(f"{train_csv} missing column {col}")
        if col not in manifest_df.columns:
            raise ValueError(f"{queue_manifest_csv} missing column {col}")
    if "label" not in train_df.columns:
        raise ValueError(f"{train_csv} missing column label")

    # Keep strings stable.
    if "plu" in manifest_df.columns:
        manifest_df["plu"] = manifest_df["plu"].astype(str)

    train_before = len(train_df)
    manifest_before = len(manifest_df)

    logs: List[Dict[str, object]] = []
    action_counts = Counter()
    skipped_missing = 0
    moved_files = 0
    relabeled = 0
    trashed = 0
    train_removed = 0
    manifest_removed = 0

    # Keep a deterministic order: trash first then relabel.
    priority = {"trash": 0, "relabel": 1}
    sort_key = []
    if "relative_path" in df_actions.columns:
        sort_key.append("relative_path")
    else:
        sort_key.append("abs_path")
    df_actions["_priority"] = df_actions["action"].astype(str).map(lambda a: priority.get(a, 99))
    df_actions = df_actions.sort_values(["_priority"] + sort_key).drop(columns=["_priority"])

    for row in df_actions.itertuples(index=False):
        abs_path = Path(str(row.abs_path))
        action = str(row.action).strip()
        if action not in {"trash", "relabel"}:
            logs.append({"abs_path": str(abs_path), "action": action, "status": "unsupported_action"})
            continue
        action_counts[action] += 1

        rec: Dict[str, object] = {
            "abs_path_old": str(abs_path),
            "action": action,
            "status": "",
        }

        if not abs_path.exists():
            rec["status"] = "missing_source"
            logs.append(rec)
            skipped_missing += 1
            continue
        try:
            rel_old = rel_from_root(abs_path, train_root)
        except Exception:
            rec["status"] = "outside_train_root"
            logs.append(rec)
            continue

        # target metadata
        target_class = ""
        if hasattr(row, "target_class"):
            v = getattr(row, "target_class")
            if isinstance(v, str):
                target_class = v.strip()
            elif pd.notna(v):
                target_class = str(v).strip()

        target_label = None
        if hasattr(row, "target_label"):
            v = getattr(row, "target_label")
            if pd.notna(v) and str(v).strip() != "":
                target_label = int(float(v))

        if action == "relabel":
            if not target_class:
                rec["status"] = "error_no_target_class"
                logs.append(rec)
                continue
            if target_label is None:
                if target_class not in CLASS_TO_LABEL:
                    rec["status"] = "error_unknown_target_class"
                    logs.append(rec)
                    continue
                target_label = CLASS_TO_LABEL[target_class]
            dst_rel = relabel_rel_path(rel_old, target_class)
            dst_abs = unique_path(train_root / dst_rel)
        else:
            dst_rel = (Path("_manual_quarantine_batches") / batch_name / "trash" / rel_old).as_posix()
            dst_abs = unique_path(dataset_root / dst_rel)

        rec["image_id_old"] = rel_old
        rec["dst_abs"] = str(dst_abs)
        rec["target_class"] = target_class if action == "relabel" else ""
        rec["target_label"] = target_label if action == "relabel" else ""

        # Update train.csv
        train_mask = train_df["image_id"].astype(str) == rel_old
        if not train_mask.any():
            rec["status"] = "missing_in_train_csv"
            logs.append(rec)
            continue

        # Update queue manifest
        man_mask = manifest_df["image_id"].astype(str) == rel_old
        if not man_mask.any():
            rec["manifest_row_found"] = False
        else:
            rec["manifest_row_found"] = True

        if action == "relabel":
            new_rel_for_train = rel_from_root(dst_abs, train_root)
            rec["image_id_new"] = new_rel_for_train
            train_df.loc[train_mask, "image_id"] = new_rel_for_train
            train_df.loc[train_mask, "label"] = int(target_label)

            if man_mask.any():
                manifest_df.loc[man_mask, "image_id"] = new_rel_for_train
                if "label" in manifest_df.columns:
                    manifest_df.loc[man_mask, "label"] = int(target_label)
                if "class_name" in manifest_df.columns:
                    manifest_df.loc[man_mask, "class_name"] = target_class
                if "plu" in manifest_df.columns:
                    manifest_df.loc[man_mask, "plu"] = Path(new_rel_for_train).parts[1]
                if "filename" in manifest_df.columns:
                    manifest_df.loc[man_mask, "filename"] = Path(new_rel_for_train).name
                if "abs_path" in manifest_df.columns:
                    manifest_df.loc[man_mask, "abs_path"] = str(dst_abs)

            if run_apply:
                dst_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(abs_path), str(dst_abs))
                moved_files += 1
            relabeled += 1
            rec["status"] = "moved_relabel" if run_apply else "dry_run_relabel"
        else:
            rec["image_id_new"] = str(Path(dst_rel).as_posix())
            before_train_len = len(train_df)
            train_df = train_df.loc[~train_mask].copy()
            train_removed += before_train_len - len(train_df)

            if man_mask.any():
                before_man_len = len(manifest_df)
                manifest_df = manifest_df.loc[~man_mask].copy()
                manifest_removed += before_man_len - len(manifest_df)

            if run_apply:
                dst_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(abs_path), str(dst_abs))
                moved_files += 1
            trashed += 1
            rec["status"] = "moved_trash" if run_apply else "dry_run_trash"

        logs.append(rec)

    # Validation on dataframes (before writing)
    train_df["image_id"] = train_df["image_id"].astype(str)
    if train_df["image_id"].duplicated().any():
        dups = train_df.loc[train_df["image_id"].duplicated(keep=False), "image_id"].astype(str).tolist()[:20]
        raise RuntimeError(f"Duplicate image_id in train.csv after apply: sample={dups}")

    # Label-folder consistency for train.csv rows
    invalid_folder_labels = 0
    for r in train_df.itertuples(index=False):
        image_id = str(r.image_id)
        label = int(r.label)
        cls_from_path = Path(image_id).parts[0]
        if CLASS_TO_LABEL.get(cls_from_path) != label:
            invalid_folder_labels += 1

    # Save dry-run/apply outputs
    log_csv = out_dir / "apply_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    train_after_csv = out_dir / "train_after.csv"
    manifest_after_csv = out_dir / "manual_review_queue_manifest_after.csv"
    train_df.to_csv(train_after_csv, index=False)
    manifest_df.to_csv(manifest_after_csv, index=False)

    backups = {}
    if run_apply:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_backup = train_csv.with_name(f"train.pre_{batch_name}_{ts_tag}.backup.csv")
        manifest_backup = queue_manifest_csv.with_name(
            f"manifest.pre_{batch_name}_{ts_tag}.backup.csv"
        )
        shutil.copy2(train_csv, train_backup)
        shutil.copy2(queue_manifest_csv, manifest_backup)
        train_df.to_csv(train_csv, index=False)
        manifest_df.to_csv(queue_manifest_csv, index=False)
        backups = {"train_csv": str(train_backup), "queue_manifest_csv": str(manifest_backup)}

        # Update queue summary to stay consistent with current dataset snapshot.
        if queue_summary_json.exists():
            try:
                q = json.loads(queue_summary_json.read_text(encoding="utf-8"))
            except Exception:
                q = {}
            q.update(
                {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "dataset_root": str(dataset_root),
                    "manifest": str(queue_manifest_csv),
                    "decisions": str(queue_root / "decisions.csv"),
                    "num_images": int(len(train_df)),
                    "classes_in_order": sorted(
                        [p.name for p in train_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
                    ),
                    "last_apply_batch": batch_name,
                }
            )
            queue_summary_json.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")

    # Post-apply physical validation if apply=true
    disk_active_images = None
    missing_files_after = None
    if run_apply:
        active_files = scan_active_image_files(train_root)
        disk_active_images = len(active_files)
        missing = []
        for rel in train_df["image_id"].astype(str).tolist():
            if not (train_root / rel).exists():
                missing.append(rel)
                if len(missing) >= 20:
                    break
        missing_files_after = missing

    summary = {
        "dataset_root": str(dataset_root),
        "train_root": str(train_root),
        "actions_csv": str(actions_csv),
        "apply_mode": run_apply,
        "batch_name": batch_name,
        "counts": {
            "actions_total": int(len(df_actions)),
            "relabel_actions": int(action_counts.get("relabel", 0)),
            "trash_actions": int(action_counts.get("trash", 0)),
            "relabeled_processed": int(relabeled),
            "trashed_processed": int(trashed),
            "moved_files": int(moved_files),
            "skipped_missing": int(skipped_missing),
            "train_rows_before": int(train_before),
            "train_rows_after": int(len(train_df)),
            "manifest_rows_before": int(manifest_before),
            "manifest_rows_after": int(len(manifest_df)),
            "train_removed_rows": int(train_removed),
            "manifest_removed_rows": int(manifest_removed),
        },
        "validation": {
            "duplicate_image_id_in_train_csv": False,
            "invalid_folder_label_rows_in_train_csv": int(invalid_folder_labels),
            "disk_active_images_after_apply": int(disk_active_images) if disk_active_images is not None else None,
            "train_csv_matches_disk_active_images": (int(disk_active_images) == int(len(train_df))) if disk_active_images is not None else None,
            "missing_train_csv_files_after_apply_sample": missing_files_after,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "apply_log_csv": str(log_csv),
            "train_after_csv": str(train_after_csv),
            "manual_review_queue_manifest_after_csv": str(manifest_after_csv),
        },
        "backups": backups,
        "quarantine_root": str(quarantine_root),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
