#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply manual actions from CSV to dataset files and cleaning CSVs.")
    p.add_argument("--actions-csv", type=Path, required=True, help="CSV with columns: abs_path, action, comment_raw")
    p.add_argument(
        "--base",
        type=Path,
        default=Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped"),
        help="Dataset base dir",
    )
    p.add_argument(
        "--batch-name",
        type=str,
        default=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Batch identifier",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files and update CSVs. Without this flag only dry-run summary is produced.",
    )
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


def rel_from_train_root(abs_path: Path, train_root: Path) -> str:
    return str(abs_path.relative_to(train_root)).replace("\\", "/")


def relabel_rel_path(old_rel: str, target_class: str) -> str:
    old = Path(old_rel)
    if len(old.parts) < 3:
        raise ValueError(f"Unexpected image_id format: {old_rel}")
    plu = old.parts[1]
    basename = old.parts[-1]
    return str(Path(target_class) / plu / basename)


def main() -> None:
    args = parse_args()
    actions_csv: Path = args.actions_csv
    base: Path = args.base
    batch_name = args.batch_name
    run_apply = bool(args.apply)

    train_root = base / "train" / "train"
    cleaning_dir = base / "cleaning"
    strict_csv = cleaning_dir / "train_clean_strict.csv"
    manifest_csv = cleaning_dir / "train_clean_manifest.csv"

    out_dir = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review") / batch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_actions = pd.read_csv(actions_csv)
    need_cols = {"abs_path", "action"}
    miss = need_cols - set(df_actions.columns)
    if miss:
        raise ValueError(f"actions csv missing columns: {sorted(miss)}")

    strict_df = pd.read_csv(strict_csv)
    manifest_df = pd.read_csv(manifest_csv)

    # Keep manifest plu as string to avoid dtype issues
    if "plu" in manifest_df.columns:
        manifest_df["plu"] = manifest_df["plu"].astype(str)

    strict_before = len(strict_df)
    manifest_before = len(manifest_df)

    logs: List[Dict[str, object]] = []
    skipped_missing = 0
    moved = 0
    removed_from_strict = 0

    for row in df_actions.itertuples(index=False):
        abs_path = Path(str(row.abs_path))
        action = str(row.action)
        comment = str(getattr(row, "comment_raw", "")) if hasattr(row, "comment_raw") else ""

        rec: Dict[str, object] = {
            "abs_path": str(abs_path),
            "action": action,
            "comment_raw": comment,
            "exists_before": abs_path.exists(),
            "status": "",
        }

        if not abs_path.exists():
            rec["status"] = "missing_source"
            logs.append(rec)
            skipped_missing += 1
            continue

        try:
            rel_old = rel_from_train_root(abs_path, train_root)
        except Exception:
            rec["status"] = "outside_train_root"
            logs.append(rec)
            continue

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
                target_label = int(v)

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
        else:
            dst_rel = str(Path("_manual_quarantine_batches") / batch_name / action / rel_old)

        dst_abs = train_root / dst_rel
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        dst_abs = unique_path(dst_abs)

        rec["image_id_old"] = rel_old
        rec["image_id_new"] = rel_from_train_root(dst_abs, train_root)
        rec["target_class"] = target_class
        rec["target_label"] = target_label

        if run_apply:
            shutil.move(str(abs_path), str(dst_abs))
            moved += 1
            rec["status"] = "moved"
        else:
            rec["status"] = "dry_run"

        if action == "relabel":
            strict_mask = strict_df["image_id"] == rel_old
            rec["removed_from_strict"] = 0
            if strict_mask.any():
                strict_df.loc[strict_mask, "image_id"] = rec["image_id_new"]
                strict_df.loc[strict_mask, "label"] = target_label
            else:
                # If missing in strict, append as active training sample
                strict_df = pd.concat(
                    [strict_df, pd.DataFrame([{"image_id": rec["image_id_new"], "label": target_label}])],
                    ignore_index=True,
                )
        else:
            # Update strict: remove this image_id
            cnt_before = len(strict_df)
            strict_df = strict_df[strict_df["image_id"] != rel_old].copy()
            cnt_after = len(strict_df)
            removed = cnt_before - cnt_after
            removed_from_strict += removed
            rec["removed_from_strict"] = removed

        # Update manifest
        m = manifest_df["image_id"] == rel_old
        if m.any():
            manifest_df.loc[m, "image_id"] = rec["image_id_new"]
            manifest_df.loc[m, "plu"] = Path(rec["image_id_new"]).parts[1]
            manifest_df.loc[m, "basename"] = Path(rec["image_id_new"]).name
            if action == "relabel":
                manifest_df.loc[m, "status"] = "keep"
                manifest_df.loc[m, "label"] = target_label
                manifest_df.loc[m, "class_name"] = target_class
            else:
                manifest_df.loc[m, "status"] = "drop"
                reason_col = "drop_reasons"
                token = f"manual_batch:{batch_name}:{action}"
                if reason_col in manifest_df.columns:
                    cur = manifest_df.loc[m, reason_col].fillna("").astype(str).str.strip("|")
                    manifest_df.loc[m, reason_col] = cur.apply(
                        lambda s: token if not s else (s if token in s.split("|") else f"{s}|{token}")
                    )
        logs.append(rec)

    strict_after = len(strict_df)
    manifest_after = len(manifest_df)

    # Save outputs
    log_csv = out_dir / "apply_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)

    strict_new = out_dir / "train_clean_strict_after.csv"
    manifest_new = out_dir / "train_clean_manifest_after.csv"
    strict_df.to_csv(strict_new, index=False)
    manifest_df.to_csv(manifest_new, index=False)

    if run_apply:
        # Backups (once per batch)
        strict_backup = cleaning_dir / f"train_clean_strict.pre_{batch_name}.backup.csv"
        manifest_backup = cleaning_dir / f"train_clean_manifest.pre_{batch_name}.backup.csv"
        if not strict_backup.exists():
            shutil.copy2(strict_csv, strict_backup)
        if not manifest_backup.exists():
            shutil.copy2(manifest_csv, manifest_backup)
        strict_df.to_csv(strict_csv, index=False)
        manifest_df.to_csv(manifest_csv, index=False)

    summary = {
        "batch_name": batch_name,
        "apply_mode": run_apply,
        "actions_csv": str(actions_csv),
        "strict_before": strict_before,
        "strict_after": strict_after,
        "manifest_before": manifest_before,
        "manifest_after": manifest_after,
        "moved_files": moved,
        "removed_rows_from_strict": removed_from_strict,
        "skipped_missing": skipped_missing,
        "outputs": {
            "out_dir": str(out_dir),
            "apply_log_csv": str(log_csv),
            "strict_after_csv": str(strict_new),
            "manifest_after_csv": str(manifest_new),
            "strict_current": str(strict_csv),
            "manifest_current": str(manifest_csv),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
