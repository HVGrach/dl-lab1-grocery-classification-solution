#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def norm_rel_image_id(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    s = s.lstrip("/")
    for prefix in ("train/train/", "train/"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s


def review_root(dataset_root: Path) -> Path:
    return dataset_root / "cleaning" / "manual_review_queue"


def manifest_path(dataset_root: Path) -> Path:
    return review_root(dataset_root) / "manifest.csv"


def decisions_path(dataset_root: Path) -> Path:
    return review_root(dataset_root) / "decisions.csv"


def summary_path(dataset_root: Path) -> Path:
    return review_root(dataset_root) / "queue_summary.json"


@dataclass
class QueuePaths:
    dataset_root: Path
    train_csv: Path
    train_dir: Path
    root: Path
    manifest: Path
    decisions: Path
    summary: Path


def get_paths(dataset_root: str) -> QueuePaths:
    root = Path(dataset_root).expanduser().resolve()
    qroot = review_root(root)
    return QueuePaths(
        dataset_root=root,
        train_csv=root / "train.csv",
        train_dir=root / "train" / "train",
        root=qroot,
        manifest=manifest_path(root),
        decisions=decisions_path(root),
        summary=summary_path(root),
    )


def load_train_df(paths: QueuePaths) -> pd.DataFrame:
    df = pd.read_csv(paths.train_csv)
    if not {"image_id", "label"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected train.csv columns: {list(df.columns)}")
    df["image_id"] = df["image_id"].astype(str).map(norm_rel_image_id)
    df["class_name"] = df["image_id"].str.split("/").str[0]
    df["plu"] = df["image_id"].str.split("/").str[1]
    df["filename"] = df["image_id"].str.split("/").str[-1]
    return df


def build_label_maps(df: pd.DataFrame) -> tuple[Dict[int, str], Dict[str, int]]:
    m1 = (
        df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values(["label", "class_name"])
        .groupby("label")["class_name"]
        .agg(list)
        .to_dict()
    )
    bad = {k: v for k, v in m1.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"Ambiguous label->class mapping: {bad}")
    label_to_class = {int(k): v[0] for k, v in m1.items()}
    class_to_label = {v: k for k, v in label_to_class.items()}
    return label_to_class, class_to_label


def cmd_init(args: argparse.Namespace) -> None:
    paths = get_paths(args.dataset_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    df = load_train_df(paths)
    label_to_class, _ = build_label_maps(df)
    class_order = [label_to_class[i] for i in sorted(label_to_class)]
    class_rank = {c: i for i, c in enumerate(class_order)}
    df["class_rank"] = df["class_name"].map(class_rank)
    df["abs_path"] = df["image_id"].map(lambda s: str((paths.train_dir / s).resolve()))
    df["exists"] = df["abs_path"].map(lambda s: Path(s).exists())
    missing = int((~df["exists"]).sum())
    if missing:
        raise RuntimeError(f"Manifest init aborted: {missing} train images missing on disk")
    df = df.sort_values(["class_rank", "class_name", "plu", "filename"]).reset_index(drop=True)
    df.insert(0, "queue_idx", range(1, len(df) + 1))
    out = df[["queue_idx", "image_id", "label", "class_name", "plu", "filename", "abs_path"]].copy()
    out.to_csv(paths.manifest, index=False)

    if not paths.decisions.exists():
        with paths.decisions.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts",
                    "queue_idx",
                    "image_id",
                    "action",
                    "target_class",
                    "note",
                ]
            )

    payload = {
        "created_at": now_iso(),
        "dataset_root": str(paths.dataset_root),
        "manifest": str(paths.manifest),
        "decisions": str(paths.decisions),
        "num_images": int(len(out)),
        "classes_in_order": class_order,
        "missing_on_init": missing,
    }
    paths.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_manifest(paths: QueuePaths) -> pd.DataFrame:
    if not paths.manifest.exists():
        raise RuntimeError(f"Manifest not found: {paths.manifest}. Run init first.")
    df = pd.read_csv(paths.manifest)
    return df


def load_decisions(paths: QueuePaths) -> pd.DataFrame:
    if not paths.decisions.exists():
        return pd.DataFrame(columns=["ts", "queue_idx", "image_id", "action", "target_class", "note"])
    df = pd.read_csv(paths.decisions)
    if df.empty:
        return df
    df["image_id"] = df["image_id"].astype(str).map(norm_rel_image_id)
    if "queue_idx" in df.columns:
        # nullable int support
        try:
            df["queue_idx"] = pd.to_numeric(df["queue_idx"], errors="coerce").astype("Int64")
        except Exception:
            pass
    return df


def latest_decisions_map(dec: pd.DataFrame) -> pd.DataFrame:
    if dec.empty:
        return dec
    # keep last decision per image
    return dec.drop_duplicates(subset=["image_id"], keep="last").copy()


def pending_manifest(
    man: pd.DataFrame, dec_latest: pd.DataFrame, class_name: Optional[str] = None
) -> pd.DataFrame:
    done = set(dec_latest["image_id"].tolist()) if not dec_latest.empty else set()
    out = man[~man["image_id"].isin(done)].copy()
    if class_name:
        out = out[out["class_name"] == class_name].copy()
    return out


def cmd_next(args: argparse.Namespace) -> None:
    paths = get_paths(args.dataset_root)
    man = load_manifest(paths)
    dec = load_decisions(paths)
    dec_latest = latest_decisions_map(dec)

    p = pending_manifest(man, dec_latest, args.class_name)
    if p.empty:
        print(json.dumps({"status": "empty", "class_name": args.class_name or None}, ensure_ascii=False))
        return
    row = p.iloc[int(args.offset)]
    payload = {
        "status": "ok",
        "queue_idx": int(row["queue_idx"]),
        "image_id": str(row["image_id"]),
        "label": int(row["label"]),
        "class_name": str(row["class_name"]),
        "plu": str(row["plu"]),
        "filename": str(row["filename"]),
        "abs_path": str(row["abs_path"]),
        "pending_total": int(len(p)),
        "pending_in_class": int(len(p)) if args.class_name else None,
    }
    if not args.class_name:
        payload["global_pending"] = int(len(p))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_decide(args: argparse.Namespace) -> None:
    paths = get_paths(args.dataset_root)
    man = load_manifest(paths)
    dec = load_decisions(paths)

    action = args.action.lower()
    if action not in {"keep", "trash", "relabel", "skip"}:
        raise RuntimeError("action must be one of: keep, trash, relabel, skip")
    target = args.target_class or ""
    if action == "relabel" and not target:
        raise RuntimeError("relabel requires --target-class")
    if action != "relabel" and target:
        raise RuntimeError("--target-class is only allowed for relabel")

    image_id = norm_rel_image_id(args.image_id)
    hit = man[man["image_id"] == image_id]
    if hit.empty:
        raise RuntimeError(f"image_id not found in manifest: {image_id}")
    qidx = int(hit.iloc[0]["queue_idx"])

    row = {
        "ts": now_iso(),
        "queue_idx": qidx,
        "image_id": image_id,
        "action": action,
        "target_class": target,
        "note": args.note or "",
    }
    write_header = not paths.decisions.exists()
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.decisions.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "queue_idx", "image_id", "action", "target_class", "note"])
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(json.dumps({"status": "ok", **row}, ensure_ascii=False, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    paths = get_paths(args.dataset_root)
    man = load_manifest(paths)
    dec = load_decisions(paths)
    dec_latest = latest_decisions_map(dec)
    p = pending_manifest(man, dec_latest, args.class_name)

    out = {
        "dataset_root": str(paths.dataset_root),
        "manifest": str(paths.manifest),
        "decisions": str(paths.decisions),
        "total_images": int(len(man)),
        "decisions_total_rows": int(len(dec)),
        "decided_unique_images": int(len(dec_latest)) if not dec_latest.empty else 0,
        "pending_images": int(len(man) - (len(dec_latest) if not dec_latest.empty else 0)),
        "pending_for_filter": int(len(p)),
        "class_name_filter": args.class_name or None,
    }
    if not dec_latest.empty:
        out["latest_action_counts"] = dec_latest["action"].value_counts().to_dict()
        by_class = (
            man.merge(dec_latest[["image_id", "action"]], on="image_id", how="left")
            .fillna({"action": "pending"})
            .groupby(["class_name", "action"])
            .size()
            .reset_index(name="n")
        )
        if args.class_name:
            by_class = by_class[by_class["class_name"] == args.class_name]
        out["by_class_action"] = by_class.to_dict(orient="records")
    else:
        out["latest_action_counts"] = {}
        out["by_class_action"] = []
    print(json.dumps(out, ensure_ascii=False, indent=2))


def apply_actions_to_dataset(paths: QueuePaths, dec_latest: pd.DataFrame) -> dict:
    train_df = pd.read_csv(paths.train_csv)
    train_df["image_id"] = train_df["image_id"].astype(str).map(norm_rel_image_id)
    # label maps from current train.csv
    tmp = train_df.copy()
    tmp["class_name"] = tmp["image_id"].str.split("/").str[0]
    _, class_to_label = build_label_maps(tmp)

    train_dir = paths.train_dir
    removed_root = paths.dataset_root / "_removed_train" / "manual_review_queue"
    removed_root.mkdir(parents=True, exist_ok=True)

    status_rows = []
    for _, d in dec_latest.iterrows():
        image_id = norm_rel_image_id(d["image_id"])
        action = str(d["action"]).strip().lower()
        target_class = str(d.get("target_class", "") or "").strip()
        src = train_dir / image_id

        if action in {"keep", "skip", "pending"}:
            status_rows.append({"image_id": image_id, "action": action, "status": "ignored"})
            continue

        if image_id not in set(train_df["image_id"]):
            status_rows.append({"image_id": image_id, "action": action, "status": "missing_in_csv"})
            continue
        if not src.exists():
            status_rows.append({"image_id": image_id, "action": action, "status": "missing_on_disk"})
            continue

        if action == "trash":
            dst = removed_root / "trash" / image_id
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = dst.with_name(dst.stem + "__dup" + dst.suffix)
            src.rename(dst)
            train_df = train_df[train_df["image_id"] != image_id]
            status_rows.append({"image_id": image_id, "action": action, "status": "ok", "dst": str(dst)})
            continue

        if action == "relabel":
            if not target_class or target_class not in class_to_label:
                status_rows.append({"image_id": image_id, "action": action, "status": "invalid_target", "target_class": target_class})
                continue
            parts = image_id.split("/")
            if len(parts) < 3:
                status_rows.append({"image_id": image_id, "action": action, "status": "invalid_path"})
                continue
            plu = parts[1]
            fname = parts[-1]
            new_rel = f"{target_class}/{plu}/{fname}"
            dst = train_dir / new_rel
            if dst.exists():
                status_rows.append({"image_id": image_id, "action": action, "status": "collision", "target_class": target_class, "dst": str(dst)})
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            mask = train_df["image_id"] == image_id
            train_df.loc[mask, "image_id"] = new_rel
            train_df.loc[mask, "label"] = int(class_to_label[target_class])
            status_rows.append({"image_id": image_id, "action": action, "status": "ok", "target_class": target_class, "dst": str(dst)})
            continue

        status_rows.append({"image_id": image_id, "action": action, "status": "unknown_action"})

    before = len(pd.read_csv(paths.train_csv))
    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    train_df.to_csv(paths.train_csv, index=False)
    (paths.dataset_root / "cleaning").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(paths.dataset_root / "cleaning" / "train_clean_strict.csv", index=False)
    train_df.to_csv(paths.dataset_root / "cleaning" / "train_clean_manual_review.csv", index=False)
    after = len(train_df)

    status_df = pd.DataFrame(status_rows)
    apply_dir = paths.root / "applied"
    apply_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_csv = apply_dir / f"apply_status_{ts}.csv"
    status_df.to_csv(status_csv, index=False)

    summary = {
        "applied_at": now_iso(),
        "statuses": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
        "actions": status_df["action"].value_counts().to_dict() if not status_df.empty else {},
        "rows_before_train_csv": int(before),
        "rows_after_train_csv": int(after),
        "delta_rows_train_csv": int(after - before),
        "status_csv": str(status_csv),
        "train_csv": str(paths.train_csv),
        "train_clean_strict_csv": str(paths.dataset_root / "cleaning" / "train_clean_strict.csv"),
    }
    summary_file = apply_dir / f"apply_summary_{ts}.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_file)
    return summary


def cmd_apply(args: argparse.Namespace) -> None:
    paths = get_paths(args.dataset_root)
    man = load_manifest(paths)
    dec = load_decisions(paths)
    if dec.empty:
        print(json.dumps({"status": "no_decisions"}, ensure_ascii=False))
        return
    dec_latest = latest_decisions_map(dec)
    # Join manifest to keep only current manifest rows, preserving order relevance.
    dec_latest = man[["image_id"]].merge(dec_latest, on="image_id", how="inner")
    if args.only_class:
        # apply subset only
        man_subset = man[man["class_name"] == args.only_class][["image_id"]]
        dec_latest = man_subset.merge(dec_latest, on="image_id", how="inner")
    if dec_latest.empty:
        print(json.dumps({"status": "empty_after_filter"}, ensure_ascii=False))
        return
    summary = apply_actions_to_dataset(paths, dec_latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manual review queue for dataset cleaning.")
    p.add_argument("--dataset-root", required=True, help="Dataset root containing train.csv and train/train")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create manifest and decisions file")
    p_init.set_defaults(func=cmd_init)

    p_next = sub.add_parser("next", help="Return next undecided image as JSON")
    p_next.add_argument("--class-name", default="", help="Restrict to a class name")
    p_next.add_argument("--offset", type=int, default=0, help="Offset among pending items")
    p_next.set_defaults(func=cmd_next)

    p_dec = sub.add_parser("decide", help="Append a decision for an image")
    p_dec.add_argument("--image-id", required=True, help="Relative image path like Класс/plu/file.jpg")
    p_dec.add_argument("--action", required=True, choices=["keep", "trash", "relabel", "skip"])
    p_dec.add_argument("--target-class", default="", help="Required for relabel")
    p_dec.add_argument("--note", default="", help="Optional note")
    p_dec.set_defaults(func=cmd_decide)

    p_stats = sub.add_parser("stats", help="Queue and decisions statistics")
    p_stats.add_argument("--class-name", default="", help="Optional class filter")
    p_stats.set_defaults(func=cmd_stats)

    p_apply = sub.add_parser("apply", help="Apply current decisions to dataset (trash/relabel only)")
    p_apply.add_argument("--only-class", default="", help="Apply only decisions for one class")
    p_apply.set_defaults(func=cmd_apply)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "class_name") and args.class_name == "":
        args.class_name = None
    if hasattr(args, "only_class") and args.only_class == "":
        args.only_class = None
    args.func(args)


if __name__ == "__main__":
    main()
