#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


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


@dataclass
class ReplayResult:
    per_class_history: Dict[str, List[dict]]
    event_counts: Counter
    anomalies: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export final actions/manifests from Dataset Tinder review session (replaying undo history)."
    )
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument(
        "--train-root",
        type=Path,
        default=Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/train/train"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <session-dir>/export_final_actions",
    )
    return p.parse_args()


def scan_dataset_counts(train_root: Path) -> Dict[str, int]:
    if not train_root.is_dir():
        raise FileNotFoundError(f"train_root not found: {train_root}")
    out: Dict[str, int] = {}
    for class_dir in sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        n = sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        out[class_dir.name] = n
    return out


def replay_session(all_decisions_jsonl: Path) -> ReplayResult:
    per_class_history: Dict[str, List[dict]] = defaultdict(list)
    event_counts: Counter = Counter()
    anomalies: List[str] = []

    with all_decisions_jsonl.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            et = str(rec.get("event_type", "decision"))
            event_counts[et] += 1
            class_name = str(rec.get("class_name", "")).strip()
            if not class_name:
                anomalies.append(f"line {lineno}: missing class_name")
                continue
            if et == "decision":
                per_class_history[class_name].append(rec)
            elif et == "undo":
                if not per_class_history[class_name]:
                    anomalies.append(f"line {lineno}: undo on empty history for class={class_name}")
                    continue
                popped = per_class_history[class_name].pop()
                if (
                    rec.get("image_key")
                    and popped.get("image_key")
                    and str(rec["image_key"]) != str(popped["image_key"])
                ):
                    anomalies.append(
                        f"line {lineno}: undo mismatch for class={class_name}: "
                        f"{rec.get('image_key')} != {popped.get('image_key')}"
                    )
            else:
                anomalies.append(f"line {lineno}: unknown event_type={et}")
    return ReplayResult(per_class_history=per_class_history, event_counts=event_counts, anomalies=anomalies)


def build_review_manifest_rows(replay: ReplayResult) -> List[dict]:
    rows: List[dict] = []
    for class_name, stack in replay.per_class_history.items():
        for rec in stack:
            decision = str(rec.get("decision", "")).strip()
            target_class = rec.get("target_class")
            if isinstance(target_class, str):
                target_class = target_class.strip() or None
            action = "keep" if decision == "ok" else ("relabel" if decision == "other_class" else "trash")
            row = {
                "session_id": rec.get("session_id"),
                "timestamp": rec.get("timestamp"),
                "source_class": class_name,
                "final_decision": decision,
                "action": action,
                "target_class": target_class,
                "target_label": CLASS_TO_LABEL.get(target_class) if target_class else None,
                "image_key": rec.get("image_key"),
                "image_index": rec.get("image_index"),
                "relative_path": rec.get("relative_path"),
                "relative_in_class": rec.get("relative_in_class"),
                "filename": rec.get("filename"),
                "absolute_path": rec.get("absolute_path"),
            }
            rows.append(row)
    rows.sort(key=lambda r: (str(r["source_class"]), int(r["image_index"])))
    return rows


def build_actions_rows(review_manifest_rows: List[dict]) -> List[dict]:
    actions: List[dict] = []
    for r in review_manifest_rows:
        action = str(r["action"])
        if action not in {"trash", "relabel"}:
            continue
        target_class = r.get("target_class")
        target_label = r.get("target_label")
        actions.append(
            {
                "abs_path": r.get("absolute_path"),
                "action": action,
                "comment_raw": f"tinder_session:{r.get('session_id')} source_class:{r.get('source_class')}",
                "target_class": target_class if action == "relabel" else "",
                "target_label": target_label if action == "relabel" else "",
                "source_class": r.get("source_class"),
                "relative_path": r.get("relative_path"),
                "image_key": r.get("image_key"),
            }
        )
    actions.sort(key=lambda r: (str(r["action"]), str(r.get("relative_path") or "")))
    return actions


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Preserve schema even when empty.
        fieldnames = ["empty"]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_per_class_stats(review_manifest_rows: List[dict], dataset_counts: Dict[str, int]) -> List[dict]:
    by_class = defaultdict(lambda: Counter())
    for r in review_manifest_rows:
        c = str(r["source_class"])
        by_class[c]["reviewed"] += 1
        by_class[c][str(r["final_decision"])] += 1
    rows: List[dict] = []
    for class_name in sorted(dataset_counts.keys()):
        cnt = by_class[class_name]
        total = int(dataset_counts[class_name])
        reviewed = int(cnt.get("reviewed", 0))
        rows.append(
            {
                "class_name": class_name,
                "dataset_total": total,
                "reviewed": reviewed,
                "ok": int(cnt.get("ok", 0)),
                "trash": int(cnt.get("trash", 0)),
                "other_class": int(cnt.get("other_class", 0)),
                "complete": reviewed == total,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    session_dir = args.session_dir.expanduser().resolve()
    train_root = args.train_root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else session_dir / "export_final_actions"
    )

    if not session_dir.is_dir():
        raise FileNotFoundError(f"session_dir not found: {session_dir}")
    all_jsonl = session_dir / "all_decisions.jsonl"
    if not all_jsonl.is_file():
        raise FileNotFoundError(f"Missing {all_jsonl}")

    dataset_counts = scan_dataset_counts(train_root)
    replay = replay_session(all_jsonl)
    review_manifest_rows = build_review_manifest_rows(replay)
    actions_rows = build_actions_rows(review_manifest_rows)
    per_class_rows = build_per_class_stats(review_manifest_rows, dataset_counts)

    total_dataset = sum(dataset_counts.values())
    total_reviewed = len(review_manifest_rows)
    complete_all = total_reviewed == total_dataset and all(bool(r["complete"]) for r in per_class_rows)

    anomalies_path = out_dir / "anomalies.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    anomalies_path.write_text(
        ("\n".join(replay.anomalies) + ("\n" if replay.anomalies else "")),
        encoding="utf-8",
    )

    review_manifest_csv = out_dir / "review_manifest_final.csv"
    actions_csv = out_dir / "actions_for_apply_manual_actions.csv"
    per_class_stats_csv = out_dir / "per_class_stats.csv"
    trash_abs_list = out_dir / "trash_abs_paths.txt"
    relabel_pairs_csv = out_dir / "relabel_pairs.csv"

    write_csv(review_manifest_csv, review_manifest_rows)
    write_csv(actions_csv, actions_rows)
    write_csv(per_class_stats_csv, per_class_rows)

    trash_rows = [r for r in review_manifest_rows if r["action"] == "trash"]
    relabel_rows = [r for r in review_manifest_rows if r["action"] == "relabel"]

    trash_abs_list.write_text(
        "".join(f"{r['absolute_path']}\n" for r in trash_rows),
        encoding="utf-8",
    )
    write_csv(
        relabel_pairs_csv,
        [
            {
                "absolute_path": r["absolute_path"],
                "relative_path": r["relative_path"],
                "source_class": r["source_class"],
                "target_class": r["target_class"],
                "target_label": r["target_label"],
            }
            for r in relabel_rows
        ],
    )

    counts_final = Counter(str(r["final_decision"]) for r in review_manifest_rows)
    counts_actions = Counter(str(r["action"]) for r in review_manifest_rows)

    summary = {
        "session_dir": str(session_dir),
        "train_root": str(train_root),
        "all_decisions_jsonl": str(all_jsonl),
        "event_counts": dict(replay.event_counts),
        "anomalies_count": len(replay.anomalies),
        "dataset_images_total": total_dataset,
        "review_manifest_rows": total_reviewed,
        "complete_all_classes": complete_all,
        "final_decision_counts": dict(counts_final),
        "action_counts": dict(counts_actions),
        "export_files": {
            "review_manifest_final_csv": str(review_manifest_csv),
            "actions_for_apply_manual_actions_csv": str(actions_csv),
            "per_class_stats_csv": str(per_class_stats_csv),
            "trash_abs_paths_txt": str(trash_abs_list),
            "relabel_pairs_csv": str(relabel_pairs_csv),
            "anomalies_txt": str(anomalies_path),
        },
        "note": (
            "actions_for_apply_manual_actions.csv is compatible with scripts/apply_manual_actions_csv.py "
            "(columns: abs_path, action, target_class, target_label, comment_raw, ...)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
