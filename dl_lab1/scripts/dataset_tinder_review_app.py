#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode


DEFAULT_TRAIN_ROOT = Path(
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/train/train"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review"
)
DEFAULT_CONFIDENCE_CACHE_NAME = "top_new_dataset_ensemble_confidence"

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

TRASH_FIELDS = [
    "timestamp",
    "session_id",
    "source_class",
    "relative_path",
    "relative_in_class",
    "filename",
    "absolute_path",
]

WRONG_CLASS_FIELDS = [
    "timestamp",
    "session_id",
    "source_class",
    "target_class",
    "relative_path",
    "relative_in_class",
    "filename",
    "absolute_path",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_rel_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scan_class_names(train_root: Path) -> List[str]:
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    if not train_root.is_dir():
        raise NotADirectoryError(f"Train root is not a directory: {train_root}")
    class_names = [p.name for p in train_root.iterdir() if p.is_dir()]
    return sorted(class_names, key=lambda s: s.casefold())


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


@dataclass
class ImageItem:
    index: int
    abs_path: Path
    relative_path: str
    relative_in_class: str
    filename: str

    @property
    def image_key(self) -> str:
        return self.relative_path


@dataclass
class ClassReviewState:
    class_name: str
    images: List[ImageItem]
    position: int = 0
    counts: Dict[str, int] = field(
        default_factory=lambda: {"ok": 0, "trash": 0, "other_class": 0}
    )
    started_at: str = field(default_factory=now_iso)

    @property
    def total(self) -> int:
        return len(self.images)

    @property
    def reviewed(self) -> int:
        return self.position

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.position)

    @property
    def done(self) -> bool:
        return self.position >= self.total

    def current_item(self) -> Optional[ImageItem]:
        if self.done:
            return None
        return self.images[self.position]


@dataclass
class ReviewStore:
    train_root: Path
    output_root: Path
    session_id: str
    class_names: List[str]
    session_dir: Path
    trash_csv: Path
    wrong_class_csv: Path
    decisions_jsonl: Path
    confidence_csv: Optional[Path] = None
    confidence_meta_json: Optional[Path] = None
    lock: Lock = field(default_factory=Lock, repr=False)
    class_states: Dict[str, ClassReviewState] = field(default_factory=dict)
    decision_history_by_class: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    confidence_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    label_to_class: Dict[int, str] = field(default_factory=dict)
    class_to_label: Dict[str, int] = field(default_factory=dict)
    confidence_status: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "created_at": now_iso(),
            "session_id": self.session_id,
            "train_root": str(self.train_root),
            "output_root": str(self.output_root),
            "class_names": self.class_names,
        }
        (self.session_dir / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._rewrite_result_files_locked()
        self._load_confidence_cache()

    def assert_valid_class(self, class_name: str) -> None:
        if class_name not in self.class_names:
            raise ValueError(f"Unknown class: {class_name}")

    def _load_images_for_class(self, class_name: str) -> List[ImageItem]:
        class_dir = self.train_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        image_paths = [p for p in class_dir.rglob("*") if is_image_file(p)]
        image_paths.sort(key=lambda p: safe_rel_path(p, class_dir))
        items: List[ImageItem] = []
        for idx, p in enumerate(image_paths):
            items.append(
                ImageItem(
                    index=idx,
                    abs_path=p,
                    relative_path=safe_rel_path(p, self.train_root),
                    relative_in_class=safe_rel_path(p, class_dir),
                    filename=p.name,
                )
            )
        return items

    def get_or_create_state(self, class_name: str) -> ClassReviewState:
        self.assert_valid_class(class_name)
        state = self.class_states.get(class_name)
        if state is not None:
            return state
        images = self._load_images_for_class(class_name)
        state = ClassReviewState(class_name=class_name, images=images)
        self.class_states[class_name] = state
        return state

    def _decision_event(self, event_type: str, **payload: Any) -> None:
        append_jsonl(
            self.decisions_jsonl,
            {
                "timestamp": now_iso(),
                "session_id": self.session_id,
                "event_type": event_type,
                **payload,
            },
        )

    def _iter_active_decisions_locked(self):
        for class_name in self.class_names:
            for row in self.decision_history_by_class.get(class_name, []):
                yield row

    def _rewrite_result_files_locked(self) -> None:
        trash_rows: List[Dict[str, Any]] = []
        wrong_rows: List[Dict[str, Any]] = []
        for row in self._iter_active_decisions_locked():
            if row.get("decision") == "trash":
                trash_rows.append({k: row.get(k) for k in TRASH_FIELDS})
            elif row.get("decision") == "other_class":
                wrong_rows.append({k: row.get(k) for k in WRONG_CLASS_FIELDS})
        write_csv_rows(self.trash_csv, TRASH_FIELDS, trash_rows)
        write_csv_rows(self.wrong_class_csv, WRONG_CLASS_FIELDS, wrong_rows)

    def _load_confidence_cache(self) -> None:
        self.confidence_records = {}
        self.label_to_class = {}
        self.class_to_label = {}
        csv_path = self.confidence_csv
        meta_path = self.confidence_meta_json
        if not csv_path:
            self.confidence_status = {
                "enabled": False,
                "loaded": False,
                "reason": "confidence cache path not configured",
            }
            return
        if not csv_path.exists():
            self.confidence_status = {
                "enabled": True,
                "loaded": False,
                "reason": "confidence cache file not found",
                "csv_path": str(csv_path),
                "meta_path": str(meta_path) if meta_path else None,
            }
            return

        if meta_path and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                l2c = meta.get("label_to_class") or {}
                if isinstance(l2c, dict):
                    self.label_to_class = {int(k): str(v) for k, v in l2c.items()}
                c2l = meta.get("class_to_label") or {}
                if isinstance(c2l, dict):
                    self.class_to_label = {str(k): int(v) for k, v in c2l.items()}
            except Exception as exc:
                self.confidence_status = {
                    "enabled": True,
                    "loaded": False,
                    "reason": f"failed to parse confidence meta: {exc}",
                    "csv_path": str(csv_path),
                    "meta_path": str(meta_path),
                }
                return

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise RuntimeError("empty header")
                prob_cols = [c for c in reader.fieldnames if c.startswith("prob_")]
                prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split("_", 1)[1]))
                for row in reader:
                    rel = str(row.get("relative_path", "")).strip()
                    if not rel:
                        continue
                    probs: List[float] = []
                    for c in prob_cols_sorted:
                        v = row.get(c, "")
                        probs.append(float(v) if v not in ("", None) else float("nan"))
                    self.confidence_records[rel] = {
                        "source": (row.get("source") or "").strip() or None,
                        "probs": probs,
                    }
        except Exception as exc:
            self.confidence_records = {}
            self.confidence_status = {
                "enabled": True,
                "loaded": False,
                "reason": f"failed to parse confidence csv: {exc}",
                "csv_path": str(csv_path),
                "meta_path": str(meta_path) if meta_path else None,
            }
            return

        if not self.class_to_label and self.label_to_class:
            self.class_to_label = {v: k for k, v in self.label_to_class.items()}
        if not self.label_to_class and self.class_to_label:
            self.label_to_class = {v: k for k, v in self.class_to_label.items()}

        self.confidence_status = {
            "enabled": True,
            "loaded": True,
            "csv_path": str(csv_path),
            "meta_path": str(meta_path) if (meta_path and meta_path.exists()) else None,
            "records": len(self.confidence_records),
            "label_map_size": len(self.label_to_class),
        }

    def _confidence_payload_for_item(self, class_name: str, item: Optional[ImageItem]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        rec = self.confidence_records.get(item.relative_path)
        if rec is None:
            return {
                "available": False,
                "reason": "no prediction for this image",
            }
        probs = rec.get("probs") or []
        if not probs:
            return {"available": False, "reason": "empty probabilities"}

        current_label = self.class_to_label.get(class_name)
        rows: List[Dict[str, Any]] = []
        used = set()
        if current_label is not None and 0 <= current_label < len(probs):
            rows.append(
                {
                    "class_name": class_name,
                    "label": int(current_label),
                    "prob": float(probs[current_label]),
                    "is_current_class": True,
                }
            )
            used.add(int(current_label))

        ranked_idx = sorted(range(len(probs)), key=lambda i: float(probs[i]), reverse=True)
        for idx in ranked_idx:
            if idx in used:
                continue
            rows.append(
                {
                    "class_name": self.label_to_class.get(idx, f"label_{idx}"),
                    "label": int(idx),
                    "prob": float(probs[idx]),
                    "is_current_class": False,
                }
            )

        top1_idx = int(ranked_idx[0]) if ranked_idx else None
        top1_prob = float(probs[top1_idx]) if top1_idx is not None else None
        top1_class = self.label_to_class.get(top1_idx, f"label_{top1_idx}") if top1_idx is not None else None

        return {
            "available": True,
            "source": rec.get("source"),
            "top1": {
                "label": top1_idx,
                "class_name": top1_class,
                "prob": top1_prob,
            },
            "current_class_label": current_label,
            "current_class_prob": float(probs[current_label]) if current_label is not None and 0 <= current_label < len(probs) else None,
            "rows_top_display": rows[:6],
            "num_classes": len(probs),
        }

    def _item_payload(self, class_name: str, item: ImageItem) -> Dict[str, Any]:
        query = urlencode({"class_name": class_name, "index": item.index})
        return {
            "image_key": item.image_key,
            "image_index": item.index,
            "filename": item.filename,
            "relative_path": item.relative_path,
            "relative_in_class": item.relative_in_class,
            "image_url": f"/api/image?{query}",
        }

    def _state_payload(self, class_name: str) -> Dict[str, Any]:
        state = self.get_or_create_state(class_name)
        item = state.current_item()
        return {
            "session_id": self.session_id,
            "class_name": class_name,
            "class_names": self.class_names,
            "progress": {
                "total": state.total,
                "reviewed": state.reviewed,
                "remaining": state.remaining,
                "done": state.done,
                "percent": round((state.reviewed / state.total * 100.0), 2)
                if state.total
                else 100.0,
                "counts": dict(state.counts),
            },
            "item": self._item_payload(class_name, item) if item else None,
            "confidence": self._confidence_payload_for_item(class_name, item),
            "output_files": {
                "session_dir": str(self.session_dir),
                "trash_csv": str(self.trash_csv),
                "wrong_class_csv": str(self.wrong_class_csv),
                "decisions_jsonl": str(self.decisions_jsonl),
            },
            "confidence_cache": dict(self.confidence_status),
        }

    def classes_overview(self) -> Dict[str, Any]:
        with self.lock:
            overview: List[Dict[str, Any]] = []
            for class_name in self.class_names:
                state = self.class_states.get(class_name)
                if state is None:
                    overview.append(
                        {
                            "class_name": class_name,
                            "loaded": False,
                            "reviewed": 0,
                            "total": None,
                            "remaining": None,
                            "done": False,
                        }
                    )
                    continue
                overview.append(
                    {
                        "class_name": class_name,
                        "loaded": True,
                        "reviewed": state.reviewed,
                        "total": state.total,
                        "remaining": state.remaining,
                        "done": state.done,
                    }
                )
            session_dirs = sorted(
                [p for p in self.output_root.glob("session_*") if p.is_dir()],
                key=lambda p: p.name,
            )
            return {
                "session_id": self.session_id,
                "train_root": str(self.train_root),
                "output_root": str(self.output_root),
                "session_dir": str(self.session_dir),
                "classes": overview,
                "class_names": self.class_names,
                "output_files": {
                    "trash_csv": str(self.trash_csv),
                    "wrong_class_csv": str(self.wrong_class_csv),
                    "decisions_jsonl": str(self.decisions_jsonl),
                },
                "sessions": {
                    "all_session_dirs": [str(p) for p in session_dirs],
                    "count_total": len(session_dirs),
                    "count_previous": sum(1 for p in session_dirs if p.resolve() != self.session_dir.resolve()),
                },
                "confidence_cache": dict(self.confidence_status),
            }

    def read_state(self, class_name: str) -> Dict[str, Any]:
        with self.lock:
            return self._state_payload(class_name)

    def image_path_for(self, class_name: str, index: int) -> Path:
        with self.lock:
            state = self.get_or_create_state(class_name)
            if index < 0 or index >= len(state.images):
                raise IndexError(index)
            return state.images[index].abs_path

    def apply_decision(
        self,
        class_name: str,
        decision: str,
        image_key: str,
        target_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            state = self.get_or_create_state(class_name)
            item = state.current_item()
            if item is None:
                raise RuntimeError("Class review already completed")
            if image_key != item.image_key:
                raise ValueError("Image mismatch: stale UI state or duplicate submit")

            if decision not in {"ok", "trash", "other_class"}:
                raise ValueError(f"Unsupported decision: {decision}")

            if decision == "other_class":
                if not target_class:
                    raise ValueError("target_class is required for other_class")
                self.assert_valid_class(target_class)
                if target_class == class_name:
                    raise ValueError("target_class must be different from source class")
            else:
                target_class = None

            ts = now_iso()
            base_row = {
                "session_id": self.session_id,
                "source_class": class_name,
                "target_class": target_class,
                "relative_path": item.relative_path,
                "relative_in_class": item.relative_in_class,
                "filename": item.filename,
                "absolute_path": str(item.abs_path),
                "decision": decision,
                "image_index": item.index,
                "image_key": item.image_key,
                "timestamp": ts,
            }
            self._decision_event(
                "decision",
                class_name=class_name,
                decision=decision,
                target_class=target_class,
                image_key=item.image_key,
                image_index=item.index,
                relative_path=item.relative_path,
                relative_in_class=item.relative_in_class,
                filename=item.filename,
                absolute_path=str(item.abs_path),
            )
            self.decision_history_by_class.setdefault(class_name, []).append(dict(base_row))
            self._rewrite_result_files_locked()

            state.counts[decision] += 1
            state.position += 1
            return self._state_payload(class_name)

    def undo_last(self, class_name: str) -> Dict[str, Any]:
        with self.lock:
            state = self.get_or_create_state(class_name)
            history = self.decision_history_by_class.setdefault(class_name, [])
            if not history:
                raise RuntimeError("Nothing to undo for this class")
            if state.position <= 0:
                raise RuntimeError("Internal state error: position already at start")

            last = history.pop()
            decision = str(last.get("decision"))
            if decision not in {"ok", "trash", "other_class"}:
                raise RuntimeError(f"Cannot undo unknown decision: {decision}")

            state.position -= 1
            state.counts[decision] = max(0, state.counts.get(decision, 0) - 1)

            expected_item = state.current_item()
            if expected_item is None:
                raise RuntimeError("Internal state error after undo")
            if expected_item.image_key != last.get("image_key"):
                raise RuntimeError(
                    "Internal state mismatch during undo (history/order inconsistency)"
                )

            self._decision_event(
                "undo",
                class_name=class_name,
                reverted_decision=decision,
                target_class=last.get("target_class"),
                image_key=last.get("image_key"),
                image_index=last.get("image_index"),
                relative_path=last.get("relative_path"),
                relative_in_class=last.get("relative_in_class"),
                filename=last.get("filename"),
                absolute_path=last.get("absolute_path"),
            )
            self._rewrite_result_files_locked()
            return self._state_payload(class_name)

    def delete_previous_sessions(self) -> Dict[str, Any]:
        with self.lock:
            deleted: List[str] = []
            failed: List[Dict[str, str]] = []
            current_resolved = self.session_dir.resolve()
            for p in sorted(self.output_root.glob("session_*"), key=lambda x: x.name):
                if not p.is_dir():
                    continue
                try:
                    if p.resolve() == current_resolved:
                        continue
                    shutil.rmtree(p)
                    deleted.append(str(p))
                except Exception as exc:
                    failed.append({"path": str(p), "error": str(exc)})
            return {
                "deleted_count": len(deleted),
                "failed_count": len(failed),
                "deleted": deleted,
                "failed": failed,
                "kept_current_session": str(self.session_dir),
            }


def build_html() -> str:
    return """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Tinder Review</title>
  <style>
    :root {
      --bg1: #f7efe0;
      --bg2: #e8f0e6;
      --ink: #1d1f1d;
      --muted: #5f655f;
      --card: rgba(255,255,255,0.92);
      --line: rgba(0,0,0,0.08);
      --good: #2d7d46;
      --warn: #a23a12;
      --accent: #c2572b;
      --shadow: 0 20px 60px rgba(50, 40, 20, 0.15);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      min-height: 100dvh;
      overflow: hidden;
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1000px 500px at 10% -10%, rgba(245, 165, 77, 0.32), transparent 55%),
        radial-gradient(1200px 700px at 100% 0%, rgba(80, 150, 90, 0.20), transparent 60%),
        linear-gradient(145deg, var(--bg1), var(--bg2));
    }
    .page {
      display: grid;
      grid-template-columns: 288px minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
      height: 100vh;
      height: 100dvh;
    }
    .panel, .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
    }
    .panel {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      height: calc(100vh - 24px);
      height: calc(100dvh - 24px);
      overflow: auto;
    }
    .title {
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 0.2px;
      margin: 0;
    }
    .subtle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    select, button {
      font: inherit;
    }
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 10px;
      background: white;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 8px 12px;
      cursor: pointer;
      transition: transform .08s ease, filter .12s ease;
    }
    button:hover { filter: brightness(0.98); }
    button:active { transform: translateY(1px); }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    .btn-primary { background: var(--ink); color: white; }
    .btn-ghost { background: rgba(0,0,0,0.05); color: var(--ink); }
    .btn-good { background: rgba(45,125,70,0.12); color: var(--good); }
    .btn-bad { background: rgba(194,87,43,0.12); color: var(--accent); }
    .btn-danger-soft { background: rgba(194,87,43,0.10); color: #8a3412; }
    .info-box {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.8);
      border-radius: 14px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.3;
      word-break: break-word;
    }
    .class-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 6px;
    }
    .class-item {
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .class-item.active {
      border-color: rgba(194,87,43,0.45);
      box-shadow: inset 0 0 0 1px rgba(194,87,43,0.20);
      background: rgba(194,87,43,0.05);
    }
    .class-item.done {
      border-color: rgba(45,125,70,0.35);
      background: rgba(45,125,70,0.05);
    }
    .class-name {
      font-weight: 600;
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .class-meta {
      color: var(--muted);
      font-size: 11px;
      flex-shrink: 0;
    }
    .main {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto auto;
      gap: 10px;
      height: calc(100vh - 24px);
      height: calc(100dvh - 24px);
      min-height: 0;
      overflow: hidden;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
    }
    .topbar-side {
      display: grid;
      gap: 8px;
      justify-items: end;
      min-width: 0;
    }
    .metrics-row {
      display: grid;
      grid-template-columns: auto auto auto;
      gap: 8px;
      justify-content: end;
    }
    .topbar .metric {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 6px 8px;
      background: rgba(255,255,255,0.75);
      min-width: 92px;
      text-align: center;
    }
    .metric .label {
      display: block;
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }
    .metric .value {
      display: block;
      font-size: 14px;
      font-weight: 700;
      margin-top: 1px;
    }
    .progress-wrap {
      padding: 8px 10px;
      display: grid;
      gap: 6px;
    }
    .progress-bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(0,0,0,0.07);
      overflow: hidden;
      border: 1px solid rgba(0,0,0,0.05);
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #d77d33, #4f9b5a);
      transition: width .16s ease;
    }
    .progress-text {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .viewer {
      position: relative;
      padding: 8px;
      display: grid;
      place-items: center;
      min-height: 0;
      overflow: hidden;
    }
    .image-frame {
      width: 100%;
      height: 100%;
      min-height: 0;
      padding: 6px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.75), rgba(240,244,239,0.75)),
        repeating-linear-gradient(
          45deg,
          rgba(0,0,0,0.02),
          rgba(0,0,0,0.02) 10px,
          rgba(255,255,255,0.0) 10px,
          rgba(255,255,255,0.0) 20px
        );
      display: grid;
      place-items: center;
      overflow: hidden;
      position: relative;
    }
    .image-frame img {
      display: block;
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: #f8f8f6;
      image-rendering: auto;
      user-select: none;
      -webkit-user-drag: none;
    }
    .empty-state {
      padding: 24px;
      text-align: center;
      color: var(--muted);
      line-height: 1.5;
      max-width: 600px;
    }
    .footer {
      padding: 8px 10px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto auto;
      gap: 8px;
      align-items: center;
    }
    .footer > div { min-width: 0; }
    .footer .path-line {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      line-height: 1.25;
    }
    .btn-neutral { background: rgba(0,0,0,0.06); color: var(--ink); }
    .confidence-box {
      width: min(52vw, 620px);
      max-width: 100%;
      padding: 8px 10px;
      border: 1px solid rgba(194,87,43,0.14);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(194,87,43,0.04));
      border-radius: 12px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.45);
      display: grid;
      gap: 6px;
    }
    .confidence-title {
      font-size: 12px;
      color: #5f503f;
      font-weight: 600;
      line-height: 1.15;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .confidence-chips {
      display: flex;
      gap: 6px;
      flex-wrap: nowrap;
      white-space: nowrap;
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 2px;
    }
    .conf-chip {
      border: 1px solid rgba(0,0,0,0.07);
      background: rgba(255,255,255,0.85);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 13px;
      line-height: 1.2;
      color: var(--ink);
    }
    .conf-chip.current {
      border-color: rgba(194,87,43,0.25);
      background: rgba(194,87,43,0.08);
      color: #7b3213;
      font-weight: 600;
      font-size: 14px;
    }
    .conf-chip.muted {
      color: var(--muted);
      background: rgba(0,0,0,0.03);
    }
    .hotkeys {
      font-size: 11px;
      color: var(--muted);
      display: flex;
      gap: 6px;
      flex-wrap: nowrap;
      white-space: nowrap;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 0 2px 2px;
    }
    kbd {
      border: 1px solid var(--line);
      border-bottom-width: 2px;
      border-radius: 6px;
      padding: 1px 6px;
      background: white;
      font-size: 11px;
      color: var(--ink);
    }
    .path-line {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
      word-break: break-word;
    }
    .path-line strong {
      color: var(--ink);
      font-weight: 600;
    }
    .banner {
      display: none;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(194, 87, 43, 0.25);
      background: rgba(194, 87, 43, 0.08);
      color: #6e2f15;
      font-size: 13px;
    }
    .banner.show { display: block; }
    .session-tools {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
    }
    .mini-note {
      font-size: 11px;
      color: var(--muted);
      line-height: 1.25;
    }
    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 18, 15, 0.48);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 16px;
      z-index: 50;
    }
    .overlay.open { display: flex; }
    .modal {
      width: min(560px, 100%);
      background: rgba(255,255,255,0.97);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 18px;
      box-shadow: 0 30px 80px rgba(0,0,0,0.25);
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .modal-title {
      font-weight: 700;
      font-size: 18px;
      margin: 0;
    }
    .modal-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .modal-actions button {
      padding: 14px;
      border: 1px solid transparent;
      text-align: left;
    }
    .danger-btn {
      background: rgba(194,87,43,0.08);
      color: #872f0d;
      border-color: rgba(194,87,43,0.18);
    }
    .relabel-btn {
      background: rgba(60,80,120,0.06);
      color: #243e67;
      border-color: rgba(60,80,120,0.18);
    }
    .modal small {
      color: var(--muted);
      line-height: 1.35;
    }
    .relabel-box {
      display: none;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    .relabel-box.show { display: grid; }
    .loading {
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(255,255,255,0.55);
      font-weight: 600;
      color: var(--muted);
    }
    .loading.show { display: grid; }
    @media (max-width: 980px) {
      body {
        overflow: auto;
      }
      .page {
        grid-template-columns: 1fr;
        height: auto;
        min-height: 100vh;
        min-height: 100dvh;
      }
      .panel {
        height: auto;
        max-height: none;
      }
      .main {
        height: auto;
        min-height: 60vh;
        overflow: visible;
        grid-template-rows: auto auto minmax(300px, 56vh) auto auto;
      }
      .topbar {
        grid-template-columns: 1fr;
        align-items: stretch;
      }
      .topbar-side {
        justify-items: stretch;
      }
      .metrics-row {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        justify-content: stretch;
      }
      .confidence-box {
        width: 100%;
      }
      .footer {
        grid-template-columns: 1fr;
      }
      .modal-actions {
        grid-template-columns: 1fr;
      }
      .relabel-box {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <aside class="panel">
      <h1 class="title">Dataset Tinder</h1>
      <div class="subtle">
        Выберите класс и проходите изображения по одному.<br>
        <strong>←</strong> = OK, <strong>→</strong> = подозрительное (мусор / другой класс).
      </div>

      <div class="info-box" id="sessionInfo">Загрузка...</div>

      <div class="session-tools">
        <button id="deletePrevSessionsBtn" class="btn-danger-soft">Удалить прошлые сессии</button>
        <div class="mini-note" id="sessionToolsNote">Текущая сессия не будет удалена.</div>
      </div>

      <div class="row">
        <select id="classSelect" aria-label="Класс"></select>
        <button id="openClassBtn" class="btn-primary">Открыть</button>
      </div>

      <div class="banner" id="errorBanner"></div>

      <div class="subtle" style="margin-top:4px;">Классы (можно кликать):</div>
      <div class="class-list" id="classList"></div>

      <div class="info-box">
        <div><strong>Файлы результатов (текущая сессия):</strong></div>
        <div class="path-line" id="trashPath"></div>
        <div class="path-line" id="wrongPath"></div>
        <div class="path-line" id="allPath"></div>
      </div>
    </aside>

    <main class="main">
      <div class="card topbar">
        <div>
          <div id="currentClassTitle" style="font-size:20px;font-weight:700;">Класс не выбран</div>
          <div class="subtle" id="currentClassSubtitle">Выберите папку слева.</div>
        </div>
        <div class="topbar-side">
          <div class="confidence-box">
            <div class="confidence-title" id="confidenceTitle">Уверенность модели: нет данных</div>
            <div class="confidence-chips" id="confidenceChips"></div>
          </div>
          <div class="metrics-row">
            <div class="metric">
              <span class="label">OK</span>
              <span class="value" id="metricOk">0</span>
            </div>
            <div class="metric">
              <span class="label">Мусор</span>
              <span class="value" id="metricTrash">0</span>
            </div>
            <div class="metric">
              <span class="label">Другой класс</span>
              <span class="value" id="metricOther">0</span>
            </div>
          </div>
        </div>
      </div>

      <div class="card progress-wrap">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <div class="progress-text">
          <span id="progressText">0 / 0</span>
          <span id="progressPercent">0%</span>
          <span id="remainingText">Осталось: 0</span>
        </div>
      </div>

      <div class="card viewer">
        <div class="image-frame" id="imageFrame">
          <div class="empty-state" id="emptyState">
            Выберите класс слева, чтобы начать просмотр.
          </div>
          <img id="mainImage" alt="Текущее изображение" style="display:none;" />
          <div class="loading" id="imageLoading">Загрузка изображения...</div>
        </div>
      </div>

      <div class="card footer">
        <div>
          <div class="path-line"><strong>Файл:</strong> <span id="fileNameText">—</span></div>
          <div class="path-line"><strong>Путь:</strong> <span id="filePathText">—</span></div>
        </div>
        <button id="undoBtn" class="btn-neutral" disabled>⌫ Назад</button>
        <button id="okBtn" class="btn-good" disabled>← OK</button>
        <button id="flagBtn" class="btn-bad" disabled>→ Подозрительное</button>
      </div>

      <div class="hotkeys">
        <span><kbd>←</kbd> OK</span>
        <span><kbd>→</kbd> Открыть выбор (мусор / другой класс)</span>
        <span><kbd>Backspace</kbd>/<kbd>Z</kbd> Назад (Undo)</span>
        <span><kbd>Esc</kbd> Закрыть окно выбора</span>
        <span><kbd>1</kbd> Мусор (в окне выбора)</span>
        <span><kbd>2</kbd> Другой класс (в окне выбора)</span>
        <span><kbd>Enter</kbd> Подтвердить другой класс</span>
      </div>
    </main>
  </div>

  <div class="overlay" id="decisionOverlay" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
      <h2 class="modal-title" id="modalTitle">Это подозрительное фото</h2>
      <small>
        Выберите, что делать с текущим изображением. После подтверждения фото не повторится в текущей сессии сервера.
      </small>
      <div class="modal-actions">
        <button class="danger-btn" id="trashBtn">
          <strong>1. Мусор</strong><br>
          <small>Записать в отдельный список мусора</small>
        </button>
        <button class="relabel-btn" id="showRelabelBtn">
          <strong>2. Другой класс</strong><br>
          <small>Выбрать правильный класс и записать в отдельный список</small>
        </button>
      </div>
      <div class="relabel-box" id="relabelBox">
        <select id="targetClassSelect" aria-label="Правильный класс"></select>
        <button id="confirmRelabelBtn" class="btn-primary">Подтвердить</button>
      </div>
      <div class="row" style="justify-content:flex-end;">
        <button id="cancelModalBtn" class="btn-ghost">Отмена (Esc)</button>
      </div>
    </div>
  </div>

  <script>
    const state = {
      bootstrap: null,
      currentClass: null,
      currentPayload: null,
      modalOpen: false,
      relabelMode: false,
      submitting: false,
    };

    const els = {
      classSelect: document.getElementById("classSelect"),
      openClassBtn: document.getElementById("openClassBtn"),
      classList: document.getElementById("classList"),
      sessionInfo: document.getElementById("sessionInfo"),
      deletePrevSessionsBtn: document.getElementById("deletePrevSessionsBtn"),
      sessionToolsNote: document.getElementById("sessionToolsNote"),
      errorBanner: document.getElementById("errorBanner"),
      currentClassTitle: document.getElementById("currentClassTitle"),
      currentClassSubtitle: document.getElementById("currentClassSubtitle"),
      metricOk: document.getElementById("metricOk"),
      metricTrash: document.getElementById("metricTrash"),
      metricOther: document.getElementById("metricOther"),
      progressFill: document.getElementById("progressFill"),
      progressText: document.getElementById("progressText"),
      progressPercent: document.getElementById("progressPercent"),
      remainingText: document.getElementById("remainingText"),
      emptyState: document.getElementById("emptyState"),
      mainImage: document.getElementById("mainImage"),
      imageLoading: document.getElementById("imageLoading"),
      fileNameText: document.getElementById("fileNameText"),
      filePathText: document.getElementById("filePathText"),
      confidenceTitle: document.getElementById("confidenceTitle"),
      confidenceChips: document.getElementById("confidenceChips"),
      undoBtn: document.getElementById("undoBtn"),
      okBtn: document.getElementById("okBtn"),
      flagBtn: document.getElementById("flagBtn"),
      overlay: document.getElementById("decisionOverlay"),
      trashBtn: document.getElementById("trashBtn"),
      showRelabelBtn: document.getElementById("showRelabelBtn"),
      relabelBox: document.getElementById("relabelBox"),
      targetClassSelect: document.getElementById("targetClassSelect"),
      confirmRelabelBtn: document.getElementById("confirmRelabelBtn"),
      cancelModalBtn: document.getElementById("cancelModalBtn"),
      trashPath: document.getElementById("trashPath"),
      wrongPath: document.getElementById("wrongPath"),
      allPath: document.getElementById("allPath"),
    };

    function showError(message) {
      els.errorBanner.textContent = message || "Ошибка";
      els.errorBanner.classList.add("show");
    }

    function clearError() {
      els.errorBanner.classList.remove("show");
      els.errorBanner.textContent = "";
    }

    async function fetchJson(url, options = {}) {
      const resp = await fetch(url, options);
      const text = await resp.text();
      let data = {};
      try {
        data = text ? JSON.parse(text) : {};
      } catch (e) {
        data = { error: text || "Некорректный ответ сервера" };
      }
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      return data;
    }

    async function postJson(url, payload) {
      return fetchJson(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    function canUndo(payload) {
      const progress = (payload && payload.progress) || {};
      return !!(payload && payload.class_name && Number(progress.reviewed || 0) > 0);
    }

    function setSubmitting(flag) {
      state.submitting = flag;
      const hasItem = !!(state.currentPayload && state.currentPayload.item);
      els.okBtn.disabled = flag || !hasItem;
      els.flagBtn.disabled = flag || !hasItem;
      els.undoBtn.disabled = flag || !canUndo(state.currentPayload);
      els.openClassBtn.disabled = flag;
      els.deletePrevSessionsBtn.disabled = flag || !(state.bootstrap && state.bootstrap.sessions && Number(state.bootstrap.sessions.count_previous || 0) > 0);
      els.confirmRelabelBtn.disabled = flag;
      els.trashBtn.disabled = flag;
      els.showRelabelBtn.disabled = flag;
    }

    function updateOutputPaths(info) {
      if (!info || !info.output_files) return;
      els.trashPath.textContent = `Мусор: ${info.output_files.trash_csv || "—"}`;
      els.wrongPath.textContent = `Другой класс: ${info.output_files.wrong_class_csv || "—"}`;
      els.allPath.textContent = `Все решения: ${info.output_files.decisions_jsonl || "—"}`;
    }

    function formatPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      const pct = Number(value) * 100;
      return pct >= 10 ? `${pct.toFixed(0)}%` : `${pct.toFixed(1)}%`;
    }

    function renderConfidence(payload) {
      const conf = payload && payload.confidence;
      els.confidenceChips.innerHTML = "";

      if (!conf || !conf.available) {
        const reason = (conf && conf.reason) ? ` (${conf.reason})` : "";
        els.confidenceTitle.textContent = `Уверенность модели: нет данных${reason}`;
        const chip = document.createElement("span");
        chip.className = "conf-chip muted";
        chip.textContent = "cache не найден / фото не покрыто";
        els.confidenceChips.appendChild(chip);
        return;
      }

      const top1 = conf.top1 || {};
      const srcLabel = conf.source === "oof" ? "OOF" : (conf.source === "infer" ? "MPS" : (conf.source || "cache"));
      els.confidenceTitle.textContent =
        `Уверенность (${srcLabel}): top-1 ${formatPct(top1.prob)} ${top1.class_name || "—"}`
        + (conf.current_class_prob !== null && conf.current_class_prob !== undefined
          ? ` | текущий класс: ${formatPct(conf.current_class_prob)}`
          : "");

      const rows = conf.rows_top_display || [];
      if (!rows.length) {
        const chip = document.createElement("span");
        chip.className = "conf-chip muted";
        chip.textContent = "пустой список вероятностей";
        els.confidenceChips.appendChild(chip);
        return;
      }
      for (const row of rows) {
        const chip = document.createElement("span");
        chip.className = `conf-chip${row.is_current_class ? " current" : ""}`;
        chip.textContent = `${formatPct(row.prob)} ${row.class_name}`;
        els.confidenceChips.appendChild(chip);
      }
      if ((conf.num_classes || 0) > rows.length) {
        const more = document.createElement("span");
        more.className = "conf-chip muted";
        more.textContent = `ещё ${Math.max(0, (conf.num_classes || 0) - rows.length)}`;
        els.confidenceChips.appendChild(more);
      }
    }

    function renderClassList() {
      const bootstrap = state.bootstrap;
      if (!bootstrap) return;
      els.classList.innerHTML = "";
      const classes = bootstrap.classes || [];
      for (const item of classes) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "class-item";
        if (state.currentClass === item.class_name) btn.classList.add("active");
        if (item.done) btn.classList.add("done");

        const left = document.createElement("span");
        left.className = "class-name";
        left.textContent = item.class_name;

        const right = document.createElement("span");
        right.className = "class-meta";
        if (item.loaded && item.total !== null) {
          right.textContent = `${item.reviewed}/${item.total}`;
        } else {
          right.textContent = "не открыт";
        }

        btn.append(left, right);
        btn.addEventListener("click", () => openClass(item.class_name));
        els.classList.appendChild(btn);
      }
    }

    function renderBootstrapInfo() {
      if (!state.bootstrap) return;
      const b = state.bootstrap;
      const sessions = b.sessions || {};
      const conf = b.confidence_cache || {};
      const confText = conf.loaded
        ? `confidence: ${conf.records || 0} фото`
        : `confidence: нет (${conf.reason || "cache не загружен"})`;
      els.sessionInfo.innerHTML =
        `<div><strong>Сессия:</strong> ${b.session_id}</div>` +
        `<div><strong>Классов:</strong> ${(b.class_names || []).length}</div>` +
        `<div><strong>Сессий:</strong> ${sessions.count_total || 0} (предыдущих: ${sessions.count_previous || 0})</div>` +
        `<div><strong>${confText}</strong></div>` +
        `<div class="path-line"><strong>Датасет:</strong> ${b.train_root}</div>` +
        `<div class="path-line"><strong>Сессия сохраняется в:</strong> ${b.session_dir}</div>`;
      els.deletePrevSessionsBtn.disabled = state.submitting || Number(sessions.count_previous || 0) <= 0;
      els.sessionToolsNote.textContent = Number(sessions.count_previous || 0) > 0
        ? `Будут удалены ${sessions.count_previous} предыдущих сессий. Текущая останется.`
        : "Предыдущих сессий нет.";
      updateOutputPaths(b);
    }

    function fillClassSelect(classNames) {
      els.classSelect.innerHTML = "";
      for (const name of classNames || []) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        els.classSelect.appendChild(option);
      }
    }

    function resetModal() {
      state.relabelMode = false;
      els.relabelBox.classList.remove("show");
      els.targetClassSelect.innerHTML = "";
    }

    function openModal() {
      if (!state.currentPayload || !state.currentPayload.item || state.submitting) return;
      state.modalOpen = true;
      resetModal();
      els.overlay.classList.add("open");
      els.overlay.setAttribute("aria-hidden", "false");
    }

    function closeModal() {
      state.modalOpen = false;
      resetModal();
      els.overlay.classList.remove("open");
      els.overlay.setAttribute("aria-hidden", "true");
    }

    function buildRelabelOptions() {
      els.targetClassSelect.innerHTML = "";
      const classNames = (state.currentPayload && state.currentPayload.class_names) || [];
      const current = state.currentClass;
      const options = classNames.filter((n) => n !== current);
      for (const name of options) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        els.targetClassSelect.appendChild(opt);
      }
      if (!options.length && current) {
        const opt = document.createElement("option");
        opt.value = current;
        opt.textContent = current;
        els.targetClassSelect.appendChild(opt);
      }
    }

    function showRelabelChooser() {
      if (!state.currentPayload || !state.currentPayload.item) return;
      state.relabelMode = true;
      buildRelabelOptions();
      els.relabelBox.classList.add("show");
      els.targetClassSelect.focus();
    }

    function updateButtonsByPayload(payload) {
      const hasItem = !!(payload && payload.item);
      els.okBtn.disabled = !hasItem || state.submitting;
      els.flagBtn.disabled = !hasItem || state.submitting;
      els.undoBtn.disabled = !canUndo(payload) || state.submitting;
    }

    function renderPayload(payload) {
      state.currentPayload = payload;
      state.currentClass = payload ? payload.class_name : state.currentClass;
      clearError();
      updateButtonsByPayload(payload);
      updateOutputPaths(payload);
      renderConfidence(payload);

      const progress = (payload && payload.progress) || {};
      const counts = (progress && progress.counts) || { ok: 0, trash: 0, other_class: 0 };
      els.metricOk.textContent = String(counts.ok || 0);
      els.metricTrash.textContent = String(counts.trash || 0);
      els.metricOther.textContent = String(counts.other_class || 0);

      const total = Number(progress.total || 0);
      const reviewed = Number(progress.reviewed || 0);
      const remaining = Number(progress.remaining || 0);
      const percent = Number(progress.percent || 0);
      els.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
      els.progressText.textContent = `${reviewed} / ${total}`;
      els.progressPercent.textContent = `${percent.toFixed(2)}%`;
      els.remainingText.textContent = `Осталось: ${remaining}`;

      if (!payload || !payload.class_name) {
        els.currentClassTitle.textContent = "Класс не выбран";
        els.currentClassSubtitle.textContent = "Выберите папку слева.";
      } else if (progress.done) {
        els.currentClassTitle.textContent = `Готово: ${payload.class_name}`;
        els.currentClassSubtitle.textContent = "Все изображения этого класса просмотрены в текущей сессии.";
      } else {
        els.currentClassTitle.textContent = payload.class_name;
        els.currentClassSubtitle.textContent = "Листайте стрелками ← / →, Undo: Backspace или Z.";
      }

      if (!payload || !payload.item) {
        els.mainImage.style.display = "none";
        els.mainImage.removeAttribute("src");
        els.emptyState.style.display = "block";
        els.emptyState.textContent = progress.done
          ? "Класс полностью просмотрен. Выберите другой класс слева."
          : "Выберите класс слева, чтобы начать просмотр.";
        els.fileNameText.textContent = "—";
        els.filePathText.textContent = "—";
      } else {
        const item = payload.item;
        els.emptyState.style.display = "none";
        els.fileNameText.textContent = item.filename || "—";
        els.filePathText.textContent = item.relative_path || "—";
        els.imageLoading.classList.add("show");
        els.mainImage.style.display = "block";
        els.mainImage.onload = () => els.imageLoading.classList.remove("show");
        els.mainImage.onerror = () => {
          els.imageLoading.classList.remove("show");
          showError("Не удалось загрузить изображение.");
        };
        const cacheBust = `cb=${Date.now()}`;
        els.mainImage.src = `${item.image_url}&${cacheBust}`;
      }

      renderClassList();
      if (payload && payload.class_name) {
        els.classSelect.value = payload.class_name;
      }
    }

    async function refreshBootstrap() {
      const data = await fetchJson("/api/classes");
      state.bootstrap = data;
      renderBootstrapInfo();
      fillClassSelect(data.class_names || []);
      if (state.currentClass) {
        els.classSelect.value = state.currentClass;
      }
      renderClassList();
    }

    async function openClass(className) {
      if (!className) return;
      closeModal();
      setSubmitting(true);
      try {
        const payload = await postJson("/api/select_class", { class_name: className });
        state.currentClass = className;
        renderPayload(payload);
        await refreshBootstrap();
      } catch (err) {
        showError(err.message);
      } finally {
        setSubmitting(false);
      }
    }

    async function submitDecision(decision, targetClass = null) {
      if (state.submitting) return;
      const payload = state.currentPayload;
      if (!payload || !payload.item) return;
      setSubmitting(true);
      try {
        const nextState = await postJson("/api/decision", {
          class_name: payload.class_name,
          decision,
          image_key: payload.item.image_key,
          target_class: targetClass,
        });
        closeModal();
        renderPayload(nextState);
        await refreshBootstrap();
      } catch (err) {
        showError(err.message);
      } finally {
        setSubmitting(false);
      }
    }

    async function submitUndo() {
      if (state.submitting) return;
      const payload = state.currentPayload;
      if (!canUndo(payload)) return;
      closeModal();
      setSubmitting(true);
      try {
        const nextState = await postJson("/api/undo", {
          class_name: payload.class_name,
        });
        renderPayload(nextState);
        await refreshBootstrap();
      } catch (err) {
        showError(err.message);
      } finally {
        setSubmitting(false);
      }
    }

    async function deletePreviousSessions() {
      if (state.submitting) return;
      const prevCount = Number((state.bootstrap && state.bootstrap.sessions && state.bootstrap.sessions.count_previous) || 0);
      if (prevCount <= 0) return;
      const ok = window.confirm(`Удалить ${prevCount} предыдущих сессий? Текущая сессия останется.`);
      if (!ok) return;
      setSubmitting(true);
      try {
        const res = await postJson("/api/delete_previous_sessions", {});
        const deletedCount = Number(res.deleted_count || 0);
        const failedCount = Number(res.failed_count || 0);
        if (failedCount > 0) {
          showError(`Удалено ${deletedCount}, ошибок: ${failedCount}`);
        } else {
          clearError();
          els.sessionToolsNote.textContent = `Удалено предыдущих сессий: ${deletedCount}. Текущая сохранена.`;
        }
        await refreshBootstrap();
      } catch (err) {
        showError(err.message);
      } finally {
        setSubmitting(false);
      }
    }

    function activeTagName() {
      return (document.activeElement && document.activeElement.tagName || "").toLowerCase();
    }

    function handleGlobalKeydown(event) {
      const tag = activeTagName();
      const typing = tag === "input" || tag === "textarea" || tag === "select";

      if (state.modalOpen) {
        if (event.key === "Escape") {
          event.preventDefault();
          closeModal();
          return;
        }
        if (event.key === "1") {
          event.preventDefault();
          submitDecision("trash");
          return;
        }
        if (event.key === "2") {
          event.preventDefault();
          showRelabelChooser();
          return;
        }
        if (event.key === "Enter" && state.relabelMode) {
          event.preventDefault();
          if (els.targetClassSelect.value) {
            submitDecision("other_class", els.targetClassSelect.value);
          }
          return;
        }
        return;
      }

      if (typing) return;

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        submitDecision("ok");
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        openModal();
      } else if (
        !event.repeat &&
        (event.key === "Backspace" || event.code === "KeyZ")
      ) {
        event.preventDefault();
        submitUndo();
      }
    }

    function bindEvents() {
      els.openClassBtn.addEventListener("click", () => openClass(els.classSelect.value));
      els.deletePrevSessionsBtn.addEventListener("click", deletePreviousSessions);
      els.undoBtn.addEventListener("click", submitUndo);
      els.okBtn.addEventListener("click", () => submitDecision("ok"));
      els.flagBtn.addEventListener("click", openModal);

      els.trashBtn.addEventListener("click", () => submitDecision("trash"));
      els.showRelabelBtn.addEventListener("click", showRelabelChooser);
      els.confirmRelabelBtn.addEventListener("click", () => {
        if (!els.targetClassSelect.value) {
          showError("Выберите правильный класс.");
          return;
        }
        submitDecision("other_class", els.targetClassSelect.value);
      });
      els.cancelModalBtn.addEventListener("click", closeModal);
      els.overlay.addEventListener("click", (e) => {
        if (e.target === els.overlay) closeModal();
      });

      document.addEventListener("keydown", handleGlobalKeydown);
    }

    async function init() {
      bindEvents();
      setSubmitting(false);
      try {
        await refreshBootstrap();
        if (state.bootstrap && state.bootstrap.class_names && state.bootstrap.class_names.length) {
          els.classSelect.value = state.bootstrap.class_names[0];
        }
      } catch (err) {
        showError(err.message);
      }
    }

    init();
  </script>
</body>
</html>
"""


def create_app(store: ReviewStore):
    try:
        from flask import Flask, Response, jsonify, request, send_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Flask is not installed. Install it with: python3 -m pip install flask"
        ) from exc

    app = Flask(__name__)
    html = build_html()

    @app.get("/")
    def index():
        return Response(html, content_type="text/html; charset=utf-8")

    @app.get("/api/classes")
    def api_classes():
        return jsonify(store.classes_overview())

    @app.post("/api/select_class")
    def api_select_class():
        data = request.get_json(silent=True) or {}
        class_name = (data.get("class_name") or "").strip()
        if not class_name:
            return jsonify({"error": "class_name is required"}), 400
        try:
            payload = store.read_state(class_name)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.get("/api/image")
    def api_image():
        class_name = (request.args.get("class_name") or "").strip()
        index_raw = request.args.get("index")
        if not class_name:
            return jsonify({"error": "class_name is required"}), 400
        try:
            index = int(index_raw) if index_raw is not None else -1
        except ValueError:
            return jsonify({"error": "index must be integer"}), 400
        try:
            image_path = store.image_path_for(class_name, index)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        mimetype = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        response = send_file(image_path, mimetype=mimetype, conditional=True, etag=False)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/decision")
    def api_decision():
        data = request.get_json(silent=True) or {}
        class_name = (data.get("class_name") or "").strip()
        decision = (data.get("decision") or "").strip()
        image_key = (data.get("image_key") or "").strip()
        target_class = data.get("target_class")
        if isinstance(target_class, str):
            target_class = target_class.strip() or None
        if not class_name:
            return jsonify({"error": "class_name is required"}), 400
        if not decision:
            return jsonify({"error": "decision is required"}), 400
        if not image_key:
            return jsonify({"error": "image_key is required"}), 400
        try:
            payload = store.apply_decision(
                class_name=class_name,
                decision=decision,
                image_key=image_key,
                target_class=target_class,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.post("/api/undo")
    def api_undo():
        data = request.get_json(silent=True) or {}
        class_name = (data.get("class_name") or "").strip()
        if not class_name:
            return jsonify({"error": "class_name is required"}), 400
        try:
            payload = store.undo_last(class_name)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.post("/api/delete_previous_sessions")
    def api_delete_previous_sessions():
        try:
            result = store.delete_previous_sessions()
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Local Tinder-like dataset cleaning tool: left arrow = OK, right arrow = trash/relabel."
        )
    )
    p.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help=f"Path to class folders (default: {DEFAULT_TRAIN_ROOT})",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Where to save review session files (default: {DEFAULT_OUTPUT_ROOT})",
    )
    p.add_argument(
        "--confidence-csv",
        type=Path,
        default=None,
        help="Optional confidence cache CSV for per-image ensemble probabilities",
    )
    p.add_argument(
        "--confidence-meta-json",
        type=Path,
        default=None,
        help="Optional confidence cache metadata JSON (class label mapping)",
    )
    p.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode (not needed for normal usage)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_root = args.train_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    confidence_dir_default = output_root / "confidence_cache"
    confidence_csv = (
        args.confidence_csv.expanduser().resolve()
        if args.confidence_csv is not None
        else confidence_dir_default / f"{DEFAULT_CONFIDENCE_CACHE_NAME}.csv"
    )
    confidence_meta_json = (
        args.confidence_meta_json.expanduser().resolve()
        if args.confidence_meta_json is not None
        else confidence_dir_default / f"{DEFAULT_CONFIDENCE_CACHE_NAME}.meta.json"
    )

    if not train_root.exists():
        print(f"[ERROR] Train root not found: {train_root}", file=sys.stderr)
        return 2
    if not train_root.is_dir():
        print(f"[ERROR] Train root is not a directory: {train_root}", file=sys.stderr)
        return 2

    try:
        class_names = scan_class_names(train_root)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not class_names:
        print(f"[ERROR] No class folders found in: {train_root}", file=sys.stderr)
        return 2

    session_id = make_session_id()
    session_dir = output_root / f"session_{session_id}"
    store = ReviewStore(
        train_root=train_root,
        output_root=output_root,
        session_id=session_id,
        class_names=class_names,
        session_dir=session_dir,
        trash_csv=session_dir / "trash_images.csv",
        wrong_class_csv=session_dir / "wrong_class_images.csv",
        decisions_jsonl=session_dir / "all_decisions.jsonl",
        confidence_csv=confidence_csv,
        confidence_meta_json=confidence_meta_json,
    )

    try:
        app = create_app(store)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    print(f"Session ID: {session_id}")
    print(f"Train root:  {train_root}")
    print(f"Output dir:  {session_dir}")
    print(f"Confidence cache CSV:  {confidence_csv}")
    print(f"Confidence meta JSON:  {confidence_meta_json}")
    print(f"Open in browser: http://{args.host}:{args.port}")
    print("Controls: Left arrow = OK, Right arrow = suspicious (trash/relabel)")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
