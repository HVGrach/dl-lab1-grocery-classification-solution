#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class Score:
    acc: float
    f1_macro: float
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outputs-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps",
    )
    p.add_argument(
        "--train-csv",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/train_clean_strict.csv",
    )
    p.add_argument(
        "--sample-submission",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/sample_submission.csv",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-f1-weight", type=float, default=0.15)
    p.add_argument(
        "--margin-grid",
        type=str,
        default="0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30",
    )
    p.add_argument("--prob-grid", type=str, default="0.45,0.50,0.55")
    return p.parse_args()


def parse_float_grid(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def evaluate(y_true: np.ndarray, pred: np.ndarray, score_f1_weight: float) -> Score:
    acc = float(accuracy_score(y_true, pred))
    f1m = float(f1_score(y_true, pred, average="macro"))
    return Score(acc=acc, f1_macro=f1m, score=acc + score_f1_weight * f1m)


def build_pair_features(
    alias_logits: Dict[str, np.ndarray],
    aliases: List[str],
    blend_logits: np.ndarray,
    a: int,
    b: int,
) -> np.ndarray:
    feats = []
    for alias in aliases:
        lg = alias_logits[alias]
        la = lg[:, a]
        lb = lg[:, b]
        feats.extend([la, lb, la - lb])
    ba = blend_logits[:, a]
    bb = blend_logits[:, b]
    feats.extend([ba, bb, ba - bb, ba + bb])
    return np.stack(feats, axis=1)


def fit_pair_specialist(
    X_all: np.ndarray,
    y_all: np.ndarray,
    a: int,
    b: int,
    seed: int,
) -> Tuple[np.ndarray, LogisticRegression]:
    pair_mask = (y_all == a) | (y_all == b)
    idx = np.where(pair_mask)[0]
    X_pair = X_all[idx]
    y_pair = (y_all[idx] == a).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_pair_prob = np.zeros(len(idx), dtype=np.float64)
    for tr, va in skf.split(X_pair, y_pair):
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_pair[tr], y_pair[tr])
        oof_pair_prob[va] = clf.predict_proba(X_pair[va])[:, 1]

    final_model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=seed,
    )
    final_model.fit(X_pair, y_pair)

    # For non-pair rows we can use final model outputs; pair rows use strict OOF probs.
    prob_all = final_model.predict_proba(X_all)[:, 1]
    prob_all[idx] = oof_pair_prob
    return prob_all, final_model


def apply_pair_rerank(
    pred: np.ndarray,
    top2: np.ndarray,
    margins: np.ndarray,
    prob_a: np.ndarray,
    a: int,
    b: int,
    margin_thr: float,
    prob_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pair_top2 = ((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a))
    mask = pair_top2 & (margins <= margin_thr)
    out = pred.copy()
    out[mask] = np.where(prob_a[mask] >= prob_thr, a, b)
    return out, mask


def label_name_map(train_csv: Path) -> Dict[int, str]:
    df = pd.read_csv(train_csv)
    df["label"] = df["label"].astype(int)
    df["class_name"] = df["image_id"].str.split("/").str[0]
    return (
        df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )


def main() -> None:
    args = parse_args()
    margin_grid = parse_float_grid(args.margin_grid)
    prob_grid = parse_float_grid(args.prob_grid)

    out_dir = Path(args.outputs_dir)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Prefer refined weights/bias from previous step if available.
    refined_w_path = analysis_dir / "refined_ensemble_weights.json"
    refined_b_path = analysis_dir / "refined_ensemble_bias.json"
    if refined_w_path.exists():
        weights = json.loads(refined_w_path.read_text(encoding="utf-8"))
    else:
        weights = json.loads((out_dir / "ensemble_weights.json").read_text(encoding="utf-8"))

    aliases = list(weights.keys())
    alias_oof = {a: np.load(out_dir / a / "oof_logits.npy") for a in aliases}
    alias_test = {a: np.load(out_dir / a / "test_logits.npy") for a in aliases}
    y_true = np.load(out_dir / aliases[0] / "oof_targets.npy")

    num_classes = alias_oof[aliases[0]].shape[1]
    if refined_b_path.exists():
        bias_map = json.loads(refined_b_path.read_text(encoding="utf-8"))
        bias = np.array([float(bias_map.get(str(i), 0.0)) for i in range(num_classes)], dtype=np.float64)
    else:
        bias = np.zeros(num_classes, dtype=np.float64)

    # Base refined logits.
    blend_oof = np.zeros_like(alias_oof[aliases[0]], dtype=np.float64)
    blend_test = np.zeros_like(alias_test[aliases[0]], dtype=np.float64)
    for a in aliases:
        w = float(weights[a])
        blend_oof += w * alias_oof[a]
        blend_test += w * alias_test[a]
    blend_oof += bias
    blend_test += bias

    base_prob_oof = softmax(blend_oof)
    base_pred_oof = base_prob_oof.argmax(1)
    top2_oof = np.argsort(-base_prob_oof, axis=1)[:, :2]
    margins_oof = base_prob_oof[np.arange(len(base_prob_oof)), top2_oof[:, 0]] - base_prob_oof[np.arange(len(base_prob_oof)), top2_oof[:, 1]]

    base_prob_test = softmax(blend_test)
    base_pred_test = base_prob_test.argmax(1)
    top2_test = np.argsort(-base_prob_test, axis=1)[:, :2]
    margins_test = base_prob_test[np.arange(len(base_prob_test)), top2_test[:, 0]] - base_prob_test[np.arange(len(base_prob_test)), top2_test[:, 1]]

    label_to_name = label_name_map(Path(args.train_csv))
    name_to_label = {v: k for k, v in label_to_name.items()}

    pair_names = [
        ("Киви", "Картофель"),
        ("Мандарины", "Апельсин"),
        ("Яблоки красные", "Томаты"),
    ]
    pairs: List[Tuple[int, int]] = [(name_to_label[a], name_to_label[b]) for a, b in pair_names]

    # Train specialists.
    specialist_oof_prob: Dict[Tuple[int, int], np.ndarray] = {}
    specialist_test_prob: Dict[Tuple[int, int], np.ndarray] = {}
    specialist_models: Dict[Tuple[int, int], LogisticRegression] = {}
    for a, b in pairs:
        X_oof = build_pair_features(alias_oof, aliases, blend_oof, a, b)
        X_test = build_pair_features(alias_test, aliases, blend_test, a, b)
        oof_prob, model = fit_pair_specialist(X_oof, y_true, a, b, seed=args.seed)
        test_prob = model.predict_proba(X_test)[:, 1]
        specialist_oof_prob[(a, b)] = oof_prob
        specialist_test_prob[(a, b)] = test_prob
        specialist_models[(a, b)] = model

    # Sequential threshold tuning on OOF.
    current_pred = base_pred_oof.copy()
    pair_cfg = []
    base_score = evaluate(y_true, base_pred_oof, args.score_f1_weight)
    for a, b in pairs:
        best_local = {
            "margin_thr": 0.0,
            "prob_thr": 0.5,
            "score": evaluate(y_true, current_pred, args.score_f1_weight).score,
            "pred": current_pred,
            "mask_count": 0,
        }
        for mthr in margin_grid:
            for pthr in prob_grid:
                cand_pred, mask = apply_pair_rerank(
                    pred=current_pred,
                    top2=top2_oof,
                    margins=margins_oof,
                    prob_a=specialist_oof_prob[(a, b)],
                    a=a,
                    b=b,
                    margin_thr=mthr,
                    prob_thr=pthr,
                )
                s = evaluate(y_true, cand_pred, args.score_f1_weight).score
                if s > best_local["score"] + 1e-12:
                    best_local = {
                        "margin_thr": mthr,
                        "prob_thr": pthr,
                        "score": s,
                        "pred": cand_pred,
                        "mask_count": int(mask.sum()),
                    }
        current_pred = best_local["pred"]
        pair_cfg.append(
            {
                "pair": [a, b],
                "pair_names": [label_to_name[a], label_to_name[b]],
                "margin_thr": float(best_local["margin_thr"]),
                "prob_thr_for_first_class": float(best_local["prob_thr"]),
                "affected_oof_rows": int(best_local["mask_count"]),
            }
        )

    final_oof_pred = current_pred
    final_score = evaluate(y_true, final_oof_pred, args.score_f1_weight)

    # Apply tuned configuration to test.
    test_pred = base_pred_test.copy()
    pair_test_stats = []
    for cfg in pair_cfg:
        a, b = cfg["pair"]
        test_pred, mask = apply_pair_rerank(
            pred=test_pred,
            top2=top2_test,
            margins=margins_test,
            prob_a=specialist_test_prob[(a, b)],
            a=a,
            b=b,
            margin_thr=float(cfg["margin_thr"]),
            prob_thr=float(cfg["prob_thr_for_first_class"]),
        )
        pair_test_stats.append(
            {
                "pair": cfg["pair"],
                "pair_names": cfg["pair_names"],
                "affected_test_rows": int(mask.sum()),
            }
        )

    # Save outputs.
    sub = pd.read_csv(args.sample_submission)
    sub["label"] = test_pred
    sub_path = out_dir / "submission_ensemble_refined_specialist.csv"
    sub.to_csv(sub_path, index=False)

    # Diagnostics.
    cm = confusion_matrix(y_true, final_oof_pred)
    conf_rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            conf_rows.append(
                {
                    "count": int(cm[i, j]),
                    "true_label": int(i),
                    "pred_label": int(j),
                    "true_name": label_to_name[i],
                    "pred_name": label_to_name[j],
                    "row_share": float(cm[i, j] / (cm[i].sum() + 1e-12)),
                }
            )
    pd.DataFrame(conf_rows).sort_values(["count", "row_share"], ascending=False).head(50).to_csv(
        analysis_dir / "specialist_top_confusions.csv", index=False
    )

    summary = {
        "aliases": aliases,
        "implied_checkpoints_used": len(aliases) * 5,
        "base_metrics_before_specialist": {
            "acc": base_score.acc,
            "f1_macro": base_score.f1_macro,
            "score": base_score.score,
        },
        "metrics_after_specialist": {
            "acc": final_score.acc,
            "f1_macro": final_score.f1_macro,
            "score": final_score.score,
        },
        "pair_config": pair_cfg,
        "pair_test_stats": pair_test_stats,
        "submission_path": str(sub_path),
    }
    with (analysis_dir / "specialist_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Specialist Rerank Done ===")
    print("Base acc/f1:", f"{base_score.acc:.6f}", f"{base_score.f1_macro:.6f}")
    print("After specialist acc/f1:", f"{final_score.acc:.6f}", f"{final_score.f1_macro:.6f}")
    print("Submission:", sub_path)


if __name__ == "__main__":
    main()

