#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass
class Metrics:
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
    p.add_argument("--weight-trials", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focus-classes", type=int, default=8)
    p.add_argument("--bias-steps", type=str, default="0.20,0.10,0.05,0.02,0.01")
    p.add_argument("--score-f1-weight", type=float, default=0.15)
    return p.parse_args()


def metric_bundle(y_true: np.ndarray, logits: np.ndarray, score_f1_weight: float) -> Metrics:
    pred = logits.argmax(1)
    acc = float(accuracy_score(y_true, pred))
    f1m = float(f1_score(y_true, pred, average="macro"))
    score = acc + score_f1_weight * f1m
    return Metrics(acc=acc, f1_macro=f1m, score=score)


def class_report_df(
    y_true: np.ndarray,
    logits: np.ndarray,
    label_to_name: Dict[int, str],
) -> pd.DataFrame:
    pred = logits.argmax(1)
    classes = sorted(label_to_name.keys())
    rows: List[Dict] = []
    for c in classes:
        mask_true = y_true == c
        tp = int(((pred == c) & mask_true).sum())
        fp = int(((pred == c) & (~mask_true)).sum())
        fn = int(((pred != c) & mask_true).sum())
        support = int(mask_true.sum())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        rows.append(
            {
                "label": c,
                "class_name": label_to_name[c],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return pd.DataFrame(rows).sort_values("recall")


def top_confusions_df(
    y_true: np.ndarray,
    logits: np.ndarray,
    label_to_name: Dict[int, str],
    top_k: int = 20,
) -> pd.DataFrame:
    pred = logits.argmax(1)
    cm = confusion_matrix(y_true, pred)
    rows = []
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            rows.append(
                {
                    "count": int(cm[i, j]),
                    "true_label": int(i),
                    "pred_label": int(j),
                    "true_name": label_to_name[i],
                    "pred_name": label_to_name[j],
                    "row_share": float(cm[i, j] / (row_sum + 1e-12)),
                }
            )
    return pd.DataFrame(rows).sort_values(["count", "row_share"], ascending=False).head(top_k)


def random_search_weights(
    y_true: np.ndarray,
    oof_list: List[np.ndarray],
    trials: int,
    seed: int,
    score_f1_weight: float,
) -> Tuple[np.ndarray, Metrics]:
    rng = np.random.default_rng(seed)
    n = len(oof_list)
    base_w = np.ones(n, dtype=np.float64) / n
    blend = np.zeros_like(oof_list[0], dtype=np.float64)
    for w, lg in zip(base_w, oof_list):
        blend += w * lg
    best_w = base_w.copy()
    best_m = metric_bundle(y_true, blend, score_f1_weight)

    for _ in range(trials):
        w = rng.dirichlet(np.ones(n))
        blend = np.zeros_like(oof_list[0], dtype=np.float64)
        for wi, lg in zip(w, oof_list):
            blend += wi * lg
        m = metric_bundle(y_true, blend, score_f1_weight)
        if m.score > best_m.score:
            best_m = m
            best_w = w
    return best_w, best_m


def optimize_bias(
    y_true: np.ndarray,
    blend_logits: np.ndarray,
    focus_labels: List[int],
    score_f1_weight: float,
    steps: List[float],
) -> Tuple[np.ndarray, Metrics]:
    num_classes = blend_logits.shape[1]
    bias = np.zeros(num_classes, dtype=np.float64)
    best = metric_bundle(y_true, blend_logits + bias, score_f1_weight)

    for step in steps:
        improved = True
        while improved:
            improved = False
            for c in focus_labels:
                for delta in (step, -step):
                    cand = bias.copy()
                    cand[c] += delta
                    m = metric_bundle(y_true, blend_logits + cand, score_f1_weight)
                    if m.score > best.score + 1e-10:
                        best = m
                        bias = cand
                        improved = True
    return bias, best


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outputs_dir)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "all_model_oof_metrics.json").open("r", encoding="utf-8") as f:
        aliases = list(json.load(f).keys())
    aliases = [a for a in aliases if (out_dir / a / "oof_logits.npy").exists() and (out_dir / a / "test_logits.npy").exists()]
    if not aliases:
        raise RuntimeError("No model logits found in outputs directory.")

    y_true = np.load(out_dir / aliases[0] / "oof_targets.npy")
    oof_list = [np.load(out_dir / a / "oof_logits.npy") for a in aliases]
    test_list = [np.load(out_dir / a / "test_logits.npy") for a in aliases]

    train_df = pd.read_csv(args.train_csv)
    train_df["label"] = train_df["label"].astype(int)
    train_df["class_name"] = train_df["image_id"].str.split("/").str[0]
    label_to_name = (
        train_df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )

    # Existing weights if present; otherwise uniform.
    weights_path = out_dir / "ensemble_weights.json"
    if weights_path.exists():
        old_w_dict = json.loads(weights_path.read_text(encoding="utf-8"))
        old_w = np.array([float(old_w_dict.get(a, 0.0)) for a in aliases], dtype=np.float64)
        if old_w.sum() <= 1e-12:
            old_w = np.ones(len(aliases), dtype=np.float64) / len(aliases)
        old_w = old_w / old_w.sum()
    else:
        old_w = np.ones(len(aliases), dtype=np.float64) / len(aliases)

    base_oof = np.zeros_like(oof_list[0], dtype=np.float64)
    for w, lg in zip(old_w, oof_list):
        base_oof += w * lg
    base_metrics = metric_bundle(y_true, base_oof, args.score_f1_weight)

    best_w, w_metrics = random_search_weights(
        y_true=y_true,
        oof_list=oof_list,
        trials=args.weight_trials,
        seed=args.seed,
        score_f1_weight=args.score_f1_weight,
    )
    weighted_oof = np.zeros_like(oof_list[0], dtype=np.float64)
    for w, lg in zip(best_w, oof_list):
        weighted_oof += w * lg

    before_cls = class_report_df(y_true, weighted_oof, label_to_name)
    focus_labels = before_cls.head(args.focus_classes)["label"].tolist()
    steps = [float(x.strip()) for x in args.bias_steps.split(",") if x.strip()]
    bias, bias_metrics = optimize_bias(
        y_true=y_true,
        blend_logits=weighted_oof,
        focus_labels=focus_labels,
        score_f1_weight=args.score_f1_weight,
        steps=steps,
    )
    final_oof = weighted_oof + bias

    after_cls = class_report_df(y_true, final_oof, label_to_name)
    before_conf = top_confusions_df(y_true, weighted_oof, label_to_name, top_k=30)
    after_conf = top_confusions_df(y_true, final_oof, label_to_name, top_k=30)

    # Hard-case export for quick manual cleanup / targeted augmentations.
    final_pred = final_oof.argmax(1)
    train_df = train_df.reset_index(drop=True)
    if len(train_df) == len(final_pred):
        err_df = train_df[["image_id", "label"]].copy()
        err_df["pred"] = final_pred
        err_df["true_name"] = err_df["label"].map(label_to_name)
        err_df["pred_name"] = err_df["pred"].map(label_to_name)
        worst_labels = after_cls.head(3)["label"].tolist()
        hard_err = err_df[(err_df["label"].isin(worst_labels)) & (err_df["label"] != err_df["pred"])]
        hard_err.to_csv(analysis_dir / "hard_class_errors_top3.csv", index=False)

    # Test predictions: each alias test_logits is already 5-fold mean, so blending aliases uses all 15 checkpoints.
    test_blend = np.zeros_like(test_list[0], dtype=np.float64)
    for w, lg in zip(best_w, test_list):
        test_blend += w * lg
    test_blend += bias
    test_pred = test_blend.argmax(1)

    sub = pd.read_csv(args.sample_submission)
    sub["label"] = test_pred
    sub_path = out_dir / "submission_ensemble_refined_bias.csv"
    sub.to_csv(sub_path, index=False)

    before_cls.to_csv(analysis_dir / "ensemble_before_bias_class_report.csv", index=False)
    after_cls.to_csv(analysis_dir / "ensemble_after_bias_class_report.csv", index=False)
    before_conf.to_csv(analysis_dir / "ensemble_before_bias_top_confusions.csv", index=False)
    after_conf.to_csv(analysis_dir / "ensemble_after_bias_top_confusions.csv", index=False)

    with (analysis_dir / "refined_ensemble_weights.json").open("w", encoding="utf-8") as f:
        json.dump({a: float(w) for a, w in zip(aliases, best_w)}, f, ensure_ascii=False, indent=2)
    with (analysis_dir / "refined_ensemble_bias.json").open("w", encoding="utf-8") as f:
        json.dump({str(i): float(v) for i, v in enumerate(bias)}, f, ensure_ascii=False, indent=2)

    summary = {
        "aliases": aliases,
        "num_alias_models": len(aliases),
        "implied_checkpoints_used": len(aliases) * 5,
        "base_metrics_from_existing_weights": {
            "acc": base_metrics.acc,
            "f1_macro": base_metrics.f1_macro,
            "score": base_metrics.score,
        },
        "after_weight_search_metrics": {
            "acc": w_metrics.acc,
            "f1_macro": w_metrics.f1_macro,
            "score": w_metrics.score,
        },
        "after_bias_metrics": {
            "acc": bias_metrics.acc,
            "f1_macro": bias_metrics.f1_macro,
            "score": bias_metrics.score,
        },
        "focus_labels_for_bias": focus_labels,
        "focus_class_names_for_bias": [label_to_name[i] for i in focus_labels],
        "worst_classes_after_bias": after_cls.head(3)["class_name"].tolist(),
        "submission_path": str(sub_path),
    }
    with (analysis_dir / "refined_ensemble_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Refined Ensemble Done ===")
    print("Aliases:", aliases)
    print("Implied checkpoints used:", len(aliases) * 5)
    print("Base acc/f1:", f"{base_metrics.acc:.6f}", f"{base_metrics.f1_macro:.6f}")
    print("After weight-search acc/f1:", f"{w_metrics.acc:.6f}", f"{w_metrics.f1_macro:.6f}")
    print("After bias acc/f1:", f"{bias_metrics.acc:.6f}", f"{bias_metrics.f1_macro:.6f}")
    print("Worst classes before bias:", [label_to_name[i] for i in focus_labels])
    print("Submission:", sub_path)


if __name__ == "__main__":
    main()
