#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outputs-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps",
    )
    p.add_argument(
        "--pair-dir",
        type=str,
        default="/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_pair_experts_mps",
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
    p.add_argument("--meta-folds", type=int, default=5)
    p.add_argument("--score-f1-weight", type=float, default=0.15)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--l2-regularization", type=float, default=0.02)
    return p.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def metrics(y_true: np.ndarray, pred: np.ndarray, score_f1_weight: float) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, pred))
    f1m = float(f1_score(y_true, pred, average="macro"))
    return {"acc": acc, "f1_macro": f1m, "score": acc + score_f1_weight * f1m}


def load_base_blend(outputs_dir: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    analysis_dir = outputs_dir / "analysis"
    refined_w = analysis_dir / "refined_ensemble_weights.json"
    refined_b = analysis_dir / "refined_ensemble_bias.json"
    default_w = outputs_dir / "ensemble_weights.json"

    if refined_w.exists():
        weights = json.loads(refined_w.read_text(encoding="utf-8"))
    else:
        weights = json.loads(default_w.read_text(encoding="utf-8"))
    aliases = list(weights.keys())

    oof_logits = {a: np.load(outputs_dir / a / "oof_logits.npy") for a in aliases}
    test_logits = {a: np.load(outputs_dir / a / "test_logits.npy") for a in aliases}
    y_true = np.load(outputs_dir / aliases[0] / "oof_targets.npy")

    blend_oof = np.zeros_like(oof_logits[aliases[0]], dtype=np.float64)
    blend_test = np.zeros_like(test_logits[aliases[0]], dtype=np.float64)
    for a in aliases:
        w = float(weights[a])
        blend_oof += w * oof_logits[a]
        blend_test += w * test_logits[a]

    if refined_b.exists():
        bias_map = json.loads(refined_b.read_text(encoding="utf-8"))
        bias = np.array([float(bias_map.get(str(i), 0.0)) for i in range(blend_oof.shape[1])], dtype=np.float64)
    else:
        bias = np.zeros(blend_oof.shape[1], dtype=np.float64)
    blend_oof += bias
    blend_test += bias

    # Base feature block: all alias logits + refined blend logits + refined blend probabilities + margin.
    base_feat_oof = [oof_logits[a] for a in aliases] + [blend_oof, softmax(blend_oof)]
    base_feat_test = [test_logits[a] for a in aliases] + [blend_test, softmax(blend_test)]
    prob_oof = softmax(blend_oof)
    prob_test = softmax(blend_test)
    top2_oof = np.argsort(-prob_oof, axis=1)[:, :2]
    top2_test = np.argsort(-prob_test, axis=1)[:, :2]
    margin_oof = (
        prob_oof[np.arange(len(prob_oof)), top2_oof[:, 0]]
        - prob_oof[np.arange(len(prob_oof)), top2_oof[:, 1]]
    )[:, None]
    margin_test = (
        prob_test[np.arange(len(prob_test)), top2_test[:, 0]]
        - prob_test[np.arange(len(prob_test)), top2_test[:, 1]]
    )[:, None]
    base_feat_oof.append(margin_oof)
    base_feat_test.append(margin_test)

    X_oof = np.concatenate(base_feat_oof, axis=1).astype(np.float32)
    X_test = np.concatenate(base_feat_test, axis=1).astype(np.float32)
    return aliases, y_true, X_oof, X_test


def add_pair_features(
    X_oof: np.ndarray,
    X_test: np.ndarray,
    train_csv: Path,
    pair_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    train_df = pd.read_csv(train_csv)[["image_id"]]
    pair_aliases = ["kiwi_vs_potato", "redapple_vs_tomato", "mandarin_vs_orange"]
    pair_oof_cols = []
    pair_test_cols = []
    for alias in pair_aliases:
        split_path = pair_dir / alias / "pair_train_split.csv"
        oof_path = pair_dir / alias / "oof_prob.npy"
        test_path = pair_dir / alias / "test_prob.npy"
        if not (split_path.exists() and oof_path.exists() and test_path.exists()):
            raise FileNotFoundError(f"Missing pair-expert artifacts for {alias} in {pair_dir}")

        pair_split = pd.read_csv(split_path)[["image_id"]]
        pair_split["pair_prob"] = np.load(oof_path)
        full_oof = train_df.merge(pair_split, on="image_id", how="left", sort=False)["pair_prob"].fillna(0.5).to_numpy(dtype=np.float32)
        full_test = np.load(test_path).astype(np.float32)
        pair_oof_cols.append(full_oof[:, None])
        pair_test_cols.append(full_test[:, None])

    pair_oof = np.concatenate(pair_oof_cols, axis=1)
    pair_test = np.concatenate(pair_test_cols, axis=1)
    pair_conf_oof = np.abs(pair_oof - 0.5)
    pair_conf_test = np.abs(pair_test - 0.5)

    X_oof2 = np.concatenate([X_oof, pair_oof, pair_conf_oof], axis=1)
    X_test2 = np.concatenate([X_test, pair_test, pair_conf_test], axis=1)
    return X_oof2.astype(np.float32), X_test2.astype(np.float32)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    aliases, y_true, X_oof, X_test = load_base_blend(outputs_dir)
    X_oof, X_test = add_pair_features(
        X_oof=X_oof,
        X_test=X_test,
        train_csv=Path(args.train_csv),
        pair_dir=Path(args.pair_dir),
    )

    # Base (refined blend) metric for comparison.
    ref_summary = json.loads((analysis_dir / "refined_ensemble_summary.json").read_text(encoding="utf-8"))
    base_acc = float(ref_summary["after_bias_metrics"]["acc"])
    base_f1 = float(ref_summary["after_bias_metrics"]["f1_macro"])

    # CV meta evaluation.
    classes = np.unique(y_true)
    n_classes = len(classes)
    skf = StratifiedKFold(n_splits=args.meta_folds, shuffle=True, random_state=args.seed)
    meta_oof_prob = np.zeros((len(y_true), n_classes), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y_true)):
        print(f"[meta_hgb] fold {fold+1}/{args.meta_folds} train...", flush=True)
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            l2_regularization=args.l2_regularization,
            random_state=args.seed + fold,
        )
        clf.fit(X_oof[tr_idx], y_true[tr_idx])
        meta_oof_prob[va_idx] = clf.predict_proba(X_oof[va_idx])
        pred_fold = meta_oof_prob[va_idx].argmax(1)
        fold_acc = accuracy_score(y_true[va_idx], pred_fold)
        print(f"[meta_hgb] fold {fold+1}/{args.meta_folds} done, fold_acc={fold_acc:.5f}", flush=True)

    meta_oof_pred = meta_oof_prob.argmax(1)
    meta_metrics = metrics(y_true, meta_oof_pred, args.score_f1_weight)

    # Fit on full OOF features and predict test.
    final_clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        l2_regularization=args.l2_regularization,
        random_state=args.seed,
    )
    final_clf.fit(X_oof, y_true)
    test_pred = final_clf.predict(X_test).astype(int)

    sub = pd.read_csv(args.sample_submission)
    sub["label"] = test_pred
    sub_path = outputs_dir / "submission_meta_hgb_pair_stack.csv"
    sub.to_csv(sub_path, index=False)

    # Confusion report.
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

    cm = confusion_matrix(y_true, meta_oof_pred)
    rows = []
    for i in range(cm.shape[0]):
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
                    "row_share": float(cm[i, j] / (cm[i].sum() + 1e-12)),
                }
            )
    pd.DataFrame(rows).sort_values(["count", "row_share"], ascending=False).head(60).to_csv(
        analysis_dir / "meta_hgb_top_confusions.csv",
        index=False,
    )

    summary = {
        "aliases": aliases,
        "features_dim": int(X_oof.shape[1]),
        "meta_model": "HistGradientBoostingClassifier",
        "meta_params": {
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "l2_regularization": args.l2_regularization,
            "meta_folds": args.meta_folds,
            "seed": args.seed,
        },
        "base_refined_metrics": {
            "acc": base_acc,
            "f1_macro": base_f1,
            "score": base_acc + args.score_f1_weight * base_f1,
        },
        "meta_cv_metrics": meta_metrics,
        "delta_acc_abs": meta_metrics["acc"] - base_acc,
        "delta_f1_abs": meta_metrics["f1_macro"] - base_f1,
        "submission_path": str(sub_path),
    }
    with (analysis_dir / "meta_hgb_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Meta HGB Stack Done ===")
    print("Base acc/f1:", f"{base_acc:.6f}", f"{base_f1:.6f}")
    print("Meta CV acc/f1:", f"{meta_metrics['acc']:.6f}", f"{meta_metrics['f1_macro']:.6f}")
    print("Delta acc:", f"{summary['delta_acc_abs']:.6f}")
    print("Submission:", sub_path)


if __name__ == "__main__":
    main()
