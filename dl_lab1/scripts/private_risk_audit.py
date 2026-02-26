#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--meta-folds", type=int, default=5)
    p.add_argument("--bootstrap-runs", type=int, default=1200)
    p.add_argument("--holdout-runs", type=int, default=20)
    p.add_argument("--group-runs", type=int, default=15)
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


def load_features(outputs_dir: Path, pair_dir: Path, train_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
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
    y_true = np.load(outputs_dir / aliases[0] / "oof_targets.npy")

    blend = np.zeros_like(oof_logits[aliases[0]], dtype=np.float64)
    for a in aliases:
        blend += float(weights[a]) * oof_logits[a]
    if refined_b.exists():
        bias_map = json.loads(refined_b.read_text(encoding="utf-8"))
        bias = np.array([float(bias_map.get(str(i), 0.0)) for i in range(blend.shape[1])], dtype=np.float64)
    else:
        bias = np.zeros(blend.shape[1], dtype=np.float64)
    blend += bias

    prob = softmax(blend)
    top2 = np.argsort(-prob, axis=1)[:, :2]
    margin = (prob[np.arange(len(prob)), top2[:, 0]] - prob[np.arange(len(prob)), top2[:, 1]])[:, None]

    feat = [
        oof_logits[aliases[0]],
        oof_logits[aliases[1]],
        oof_logits[aliases[2]],
        blend,
        prob,
        margin,
    ]

    train_df = pd.read_csv(train_csv)[["image_id"]]
    for alias in ["kiwi_vs_potato", "redapple_vs_tomato", "mandarin_vs_orange"]:
        sp = pd.read_csv(pair_dir / alias / "pair_train_split.csv")[["image_id"]]
        sp["p"] = np.load(pair_dir / alias / "oof_prob.npy")
        full = train_df.merge(sp, on="image_id", how="left", sort=False)["p"].fillna(0.5).to_numpy(dtype=np.float32)
        feat.append(full[:, None])

    X = np.concatenate(feat, axis=1).astype(np.float32)
    X = np.concatenate([X, np.abs(X[:, -3:] - 0.5)], axis=1).astype(np.float32)
    return X, y_true


def quant(v: np.ndarray, p: float) -> float:
    return float(np.quantile(v, p))


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    train_csv = Path(args.train_csv)
    pair_dir = Path(args.pair_dir)

    X, y = load_features(outputs_dir, pair_dir, train_csv)
    params = dict(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        l2_regularization=args.l2_regularization,
    )

    # 1) CV OOF estimate for chosen meta model.
    skf = StratifiedKFold(n_splits=args.meta_folds, shuffle=True, random_state=args.seed)
    n_classes = len(np.unique(y))
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = HistGradientBoostingClassifier(random_state=args.seed + fold, **params)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_prob[va_idx] = clf.predict_proba(X[va_idx])
        print(f"[audit] meta fold {fold}/{args.meta_folds} done", flush=True)
    oof_pred = oof_prob.argmax(1)
    cv_acc = float(accuracy_score(y, oof_pred))
    cv_f1 = float(f1_score(y, oof_pred, average="macro"))

    # 2) Bootstrap confidence over OOF predictions.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y))
    boot_acc = np.empty(args.bootstrap_runs, dtype=np.float64)
    boot_f1 = np.empty(args.bootstrap_runs, dtype=np.float64)
    for i in range(args.bootstrap_runs):
        s = rng.choice(idx, size=len(idx), replace=True)
        boot_acc[i] = accuracy_score(y[s], oof_pred[s])
        boot_f1[i] = f1_score(y[s], oof_pred[s], average="macro")
    print("[audit] bootstrap done", flush=True)

    # 3) Repeated stratified holdout (80/20).
    sss = StratifiedShuffleSplit(n_splits=args.holdout_runs, test_size=0.2, random_state=args.seed + 100)
    hold_acc: List[float] = []
    hold_f1: List[float] = []
    for i, (tr_idx, te_idx) in enumerate(sss.split(X, y), 1):
        clf = HistGradientBoostingClassifier(random_state=args.seed + 1000 + i, **params)
        clf.fit(X[tr_idx], y[tr_idx])
        p = clf.predict(X[te_idx])
        hold_acc.append(float(accuracy_score(y[te_idx], p)))
        hold_f1.append(float(f1_score(y[te_idx], p, average="macro")))
    hold_acc = np.array(hold_acc)
    hold_f1 = np.array(hold_f1)
    print("[audit] repeated holdout done", flush=True)

    # 4) Group holdout by PLU (harder shift).
    full_train = pd.read_csv(train_csv)
    groups = full_train["image_id"].str.split("/").str[1].astype(str).to_numpy()
    grp_acc: List[float] = []
    grp_f1: List[float] = []
    for i in range(args.group_runs):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + 2000 + i)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        if len(np.unique(y[tr_idx])) < len(np.unique(y)):
            continue
        clf = HistGradientBoostingClassifier(random_state=args.seed + 3000 + i, **params)
        clf.fit(X[tr_idx], y[tr_idx])
        p = clf.predict(X[te_idx])
        grp_acc.append(float(accuracy_score(y[te_idx], p)))
        grp_f1.append(float(f1_score(y[te_idx], p, average="macro")))
    grp_acc = np.array(grp_acc)
    grp_f1 = np.array(grp_f1)
    print("[audit] group holdout done", flush=True)

    report = {
        "meta_cv": {
            "acc": cv_acc,
            "f1_macro": cv_f1,
            "score": cv_acc + args.score_f1_weight * cv_f1,
        },
        "bootstrap_oof": {
            "runs": int(args.bootstrap_runs),
            "acc_p01": quant(boot_acc, 0.01),
            "acc_p05": quant(boot_acc, 0.05),
            "acc_p50": quant(boot_acc, 0.50),
            "f1_p05": quant(boot_f1, 0.05),
            "f1_p50": quant(boot_f1, 0.50),
        },
        "repeated_stratified_holdout": {
            "runs": int(len(hold_acc)),
            "acc_min": float(hold_acc.min()),
            "acc_p05": quant(hold_acc, 0.05),
            "acc_median": quant(hold_acc, 0.50),
            "f1_min": float(hold_f1.min()),
            "f1_p05": quant(hold_f1, 0.05),
            "f1_median": quant(hold_f1, 0.50),
            "runs_below_0_85_acc": int((hold_acc < 0.85).sum()),
        },
        "group_holdout_by_plu": {
            "runs": int(len(grp_acc)),
            "acc_min": float(grp_acc.min()) if len(grp_acc) else None,
            "acc_p05": quant(grp_acc, 0.05) if len(grp_acc) else None,
            "acc_median": quant(grp_acc, 0.50) if len(grp_acc) else None,
            "f1_min": float(grp_f1.min()) if len(grp_f1) else None,
            "f1_p05": quant(grp_f1, 0.05) if len(grp_f1) else None,
            "f1_median": quant(grp_f1, 0.50) if len(grp_f1) else None,
            "runs_below_0_85_acc": int((grp_acc < 0.85).sum()) if len(grp_acc) else None,
        },
    }

    out_path = analysis_dir / "private_risk_audit.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] saved: {out_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

