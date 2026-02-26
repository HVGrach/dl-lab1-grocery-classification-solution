#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldBundle:
    fold_idx: int
    model_names: List[str]
    probs_stack: np.ndarray  # [N, M, C]
    y_true: np.ndarray       # [N]
    image_ids: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOF-only bakeoff for meta-ensemble ideas (notebook-inspired) on saved val_probs."
    )
    p.add_argument("--zoo-root", type=Path, required=True, help="Directory with run_ranking.csv and referenced run dirs.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--folds", type=str, default="", help="Optional comma-separated fold ids (e.g. 0,1).")
    p.add_argument("--catboost-iterations", type=int, default=300)
    p.add_argument("--catboost-depth", type=int, default=4)
    p.add_argument("--catboost-lr", type=float, default=0.03)
    p.add_argument("--catboost-l2", type=float, default=5.0)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--classwise-alpha", type=float, default=0.2, help="Frequency adjustment exponent.")
    return p.parse_args()


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(probs.shape[1])))),
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def clip_and_norm(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum(axis=1, keepdims=True)
    return (p / (s + eps)).astype(np.float32)


def load_bundles(zoo_root: Path, selected_folds: Sequence[int] | None = None) -> List[FoldBundle]:
    rr_path = zoo_root / "run_ranking.csv"
    rr = pd.read_csv(rr_path)
    rr = rr[rr["status"] == "ok"].copy()
    rr["fold_idx"] = rr["fold_idx"].astype(int)
    if selected_folds:
        rr = rr[rr["fold_idx"].isin(list(selected_folds))].copy()
    if rr.empty:
        raise RuntimeError(f"No runs found in {rr_path}")

    bundles: List[FoldBundle] = []
    ref_order = None
    for fid in sorted(rr["fold_idx"].unique().tolist()):
        fr = rr[rr["fold_idx"] == fid].copy()
        fr["name"] = fr["name"].astype(str).str.replace(r"_f\d+$", "", regex=True)
        fr = fr.sort_values("name")
        model_names = fr["name"].astype(str).tolist()
        if ref_order is None:
            ref_order = model_names
        elif model_names != ref_order:
            raise RuntimeError(
                f"Model set/order mismatch across folds. fold0={ref_order}, fold{fid}={model_names}. "
                "Use --folds to restrict to common folds."
            )
        probs_list: List[np.ndarray] = []
        y_ref = None
        ids_ref = None
        for run_dir_str in fr["run_dir"].astype(str).tolist():
            run_dir = Path(run_dir_str)
            probs = np.load(run_dir / "val_probs.npy").astype(np.float32)
            y = np.load(run_dir / "val_labels.npy").astype(np.int64)
            pred_df = pd.read_csv(run_dir / "val_predictions.csv")
            ids = pred_df["image_id"].astype(str).tolist()
            if y_ref is None:
                y_ref = y
                ids_ref = ids
            else:
                if not np.array_equal(y_ref, y):
                    raise RuntimeError(f"val_labels mismatch in {run_dir}")
                if ids_ref != ids:
                    raise RuntimeError(f"val image_id order mismatch in {run_dir}")
            probs_list.append(probs)
        probs_stack = np.stack(probs_list, axis=1)  # [N, M, C]
        bundles.append(
            FoldBundle(
                fold_idx=int(fid),
                model_names=model_names,
                probs_stack=probs_stack,
                y_true=y_ref if y_ref is not None else np.empty(0, dtype=np.int64),
                image_ids=ids_ref or [],
            )
        )
    return bundles


def probs_features_basic(p_stack: np.ndarray) -> np.ndarray:
    # p_stack: [N, M, C]
    n, m, c = p_stack.shape
    flat = p_stack.reshape(n, m * c)
    p_sorted = np.sort(p_stack, axis=2)
    margin = p_sorted[:, :, -1] - p_sorted[:, :, -2]  # [N, M]
    entropy = -(p_stack * np.log(np.clip(p_stack, 1e-12, 1.0))).sum(axis=2)  # [N, M]
    maxp = p_stack.max(axis=2)
    mean_probs = p_stack.mean(axis=1)  # [N,C]
    return np.concatenate([flat, maxp, margin, entropy, mean_probs], axis=1).astype(np.float32)


def probs_features_advanced_notebook(p_stack: np.ndarray) -> np.ndarray:
    # Notebook-style features from probabilities.
    # p_stack [N,M,C] -> P_tensor [N,C,M]
    p_tensor = np.transpose(p_stack, (0, 2, 1))
    p_list = [p_stack[:, i, :] for i in range(p_stack.shape[1])]
    stack = np.hstack(p_list)
    mean_p = np.mean(p_tensor, axis=-1)
    max_p = np.max(p_tensor, axis=-1)
    min_p = np.min(p_tensor, axis=-1)
    std_p = np.std(p_tensor, axis=-1)
    entropy = -np.sum(mean_p * np.log(np.clip(mean_p, 1e-12, 1.0)), axis=1, keepdims=True)
    sorted_mean = np.sort(mean_p, axis=1)
    margin = (sorted_mean[:, -1] - sorted_mean[:, -2]).reshape(-1, 1)

    pairwise = []
    for i in range(len(p_list)):
        for j in range(i + 1, len(p_list)):
            pairwise.append(np.abs(p_list[i] - p_list[j]))
    if pairwise:
        pairwise_block = np.hstack(pairwise)
        return np.hstack([stack, mean_p, max_p, min_p, std_p, entropy, margin, pairwise_block]).astype(np.float32)
    return np.hstack([stack, mean_p, max_p, min_p, std_p, entropy, margin]).astype(np.float32)


def fit_lr_mse_weights(prob_list: List[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    n_models = len(prob_list)
    prob_stack = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list], axis=-1)  # [N,C,M]
    n_samples, n_classes, _ = prob_stack.shape
    y_idx = np.asarray(y_true, dtype=np.int64)
    y_onehot = np.eye(n_classes, dtype=np.float64)[y_idx]
    X = prob_stack.reshape(n_samples * n_classes, n_models)
    y = y_onehot.reshape(n_samples * n_classes)
    try:
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y)
        w = np.asarray(reg.coef_, dtype=np.float64).reshape(-1)
    except TypeError:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        w = np.clip(np.asarray(reg.coef_, dtype=np.float64).reshape(-1), 0.0, None)
    w = np.clip(w, 0.0, None)
    if not np.all(np.isfinite(w)) or float(w.sum()) <= 0:
        w = np.ones(n_models, dtype=np.float64) / n_models
    else:
        w = w / float(w.sum())
    return w.astype(np.float32)


def oof_eval_per_fold_lr_mse(bundles: List[FoldBundle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_all, y_all, fold_ids = [], [], []
    for b in bundles:
        prob_list = [b.probs_stack[:, i, :] for i in range(b.probs_stack.shape[1])]
        w = fit_lr_mse_weights(prob_list, b.y_true)
        blended = np.zeros_like(prob_list[0], dtype=np.float64)
        for wi, pi in zip(w, prob_list):
            blended += float(wi) * pi
        probs_all.append(blended.astype(np.float32))
        y_all.append(b.y_true)
        fold_ids.append(np.full(len(b.y_true), b.fold_idx, dtype=np.int64))
    return np.concatenate(probs_all), np.concatenate(y_all), np.concatenate(fold_ids)


def fit_classwise_ridge(P_list: List[np.ndarray], y: np.ndarray, train_mask: np.ndarray, alpha: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    # P_list: list of [N,C]
    m = len(P_list)
    c = P_list[0].shape[1]
    X = np.stack(P_list, axis=-1)  # [N,C,M]

    y_fit = y[train_mask]
    class_freq = np.bincount(y_fit, minlength=c).astype(np.float64)
    class_freq = np.maximum(class_freq, 1.0)
    class_freq /= class_freq.sum()

    W = np.zeros((c, m), dtype=np.float64)
    for cls in range(c):
        Xc = X[train_mask, cls, :]
        yc = (y_fit == cls).astype(np.float64)
        Xc_adj = Xc / (class_freq[cls] ** alpha)
        reg = Ridge(alpha=lam, fit_intercept=False)
        reg.fit(Xc_adj, yc)
        w = np.clip(reg.coef_, 0.0, None)
        s = float(w.sum())
        W[cls] = w / s if s > 0 else (np.ones(m, dtype=np.float64) / m)
    return W, class_freq


def predict_classwise(P_list: List[np.ndarray], W: np.ndarray, class_freq: np.ndarray, alpha: float) -> np.ndarray:
    X = np.stack(P_list, axis=-1)  # [N,C,M]
    X_adj = X / (class_freq[None, :, None] ** alpha)
    blended = np.einsum("ncm,cm->nc", X_adj, W)
    return clip_and_norm(blended)


def oof_meta_cv(
    bundles: List[FoldBundle],
    method_name: str,
    X_builder,
    fit_predict_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[str, float]]]:
    X_list = []
    y_list = []
    fold_ids = []
    for b in bundles:
        X_list.append(X_builder(b.probs_stack))
        y_list.append(b.y_true)
        fold_ids.append(np.full(len(b.y_true), b.fold_idx, dtype=np.int64))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    fids = np.concatenate(fold_ids, axis=0)
    n_classes = int(np.max(y) + 1)
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold = {}
    for fid in sorted(np.unique(fids).tolist()):
        tr = fids != fid
        va = fids == fid
        oof_prob[va] = fit_predict_fn(X[tr], y[tr], X[va], y[va], int(fid))
        per_fold[int(fid)] = evaluate_probs(y[va], oof_prob[va])
        print(f"[{method_name}] fold={fid} acc={per_fold[int(fid)]['acc']:.5f} f1={per_fold[int(fid)]['f1_macro']:.5f}", flush=True)
    return oof_prob, y, fids, per_fold


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds = None
    if args.folds.strip():
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    bundles = load_bundles(args.zoo_root, selected_folds=folds)
    manifest = {
        "zoo_root": str(args.zoo_root),
        "folds": [int(b.fold_idx) for b in bundles],
        "model_order": bundles[0].model_names,
        "num_models": int(len(bundles[0].model_names)),
        "num_samples": int(sum(len(b.y_true) for b in bundles)),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Loaded folds:", manifest["folds"], "models:", manifest["model_order"], flush=True)

    ranking_rows = []
    saved_probs: Dict[str, np.ndarray] = {}

    # 1) Old default idea: per-fold LR(MSE) blending
    probs_lr_mse, y_all, fold_ids_all = oof_eval_per_fold_lr_mse(bundles)
    m_lr_mse = evaluate_probs(y_all, probs_lr_mse)
    ranking_rows.append({"method": "lr_mse_per_fold", **m_lr_mse})
    saved_probs["lr_mse_per_fold"] = probs_lr_mse
    print(f"[lr_mse_per_fold] acc={m_lr_mse['acc']:.5f} f1={m_lr_mse['f1_macro']:.5f}", flush=True)

    # 2) Notebook-ish logreg on raw probs (honest CV)
    def _fit_logreg(Xtr, ytr, Xva, _yva, fid):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=10.0, max_iter=2000, random_state=42 + fid)),
        ])
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva).astype(np.float32)

    def _raw_prob_flat(p_stack):
        n, m, c = p_stack.shape
        return p_stack.reshape(n, m * c).astype(np.float32)

    probs_lr_raw, y_eval, fids_eval, per_fold_lr = oof_meta_cv(
        bundles, "logreg_raw_probs", _raw_prob_flat, _fit_logreg
    )
    m_lr_raw = evaluate_probs(y_eval, probs_lr_raw)
    ranking_rows.append({"method": "logreg_raw_probs_cv", **m_lr_raw})
    saved_probs["logreg_raw_probs_cv"] = probs_lr_raw

    # 3) CatBoost on advanced notebook-style probability features
    try:
        from catboost import CatBoostClassifier

        def _fit_cb_adv(Xtr, ytr, Xva, _yva, fid):
            cb = CatBoostClassifier(
                iterations=args.catboost_iterations,
                learning_rate=args.catboost_lr,
                depth=args.catboost_depth,
                l2_leaf_reg=args.catboost_l2,
                loss_function="MultiClass",
                eval_metric="Accuracy",
                random_seed=42 + fid,
                verbose=False,
                allow_writing_files=False,
            )
            cb.fit(Xtr, ytr)
            return cb.predict_proba(Xva).astype(np.float32)

        probs_cb_adv, y_eval, fids_eval, per_fold_cb = oof_meta_cv(
            bundles, "catboost_adv_feats", probs_features_advanced_notebook, _fit_cb_adv
        )
        m_cb_adv = evaluate_probs(y_eval, probs_cb_adv)
        ranking_rows.append({"method": "catboost_adv_feats_cv", **m_cb_adv})
        saved_probs["catboost_adv_feats_cv"] = probs_cb_adv
    except ImportError:
        print("[catboost_adv_feats] skipped: catboost not installed", flush=True)
        probs_cb_adv = None

    # 4) Class-wise ridge (notebook idea), honest CV by folds
    y_all_list = []
    f_all_list = []
    oof_cwr_parts = []
    per_fold_cwr = {}
    # Build concatenated per-model arrays aligned over all selected folds
    P_full = [np.concatenate([b.probs_stack[:, i, :] for b in bundles], axis=0) for i in range(bundles[0].probs_stack.shape[1])]
    y_concat = np.concatenate([b.y_true for b in bundles], axis=0)
    f_concat = np.concatenate([np.full(len(b.y_true), b.fold_idx, dtype=np.int64) for b in bundles], axis=0)
    for fid in sorted(np.unique(f_concat).tolist()):
        tr = f_concat != fid
        va = f_concat == fid
        W_f, freq_f = fit_classwise_ridge(P_full, y_concat, tr, alpha=args.classwise_alpha, lam=args.ridge_alpha)
        p_va = [p[va] for p in P_full]
        preds_va = predict_classwise(p_va, W_f, freq_f, alpha=args.classwise_alpha)
        per_fold_cwr[int(fid)] = evaluate_probs(y_concat[va], preds_va)
        print(f"[classwise_ridge] fold={fid} acc={per_fold_cwr[int(fid)]['acc']:.5f} f1={per_fold_cwr[int(fid)]['f1_macro']:.5f}", flush=True)
        oof_cwr_parts.append((va, preds_va))
    oof_cwr = np.zeros((len(y_concat), P_full[0].shape[1]), dtype=np.float32)
    for mask, p in oof_cwr_parts:
        oof_cwr[mask] = p
    m_cwr = evaluate_probs(y_concat, oof_cwr)
    ranking_rows.append({"method": "classwise_ridge_cv", **m_cwr})
    saved_probs["classwise_ridge_cv"] = oof_cwr

    # 5) Blend classwise ridge + CatBoost (if available), optimize on OOF via grid (still honest because both are OOF)
    if probs_cb_adv is not None:
        best = None
        for w in np.linspace(0.0, 1.0, 101):
            blend = (w * probs_cb_adv + (1.0 - w) * oof_cwr).astype(np.float32)
            m = evaluate_probs(y_concat, blend)
            if best is None or (m["acc"], m["f1_macro"]) > (best[1]["acc"], best[1]["f1_macro"]):
                best = (float(w), m, blend)
        assert best is not None
        w_best, m_best, blend_best = best
        ranking_rows.append({"method": "blend_cbadv_cwr_cv", "blend_w_catboost": w_best, **m_best})
        saved_probs["blend_cbadv_cwr_cv"] = blend_best
        print(f"[blend_cbadv_cwr] best_w_catboost={w_best:.2f} acc={m_best['acc']:.5f} f1={m_best['f1_macro']:.5f}", flush=True)

    rank_df = pd.DataFrame(ranking_rows).sort_values(["acc", "f1_macro"], ascending=False)
    rank_df.to_csv(args.out_dir / "oof_methods_ranking.csv", index=False)
    for name, probs in saved_probs.items():
        np.save(args.out_dir / f"{name}_oof_probs.npy", probs.astype(np.float32))
    summary = {
        "manifest": manifest,
        "results": rank_df.to_dict(orient="records"),
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nOOF ranking:")
    print(rank_df.to_string(index=False), flush=True)
    print(f"\nSaved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
