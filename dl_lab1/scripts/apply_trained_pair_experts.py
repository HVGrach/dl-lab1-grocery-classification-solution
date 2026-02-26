#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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
    p.add_argument("--margin-grid", type=str, default="0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30")
    p.add_argument("--prob-delta-grid", type=str, default="-0.10,-0.05,0.00,0.05,0.10")
    p.add_argument("--score-f1-weight", type=float, default=0.15)
    return p.parse_args()


def parse_float_grid(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def score(y_true: np.ndarray, pred: np.ndarray, score_f1_weight: float) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, pred))
    f1m = float(f1_score(y_true, pred, average="macro"))
    return {"acc": acc, "f1_macro": f1m, "score": acc + score_f1_weight * f1m}


def load_base_blend(outputs_dir: Path) -> Dict:
    analysis_dir = outputs_dir / "analysis"
    refined_w = analysis_dir / "refined_ensemble_weights.json"
    refined_b = analysis_dir / "refined_ensemble_bias.json"
    default_w = outputs_dir / "ensemble_weights.json"

    if refined_w.exists():
        weights = json.loads(refined_w.read_text(encoding="utf-8"))
    else:
        weights = json.loads(default_w.read_text(encoding="utf-8"))
    aliases = list(weights.keys())

    oof_list = [np.load(outputs_dir / a / "oof_logits.npy") for a in aliases]
    test_list = [np.load(outputs_dir / a / "test_logits.npy") for a in aliases]
    y_true = np.load(outputs_dir / aliases[0] / "oof_targets.npy")

    blend_oof = np.zeros_like(oof_list[0], dtype=np.float64)
    blend_test = np.zeros_like(test_list[0], dtype=np.float64)
    for a, oof, tst in zip(aliases, oof_list, test_list):
        w = float(weights[a])
        blend_oof += w * oof
        blend_test += w * tst

    if refined_b.exists():
        bias_map = json.loads(refined_b.read_text(encoding="utf-8"))
        bias = np.array([float(bias_map.get(str(i), 0.0)) for i in range(blend_oof.shape[1])], dtype=np.float64)
    else:
        bias = np.zeros(blend_oof.shape[1], dtype=np.float64)
    blend_oof += bias
    blend_test += bias

    return {
        "aliases": aliases,
        "weights": weights,
        "y_true": y_true,
        "blend_oof": blend_oof,
        "blend_test": blend_test,
    }


def apply_one_pair(
    pred: np.ndarray,
    top2: np.ndarray,
    margins: np.ndarray,
    pair_prob: np.ndarray,
    a: int,
    b: int,
    margin_thr: float,
    prob_thr: float,
) -> np.ndarray:
    pair_top2 = ((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a))
    valid_prob = ~np.isnan(pair_prob)
    mask = pair_top2 & (margins <= margin_thr) & valid_prob
    out = pred.copy()
    out[mask] = np.where(pair_prob[mask] >= prob_thr, a, b)
    return out


def load_pair_prob_as_full(train_df: pd.DataFrame, pair_dir: Path) -> Dict:
    pair_split = pd.read_csv(pair_dir / "pair_train_split.csv")
    oof_prob = np.load(pair_dir / "oof_prob.npy")
    test_prob = np.load(pair_dir / "test_prob.npy")
    metrics = json.loads((pair_dir / "metrics.json").read_text(encoding="utf-8"))

    if len(pair_split) != len(oof_prob):
        raise RuntimeError(f"Length mismatch in {pair_dir.name}: pair_train_split vs oof_prob.")

    pair_prob_df = pair_split[["image_id"]].copy()
    pair_prob_df["pair_oof_prob"] = oof_prob
    merged = train_df[["image_id"]].merge(pair_prob_df, on="image_id", how="left", sort=False)
    if len(merged) != len(train_df):
        raise RuntimeError(f"Merge length mismatch for {pair_dir.name}.")
    full_oof_prob = merged["pair_oof_prob"].to_numpy(dtype=np.float64)

    return {
        "full_oof_prob": full_oof_prob,
        "test_prob": test_prob.astype(np.float64),
        "best_thr": float(metrics["best_thr"]),
    }


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    pair_root = Path(args.pair_dir)
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    margin_grid = parse_float_grid(args.margin_grid)
    prob_delta_grid = parse_float_grid(args.prob_delta_grid)

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

    base = load_base_blend(outputs_dir)
    y_true = base["y_true"]
    blend_oof = base["blend_oof"]
    blend_test = base["blend_test"]

    base_prob_oof = softmax(blend_oof)
    base_pred_oof = base_prob_oof.argmax(1)
    top2_oof = np.argsort(-base_prob_oof, axis=1)[:, :2]
    margins_oof = base_prob_oof[np.arange(len(base_prob_oof)), top2_oof[:, 0]] - base_prob_oof[np.arange(len(base_prob_oof)), top2_oof[:, 1]]

    base_prob_test = softmax(blend_test)
    base_pred_test = base_prob_test.argmax(1)
    top2_test = np.argsort(-base_prob_test, axis=1)[:, :2]
    margins_test = base_prob_test[np.arange(len(base_prob_test)), top2_test[:, 0]] - base_prob_test[np.arange(len(base_prob_test)), top2_test[:, 1]]

    base_metrics = score(y_true, base_pred_oof, args.score_f1_weight)

    pair_summary = json.loads((pair_root / "pair_experts_summary.json").read_text(encoding="utf-8"))
    # Use fixed order: hardest pairs first.
    priority = {"kiwi_vs_potato": 0, "redapple_vs_tomato": 1, "mandarin_vs_orange": 2}
    pair_summary = sorted(pair_summary, key=lambda x: priority.get(x["alias"], 99))

    current_oof_pred = base_pred_oof.copy()
    tuned_cfg = []
    for item in pair_summary:
        alias = item["alias"]
        a = int(item["pos_label"])
        b = int(item["neg_label"])
        pair_data = load_pair_prob_as_full(train_df=train_df, pair_dir=pair_root / alias)
        oof_prob = pair_data["full_oof_prob"]
        best_thr = pair_data["best_thr"]

        best_local = {
            "margin_thr": 0.0,
            "prob_thr": best_thr,
            "score": score(y_true, current_oof_pred, args.score_f1_weight)["score"],
            "pred": current_oof_pred,
            "affected_oof_rows": 0,
        }
        for mthr in margin_grid:
            for d in prob_delta_grid:
                pthr = float(np.clip(best_thr + d, 0.05, 0.95))
                cand_pred = apply_one_pair(
                    pred=current_oof_pred,
                    top2=top2_oof,
                    margins=margins_oof,
                    pair_prob=oof_prob,
                    a=a,
                    b=b,
                    margin_thr=mthr,
                    prob_thr=pthr,
                )
                s = score(y_true, cand_pred, args.score_f1_weight)["score"]
                if s > best_local["score"] + 1e-12:
                    pair_top2 = ((top2_oof[:, 0] == a) & (top2_oof[:, 1] == b)) | ((top2_oof[:, 0] == b) & (top2_oof[:, 1] == a))
                    affected = int((pair_top2 & (margins_oof <= mthr) & ~np.isnan(oof_prob)).sum())
                    best_local = {
                        "margin_thr": float(mthr),
                        "prob_thr": pthr,
                        "score": float(s),
                        "pred": cand_pred,
                        "affected_oof_rows": affected,
                    }

        current_oof_pred = best_local["pred"]
        tuned_cfg.append(
            {
                "alias": alias,
                "pair": [a, b],
                "pair_names": [label_to_name[a], label_to_name[b]],
                "expert_best_thr": best_thr,
                "used_prob_thr": best_local["prob_thr"],
                "used_margin_thr": best_local["margin_thr"],
                "affected_oof_rows": best_local["affected_oof_rows"],
            }
        )

    final_oof_pred = current_oof_pred
    final_metrics = score(y_true, final_oof_pred, args.score_f1_weight)

    # Apply tuned config on test.
    current_test_pred = base_pred_test.copy()
    test_pair_stats = []
    for cfg in tuned_cfg:
        alias = cfg["alias"]
        a, b = cfg["pair"]
        pair_test_prob = np.load(pair_root / alias / "test_prob.npy").astype(np.float64)
        current_test_pred = apply_one_pair(
            pred=current_test_pred,
            top2=top2_test,
            margins=margins_test,
            pair_prob=pair_test_prob,
            a=int(a),
            b=int(b),
            margin_thr=float(cfg["used_margin_thr"]),
            prob_thr=float(cfg["used_prob_thr"]),
        )
        pair_top2 = ((top2_test[:, 0] == a) & (top2_test[:, 1] == b)) | ((top2_test[:, 0] == b) & (top2_test[:, 1] == a))
        affected_test = int((pair_top2 & (margins_test <= float(cfg["used_margin_thr"]))).sum())
        test_pair_stats.append(
            {
                "alias": alias,
                "pair": [int(a), int(b)],
                "pair_names": cfg["pair_names"],
                "affected_test_rows": affected_test,
            }
        )

    sub = pd.read_csv(args.sample_submission)
    sub["label"] = current_test_pred
    sub_path = outputs_dir / "submission_ensemble_refined_pair_experts.csv"
    sub.to_csv(sub_path, index=False)

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
        analysis_dir / "pair_experts_top_confusions.csv",
        index=False,
    )

    summary = {
        "base_metrics_before_pair_experts": base_metrics,
        "metrics_after_pair_experts": final_metrics,
        "pair_config": tuned_cfg,
        "pair_test_stats": test_pair_stats,
        "submission_path": str(sub_path),
    }
    with (analysis_dir / "pair_experts_integration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Pair Experts Integration Done ===")
    print("Base acc/f1:", f"{base_metrics['acc']:.6f}", f"{base_metrics['f1_macro']:.6f}")
    print("After pair-experts acc/f1:", f"{final_metrics['acc']:.6f}", f"{final_metrics['f1_macro']:.6f}")
    print("Submission:", sub_path)


if __name__ == "__main__":
    main()

