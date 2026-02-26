#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed orchestrator: old (pre-clean) + new (post-clean) model-zoo via OOF-aligned blending.")
    p.add_argument("--base", type=str, required=True, help="top_new_dataset root")
    p.add_argument("--folds-csv", type=str, required=True, help="top_new_dataset folds csv (image_id,label,fold)")
    p.add_argument("--old-zoo-root", type=str, required=True)
    p.add_argument("--new-zoo-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="mps", choices=["mps", "cpu", "auto"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tta-mode", type=str, default="geo8", choices=["none", "flip", "geo4", "geo8"])
    p.add_argument("--tta-views", type=int, default=0)
    p.add_argument("--reuse-test-cache", action="store_true")
    p.add_argument("--save-test-cache", action="store_true")
    p.add_argument("--old-models", type=str, default="cnn_convnext_small_sam_swa,cnn_effnetv2_s_sam_swa,vit_deit3_small_color_safe")
    p.add_argument("--new-models", type=str, default="cnn_convnext_small_sam_swa_ls03_swa5,cnn_effnetv2_s_sam_swa_ls03_swa5,vit_deit3_small_safe")
    p.add_argument("--fold-agg", type=str, default="oof_acc", choices=["equal", "oof_acc"], help="How to aggregate folds into one test pred per base model.")
    p.add_argument("--cwr-ridge-alpha", type=float, default=1.0)
    p.add_argument("--cwr-classwise-alpha", type=float, default=0.2)
    return p.parse_args()


def load_meta_helper() -> object:
    script_path = Path("/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/meta_stack_cv5_attention.py")
    spec = importlib.util.spec_from_file_location("meta_stack_cv5_attention", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load meta_stack_cv5_attention.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_filtered_run_ranking(src_zoo: Path, dst_zoo: Path, include_names: List[str]) -> Path:
    dst_zoo.mkdir(parents=True, exist_ok=True)
    rr = pd.read_csv(src_zoo / "run_ranking.csv")
    rr = rr[(rr["status"] == "ok") & (rr["name"].astype(str).isin(include_names))].copy()
    if rr.empty:
        raise RuntimeError(f"No runs selected from {src_zoo}")
    # Preserve deterministic fold-major order.
    rr["fold_idx"] = rr["fold_idx"].astype(int)
    rr = rr.sort_values(["fold_idx", "name"]).reset_index(drop=True)
    rr.to_csv(dst_zoo / "run_ranking.csv", index=False)
    return dst_zoo


def fit_lr_mse_weights(prob_list: List[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    n_models = len(prob_list)
    p_stack = np.stack(prob_list, axis=-1)  # [N,C,M]
    n_samples, n_classes, _ = p_stack.shape
    y_idx = np.asarray(y_true, dtype=np.int64)
    y_onehot = np.eye(n_classes, dtype=np.float64)[y_idx]
    X = p_stack.reshape(n_samples * n_classes, n_models)
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


def extract_oof_maps(bundles, prefix: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    n_classes = None
    for b in bundles:
        if n_classes is None:
            n_classes = int(b.probs_stack.shape[2])
        for mi, name in enumerate(b.model_names):
            key = f"{prefix}:{name}"
            m = out.setdefault(key, {})
            probs = b.probs_stack[:, mi, :]
            for image_id, p in zip(b.image_ids, probs):
                m[str(image_id)] = np.asarray(p, dtype=np.float32)
    if n_classes is None:
        raise RuntimeError("No bundles")
    return out, n_classes


def fold_metrics_for_bundle(bundles, helper) -> Dict[Tuple[int, int], float]:
    # (fold_idx, model_idx) -> val_acc using final selected checkpoint.
    out: Dict[Tuple[int, int], float] = {}
    for b in bundles:
        for mi, run_dir in enumerate(b.run_dirs):
            summ = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
            acc = float(summ.get("final_metrics", {}).get("val_acc", 0.0))
            out[(int(b.fold_idx), int(mi))] = acc
    return out


def aggregate_test_per_model(
    test_probs_by_fold: Dict[int, np.ndarray],
    bundles,
    acc_map: Dict[Tuple[int, int], float],
    mode: str,
    helper,
) -> Tuple[List[str], List[np.ndarray]]:
    # returns model_names, list of [N_test, C]
    model_names = list(bundles[0].model_names)
    n_models = len(model_names)
    fold_ids = [int(b.fold_idx) for b in bundles]
    per_model: List[np.ndarray] = []
    for mi in range(n_models):
        probs_list = [np.asarray(test_probs_by_fold[fid][:, mi, :], dtype=np.float64) for fid in fold_ids]
        if mode == "equal":
            w = np.ones(len(probs_list), dtype=np.float64) / len(probs_list)
        else:
            raw = np.array([acc_map.get((fid, mi), 0.0) for fid in fold_ids], dtype=np.float64)
            if not np.all(np.isfinite(raw)) or float(raw.sum()) <= 0:
                w = np.ones(len(probs_list), dtype=np.float64) / len(probs_list)
            else:
                w = raw / (raw.sum() + 1e-12)
        blend = np.zeros_like(probs_list[0], dtype=np.float64)
        for wi, pi in zip(w, probs_list):
            blend += float(wi) * pi
        blend = helper.clip_and_norm(blend)
        per_model.append(blend.astype(np.float32))
    return model_names, per_model


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)
    folds_csv = Path(args.folds_csv)
    old_zoo = Path(args.old_zoo_root)
    new_zoo = Path(args.new_zoo_root)

    helper = load_meta_helper()
    device = helper.resolve_device(args.device)
    print("Device:", device, flush=True)

    old_names = [x.strip() for x in args.old_models.split(",") if x.strip()]
    new_names = [x.strip() for x in args.new_models.split(",") if x.strip()]

    with tempfile.TemporaryDirectory(prefix="mix_orch_") as td:
        td_path = Path(td)
        old_tmp = write_filtered_run_ranking(old_zoo, td_path / "old_filtered", old_names)
        new_tmp = write_filtered_run_ranking(new_zoo, td_path / "new_filtered", new_names)
        old_bundles, old_m, old_c = helper.load_cv5_bundles(old_tmp)
        new_bundles, new_m, new_c = helper.load_cv5_bundles(new_tmp)

    if old_c != new_c:
        raise RuntimeError(f"num_classes mismatch old={old_c} new={new_c}")
    n_classes = old_c
    print(f"Loaded old bundles: folds={[b.fold_idx for b in old_bundles]}, models={old_bundles[0].model_names}", flush=True)
    print(f"Loaded new bundles: folds={[b.fold_idx for b in new_bundles]}, models={new_bundles[0].model_names}", flush=True)

    # Build OOF maps and align on cleaned dataset ids + current folds.
    old_maps, _ = extract_oof_maps(old_bundles, "old")
    new_maps, _ = extract_oof_maps(new_bundles, "new")
    all_model_keys = list(old_maps.keys()) + list(new_maps.keys())
    all_maps = {**old_maps, **new_maps}

    train_df = pd.read_csv(base / "train.csv")
    folds_df = pd.read_csv(folds_csv)
    labels_map = dict(zip(train_df["image_id"].astype(str), train_df["label"].astype(int)))
    fold_map = dict(zip(folds_df["image_id"].astype(str), folds_df["fold"].astype(int)))

    ids_intersection = None
    for k in all_model_keys:
        ids_k = set(all_maps[k].keys())
        ids_intersection = ids_k if ids_intersection is None else (ids_intersection & ids_k)
    ids_intersection = (ids_intersection or set()) & set(labels_map.keys()) & set(fold_map.keys())
    if not ids_intersection:
        raise RuntimeError("No common image_ids across selected old/new models and top_new labels/folds")

    # Keep only folds that actually appear in intersection (likely 0/1 due new partial OOF coverage).
    rows = []
    for image_id in ids_intersection:
        rows.append((fold_map[image_id], image_id, labels_map[image_id]))
    rows.sort(key=lambda t: (t[0], t[1]))
    fold_ids = np.array([r[0] for r in rows], dtype=np.int64)
    image_ids = [r[1] for r in rows]
    y = np.array([r[2] for r in rows], dtype=np.int64)
    unique_folds = sorted(np.unique(fold_ids).tolist())
    print(f"Aligned OOF intersection: N={len(image_ids)} folds={unique_folds} models={len(all_model_keys)}", flush=True)

    X_stack = np.zeros((len(image_ids), len(all_model_keys), n_classes), dtype=np.float32)
    for mi, k in enumerate(all_model_keys):
        m = all_maps[k]
        X_stack[:, mi, :] = np.stack([m[iid] for iid in image_ids], axis=0)

    # Pseudo-CV evaluation (note: old models use different base folds, so this is optimistic but useful for ranking).
    ranking_rows = []
    method_artifacts = {}

    # LR(MSE) pseudo-CV
    oof_lr = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold_lr = {}
    for fid in unique_folds:
        tr = fold_ids != fid
        va = fold_ids == fid
        prob_list_tr = [X_stack[tr, i, :] for i in range(X_stack.shape[1])]
        prob_list_va = [X_stack[va, i, :] for i in range(X_stack.shape[1])]
        w = fit_lr_mse_weights(prob_list_tr, y[tr])
        pv = np.zeros_like(prob_list_va[0], dtype=np.float64)
        for wi, pi in zip(w, prob_list_va):
            pv += float(wi) * pi
        pv = helper.clip_and_norm(pv)
        oof_lr[va] = pv
        per_fold_lr[int(fid)] = helper.evaluate_probs(y[va], pv)
    m_lr = helper.evaluate_probs(y, oof_lr)
    ranking_rows.append({"method": "mixed_lr_mse", **m_lr})
    method_artifacts["mixed_lr_mse"] = {"oof_probs": oof_lr, "per_fold": per_fold_lr}
    print(f"[mixed_lr_mse] acc={m_lr['acc']:.5f} f1={m_lr['f1_macro']:.5f}", flush=True)

    # Classwise ridge pseudo-CV
    oof_cwr = np.zeros((len(y), n_classes), dtype=np.float32)
    per_fold_cwr = {}
    for fid in unique_folds:
        tr = fold_ids != fid
        va = fold_ids == fid
        W, cf = helper.fit_classwise_ridge_from_stack(
            X_stack[tr],
            y[tr],
            classwise_alpha=args.cwr_classwise_alpha,
            ridge_alpha=args.cwr_ridge_alpha,
        )
        pv = helper.predict_classwise_ridge_from_stack(X_stack[va], W=W, class_freq=cf, classwise_alpha=args.cwr_classwise_alpha)
        oof_cwr[va] = pv
        per_fold_cwr[int(fid)] = helper.evaluate_probs(y[va], pv)
    m_cwr = helper.evaluate_probs(y, oof_cwr)
    ranking_rows.append({"method": "mixed_cwr", **m_cwr})
    method_artifacts["mixed_cwr"] = {"oof_probs": oof_cwr, "per_fold": per_fold_cwr}
    print(f"[mixed_cwr] acc={m_cwr['acc']:.5f} f1={m_cwr['f1_macro']:.5f}", flush=True)

    # Build test model matrices by running inference in each zoo and aggregating folds within each model.
    print("\n=== TEST INFERENCE OLD ZOO ===", flush=True)
    old_test_by_fold, test_ids_old = helper.build_test_probs_per_fold(
        bundles=old_bundles,
        base=base,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tta_mode=args.tta_mode,
        tta_views=args.tta_views,
        reuse_cache=args.reuse_test_cache,
        save_cache=args.save_test_cache,
    )
    print("\n=== TEST INFERENCE NEW ZOO ===", flush=True)
    new_test_by_fold, test_ids_new = helper.build_test_probs_per_fold(
        bundles=new_bundles,
        base=base,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tta_mode=args.tta_mode,
        tta_views=args.tta_views,
        reuse_cache=args.reuse_test_cache,
        save_cache=args.save_test_cache,
    )
    if list(test_ids_old) != list(test_ids_new):
        raise RuntimeError("test image_id order mismatch between old and new zoo inference")
    test_ids = list(test_ids_old)

    old_acc_map = fold_metrics_for_bundle(old_bundles, helper)
    new_acc_map = fold_metrics_for_bundle(new_bundles, helper)
    old_model_names, old_test_per_model = aggregate_test_per_model(
        old_test_by_fold, old_bundles, old_acc_map, mode=args.fold_agg, helper=helper
    )
    new_model_names, new_test_per_model = aggregate_test_per_model(
        new_test_by_fold, new_bundles, new_acc_map, mode=args.fold_agg, helper=helper
    )

    # Ensure ordering matches OOF feature ordering.
    expected_keys = [f"old:{n}" for n in old_model_names] + [f"new:{n}" for n in new_model_names]
    if expected_keys != all_model_keys:
        raise RuntimeError("Model ordering mismatch between OOF and test aggregation")

    X_test_stack = np.stack(old_test_per_model + new_test_per_model, axis=1).astype(np.float32)  # [N_test, M, C]

    # Final fits and submissions.
    sample_sub = pd.read_csv(base / "sample_submission.csv")
    if sample_sub["image_id"].astype(str).tolist() != test_ids:
        sub_template = pd.DataFrame({"image_id": test_ids})
    else:
        sub_template = sample_sub.copy()

    # Final LR(MSE)
    w_lr_final = fit_lr_mse_weights([X_stack[:, i, :] for i in range(X_stack.shape[1])], y)
    test_lr = np.zeros((X_test_stack.shape[0], n_classes), dtype=np.float64)
    for wi, pi in zip(w_lr_final, [X_test_stack[:, i, :] for i in range(X_test_stack.shape[1])]):
        test_lr += float(wi) * pi
    test_lr = helper.clip_and_norm(test_lr)
    pred_lr = test_lr.argmax(1).astype(int)
    sub_lr = sub_template.copy()
    sub_lr["label"] = pred_lr
    sub_lr_path = out_dir / f"submission_mixed_lr_mse_{args.tta_mode}_{args.fold_agg}.csv"
    sub_lr.to_csv(sub_lr_path, index=False)
    np.save(out_dir / f"submission_mixed_lr_mse_{args.tta_mode}_{args.fold_agg}_probs.npy", test_lr.astype(np.float32))

    # Final CWR
    W_cwr_final, cf_cwr_final = helper.fit_classwise_ridge_from_stack(
        X_stack, y, classwise_alpha=args.cwr_classwise_alpha, ridge_alpha=args.cwr_ridge_alpha
    )
    test_cwr = helper.predict_classwise_ridge_from_stack(
        X_test_stack, W=W_cwr_final, class_freq=cf_cwr_final, classwise_alpha=args.cwr_classwise_alpha
    )
    pred_cwr = test_cwr.argmax(1).astype(int)
    sub_cwr = sub_template.copy()
    sub_cwr["label"] = pred_cwr
    sub_cwr_path = out_dir / f"submission_mixed_cwr_{args.tta_mode}_{args.fold_agg}.csv"
    sub_cwr.to_csv(sub_cwr_path, index=False)
    np.save(out_dir / f"submission_mixed_cwr_{args.tta_mode}_{args.fold_agg}_probs.npy", test_cwr.astype(np.float32))

    rank_df = pd.DataFrame(ranking_rows).sort_values(["acc", "f1_macro"], ascending=False)
    rank_df.to_csv(out_dir / "mixed_oof_ranking.csv", index=False)

    summary = {
        "status": "ok",
        "note": "Pseudo-CV on top_new folds for aligned intersection; optimistic for old-zoo models due different original fold partitions.",
        "n_models_total": int(len(all_model_keys)),
        "model_keys": all_model_keys,
        "aligned_samples": int(len(image_ids)),
        "aligned_folds": [int(x) for x in unique_folds],
        "results": rank_df.to_dict(orient="records"),
        "submissions": {
            "mixed_lr_mse": str(sub_lr_path),
            "mixed_cwr": str(sub_cwr_path),
        },
        "tta_mode": args.tta_mode,
        "fold_agg": args.fold_agg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== MIXED ORCHESTRATOR DONE ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
