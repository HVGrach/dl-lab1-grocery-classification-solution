#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_onefold_no_color_innov_mps.py"

DEFAULT_BASE = "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset"
DEFAULT_FOLDS_CSV = (
    "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/"
    "outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv"
)
DEFAULT_OUT_ROOT = "/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_final3_cv5_train"
DEFAULT_PYTHON_BIN = sys.executable or "python"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    run_name: str
    model_name: str
    img_size: int
    batch_size: int
    epochs: int
    stage1_epochs: int
    warmup_epochs: int
    lr: float
    lr_drop_factor: float
    weight_decay: float
    grad_clip_norm: float
    label_smoothing: float
    use_channels_last: bool
    use_weighted_sampler: bool
    use_mixup: bool
    mixup_alpha: float
    mixup_prob: float
    use_cutmix: bool
    cutmix_alpha: float
    cutmix_prob: float
    use_sam: bool
    sam_rho: float
    sam_adaptive: bool
    use_swa: bool
    swa_start_epoch: int


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def parse_str_csv(raw: str) -> List[str]:
    out: List[str] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(chunk)
    return out


def build_default_specs() -> Dict[str, ModelSpec]:
    # Base settings come from the known strong 3-model CV stack.
    # CNN tweaks apply the new phase1 findings (label_smoothing=0.03, earlier SWA).
    specs = [
        ModelSpec(
            key="convnext_small",
            run_name="cnn_convnext_small_sam_swa_ls03_swa5",
            model_name="convnext_small.fb_in22k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=16,
            stage1_epochs=10,
            warmup_epochs=2,
            lr=3e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
            label_smoothing=0.03,
            use_channels_last=True,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=5,
        ),
        ModelSpec(
            key="effnetv2_s",
            run_name="cnn_effnetv2_s_sam_swa_ls03_swa5",
            model_name="tf_efficientnetv2_s.in21k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=16,
            stage1_epochs=10,
            warmup_epochs=2,
            lr=2.5e-4,
            lr_drop_factor=4.0,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
            label_smoothing=0.03,
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.20,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=True,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=5,
        ),
        ModelSpec(
            key="deit3_small",
            run_name="vit_deit3_small_safe",
            model_name="deit3_small_patch16_224.fb_in22k_ft_in1k",
            img_size=224,
            batch_size=16,
            epochs=14,
            stage1_epochs=9,
            warmup_epochs=2,
            lr=2e-4,
            lr_drop_factor=3.0,
            weight_decay=8e-5,
            grad_clip_norm=1.0,
            label_smoothing=0.08,  # kept conservative until separately re-probed on ViT
            use_channels_last=False,
            use_weighted_sampler=True,
            use_mixup=True,
            mixup_alpha=0.2,
            mixup_prob=0.25,
            use_cutmix=True,
            cutmix_alpha=1.0,
            cutmix_prob=0.20,
            use_sam=False,
            sam_rho=0.05,
            sam_adaptive=False,
            use_swa=True,
            swa_start_epoch=11,
        ),
    ]
    return {s.key: s for s in specs}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the final 3-model CV5 stack on top_new_dataset using deterministic run folders. "
            "Designed for later splitting across multiple machines via --models/--folds."
        )
    )
    p.add_argument("--base", type=str, default=DEFAULT_BASE)
    p.add_argument("--clean-variant", type=str, default="raw", choices=["strict", "aggressive", "raw"])
    p.add_argument("--folds-csv", type=str, default=DEFAULT_FOLDS_CSV)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated fold indices.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--models",
        type=str,
        default="convnext_small,effnetv2_s,deit3_small",
        help="Comma-separated model keys. Available: convnext_small,effnetv2_s,deit3_small",
    )
    p.add_argument(
        "--epochs-override",
        type=int,
        default=0,
        help="If >0, override epochs for all selected models (deadline mode).",
    )
    p.add_argument(
        "--stage1-epochs-override",
        type=int,
        default=0,
        help="If >0, override stage1_epochs for all selected models.",
    )
    p.add_argument("--python-bin", type=str, default=DEFAULT_PYTHON_BIN)
    p.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--resume", action="store_true", help="Skip jobs with existing summary.json")
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--print-only", action="store_true", help="Alias of --dry-run")
    p.add_argument(
        "--order",
        type=str,
        default="fold-major",
        choices=["fold-major", "model-major"],
        help="Job ordering only; useful for cache behavior and manual interruption.",
    )
    p.set_defaults(continue_on_error=True)
    return p.parse_args()


def resolve_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    catalog = build_default_specs()
    requested = parse_str_csv(args.models)
    if not requested:
        raise ValueError("--models resolved to empty list")
    unknown = [m for m in requested if m not in catalog]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {sorted(catalog)}")
    return [catalog[m] for m in requested]


def resolve_folds(args: argparse.Namespace) -> List[int]:
    folds = sorted(set(parse_int_csv(args.folds)))
    if not folds:
        raise ValueError("--folds resolved to empty list")
    return folds


def build_cmd(
    *,
    python_bin: str,
    base: str,
    clean_variant: str,
    folds_csv: str,
    n_splits: int,
    fold_seed: int,
    fold_idx: int,
    seed: int,
    device: str,
    num_workers: int,
    epochs_override: int,
    stage1_epochs_override: int,
    spec: ModelSpec,
    out_dir: Path,
) -> List[str]:
    epochs = int(epochs_override) if int(epochs_override) > 0 else int(spec.epochs)
    stage1_epochs = int(stage1_epochs_override) if int(stage1_epochs_override) > 0 else int(spec.stage1_epochs)
    cmd = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--base",
        base,
        "--clean-variant",
        clean_variant,
        "--fold-idx",
        str(fold_idx),
        "--seed",
        str(seed),
        "--device",
        device,
        "--model-name",
        spec.model_name,
        "--img-size",
        str(spec.img_size),
        "--batch-size",
        str(spec.batch_size),
        "--num-workers",
        str(num_workers),
        "--epochs",
        str(epochs),
        "--stage1-epochs",
        str(stage1_epochs),
        "--warmup-epochs",
        str(spec.warmup_epochs),
        "--lr",
        str(spec.lr),
        "--lr-drop-factor",
        str(spec.lr_drop_factor),
        "--weight-decay",
        str(spec.weight_decay),
        "--grad-clip-norm",
        str(spec.grad_clip_norm),
        "--label-smoothing",
        str(spec.label_smoothing),
        "--mixup-alpha",
        str(spec.mixup_alpha),
        "--mixup-prob",
        str(spec.mixup_prob),
        "--cutmix-alpha",
        str(spec.cutmix_alpha),
        "--cutmix-prob",
        str(spec.cutmix_prob),
        "--sam-rho",
        str(spec.sam_rho),
        "--swa-start-epoch",
        str(spec.swa_start_epoch),
        "--out-dir",
        str(out_dir),
    ]

    if folds_csv and Path(folds_csv).exists():
        cmd.extend(["--folds-csv", folds_csv])
    else:
        cmd.extend(["--n-splits", str(n_splits), "--fold-seed", str(fold_seed)])

    if spec.use_channels_last:
        cmd.append("--use-channels-last")
    if not spec.use_weighted_sampler:
        cmd.append("--no-weighted-sampler")
    if not spec.use_mixup:
        cmd.append("--no-mixup")
    if not spec.use_cutmix:
        cmd.append("--no-cutmix")
    if not spec.use_sam:
        cmd.append("--no-sam")
    elif spec.sam_adaptive:
        cmd.append("--sam-adaptive")
    if not spec.use_swa:
        cmd.append("--no-swa")
    return cmd


def make_jobs(specs: Sequence[ModelSpec], folds: Sequence[int], order: str) -> List[tuple[ModelSpec, int]]:
    jobs: List[tuple[ModelSpec, int]] = []
    if order == "fold-major":
        for f in folds:
            for s in specs:
                jobs.append((s, int(f)))
    else:
        for s in specs:
            for f in folds:
                jobs.append((s, int(f)))
    return jobs


def read_summary_metrics(summary_path: Path) -> Dict[str, object]:
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    fm = data.get("final_metrics") or {}
    return {
        "final_model_selected": str(data.get("final_model_selected", "")),
        "val_loss": fm.get("val_loss"),
        "val_acc": fm.get("val_acc"),
        "val_f1_macro": fm.get("val_f1_macro"),
        "val_errors": fm.get("val_errors"),
        "val_size": fm.get("val_size"),
    }


def write_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    if args.print_only:
        args.dry_run = True

    specs = resolve_model_specs(args)
    folds = resolve_folds(args)
    jobs = make_jobs(specs, folds, args.order)

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "timestamp": int(time.time()),
        "trainer_script": str(TRAIN_SCRIPT),
        "base": args.base,
        "clean_variant": args.clean_variant,
        "folds_csv": args.folds_csv if args.folds_csv and Path(args.folds_csv).exists() else "",
        "n_splits": int(args.n_splits),
        "fold_seed": int(args.fold_seed),
        "folds": folds,
        "seed": int(args.seed),
        "device": args.device,
        "num_workers": int(args.num_workers),
        "python_bin": args.python_bin,
        "order": args.order,
        "resume": bool(args.resume),
        "continue_on_error": bool(args.continue_on_error),
        "epochs_override": int(args.epochs_override),
        "stage1_epochs_override": int(args.stage1_epochs_override),
        "models": [asdict(s) for s in specs],
        "num_jobs": len(jobs),
    }
    (out_root / "train_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== FINAL3 CV5 TRAIN ORCHESTRATOR ===", flush=True)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)

    records: List[Dict[str, object]] = []
    for idx, (spec, fold_idx) in enumerate(jobs, start=1):
        run_dir = runs_root / f"{spec.run_name}_f{fold_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"

        cmd = build_cmd(
            python_bin=args.python_bin,
            base=args.base,
            clean_variant=args.clean_variant,
            folds_csv=args.folds_csv,
            n_splits=args.n_splits,
            fold_seed=args.fold_seed,
            fold_idx=fold_idx,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            epochs_override=args.epochs_override,
            stage1_epochs_override=args.stage1_epochs_override,
            spec=spec,
            out_dir=run_dir,
        )
        cmd_txt = " ".join(shlex.quote(x) for x in cmd)
        (run_dir / "cmd.txt").write_text(cmd_txt + "\n", encoding="utf-8")

        row: Dict[str, object] = {
            "idx": idx,
            "job_name": spec.run_name,
            "model_key": spec.key,
            "model_name": spec.model_name,
            "fold_idx": int(fold_idx),
            "run_dir": str(run_dir),
            "status": "pending",
            "seconds": 0.0,
        }

        if args.resume and summary_path.exists():
            row["status"] = "skipped_existing"
            row.update(read_summary_metrics(summary_path))
            records.append(row)
            print(f"[{idx}/{len(jobs)}] skip existing: {spec.run_name} fold={fold_idx}", flush=True)
            continue

        print(f"\n[{idx}/{len(jobs)}] {spec.run_name} fold={fold_idx}", flush=True)
        print(cmd_txt, flush=True)

        t0 = time.time()
        if args.dry_run:
            row["status"] = "dry_run"
            row["seconds"] = 0.0
            records.append(row)
            continue

        try:
            subprocess.run(cmd, check=True)
            row["status"] = "ok"
        except subprocess.CalledProcessError as e:
            row["status"] = "failed"
            row["error"] = f"returncode={e.returncode}"
            if not args.continue_on_error:
                row["seconds"] = float(time.time() - t0)
                row.update(read_summary_metrics(summary_path))
                records.append(row)
                write_csv(records, out_root / "run_status.csv")
                (out_root / "run_status.json").write_text(
                    json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print("\nStopped on first error (--stop-on-error).", flush=True)
                sys.exit(e.returncode)
        finally:
            row["seconds"] = float(time.time() - t0)

        row.update(read_summary_metrics(summary_path))
        records.append(row)

        write_csv(records, out_root / "run_status.csv")
        (out_root / "run_status.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_jobs": len(jobs),
        "num_ok": sum(1 for r in records if r.get("status") == "ok"),
        "num_failed": sum(1 for r in records if r.get("status") == "failed"),
        "num_skipped_existing": sum(1 for r in records if r.get("status") == "skipped_existing"),
        "num_dry_run": sum(1 for r in records if r.get("status") == "dry_run"),
        "run_status_csv": str(out_root / "run_status.csv"),
        "runs_root": str(runs_root),
    }
    (out_root / "orchestrator_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ORCHESTRATOR DONE ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
