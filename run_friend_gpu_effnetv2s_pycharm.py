#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """
    PyCharm (Windows) one-click launcher for the GPU friend.
    Open this file in PyCharm and press Run (green arrow).
    """
    bundle_root = Path(__file__).resolve().parent
    orchestrator = bundle_root / "dl_lab1" / "scripts" / "top_new_final3_cv5_train_orchestrator.py"
    base = bundle_root / "dl_lab1" / "top_new_dataset"
    folds_csv = (
        bundle_root
        / "dl_lab1"
        / "outputs_post_tinder_convnext_cv2_compare"
        / "folds_used_top_new_dataset_aligned_hybrid.csv"
    )
    out_root = bundle_root / "dl_lab1" / "outputs_final3_split_effnetv2_s"

    missing = [p for p in [orchestrator, base, folds_csv] if not p.exists()]
    if missing:
        print("Missing required files/folders. Open the bundle root in PyCharm.", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 2

    py = sys.executable or "python"
    cmd = [
        py,
        str(orchestrator),
        "--base",
        str(base),
        "--clean-variant",
        "raw",
        "--folds-csv",
        str(folds_csv),
        "--python-bin",
        py,
        "--device",
        "cuda",
        "--num-workers",
        "0",
        "--models",
        "effnetv2_s",
        "--folds",
        "0,1,2,3",
        "--epochs-override",
        "14",
        "--stage1-epochs-override",
        "9",
        "--out-root",
        str(out_root),
        "--resume",
        "--continue-on-error",
    ]

    print("=== GPU Friend Launcher (PyCharm / Windows) ===", flush=True)
    print(f"Python: {py}", flush=True)
    print("Command:", " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    print(flush=True)

    try:
        import torch  # type: ignore

        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            try:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
            except Exception:
                pass
        else:
            print(
                "WARNING: CUDA is not available in the selected PyCharm interpreter. "
                "Install CUDA-enabled PyTorch in this interpreter first.",
                flush=True,
            )
    except Exception:
        print("Note: torch import check skipped (torch is not installed yet).", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.call(cmd, cwd=str(bundle_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
