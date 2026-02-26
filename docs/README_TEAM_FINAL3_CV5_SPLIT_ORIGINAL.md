# Team Final3 CV5 Split Bundle (top_new_dataset)

This bundle is for parallel training of the final 3-model CV5 stack on the cleaned `top_new_dataset`.

Current bundle version is configured for **deadline mode**:
- `4 folds` (`0,1,2,3`)
- `14 epochs` (with `stage1=9` override)

## What's included
- `dl_lab1/top_new_dataset/` (dataset)
- `dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv` (fixed folds)
- `dl_lab1/scripts/` (project scripts; includes trainer + orchestrator)
- role-specific launch scripts (`run_*.sh`)

## Python environment
Use Python 3.10 or 3.11.

Install hardware-specific PyTorch first:
- Apple Silicon (M1/M2): install PyTorch + torchvision from the official macOS instructions.
- NVIDIA CUDA PC: install PyTorch + torchvision with CUDA support from the official Linux/Windows instructions.

Then install common packages:
```bash
pip install -r requirements_common_team_train.txt
```

## Role commands (from bundle root)
- M2 (ConvNeXt-S):
```bash
bash run_me_m2_convnext_small.sh
```
- GPU friend (EfficientNetV2-S on CUDA):
```bash
bash run_friend_gpu_effnetv2s.sh
```
- M1 friend (DeiT3-S on MPS):
```bash
bash run_friend_m1_deit3_small.sh
```

## Notes
- Scripts use `--resume`, so rerun is safe.
- Outputs are written into separate folders under `dl_lab1/outputs_final3_split_*`.
- The trainer now supports `--device cuda` as well as `mps`.


## Colab Experimental (optional 4th friend)
- EffNetV2-S with class-aware final hardening (last 4 epochs):
```bash
bash run_friend_colab_effnetv2s_classaware_harden.sh
```
- Defaults to `DEVICE=cuda` and `BATCH_SIZE=12` (override via env vars if needed).
- This is an **extra experimental model family** for ensemble diversity, not a replacement for the main 3-model split.
