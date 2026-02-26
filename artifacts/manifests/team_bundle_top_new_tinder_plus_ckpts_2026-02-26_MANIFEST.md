# Team Bundle: `top_new_dataset` + task-specific checkpoints

Дата сборки: `2026-02-26`

## Что внутри

### 1) Обновлённый датасет после Tinder-cleaning
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset`

Это рабочий датасет после ручной чистки/релэйбла в Tinder UI:
- `train.csv` обновлён
- структура `train/train/*` синхронизирована с `train.csv`
- есть служебные артефакты clean/review (backup CSV, queue manifest и т.п.)

### 2) Фиксированный folds CSV для честного сравнения
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv`

Используйте этот файл для одинаковых fold-сплитов между машинами.

### 3) Сильные task-specific CV5 checkpoints (старый zoo, 3 семейства x 5 folds)
Из `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/runs/*` включены только run-директории:

- `cnn_convnext_small_sam_swa` (folds `0..4`)
- `cnn_effnetv2_s_sam_swa` (folds `0..4`)
- `vit_deit3_small_color_safe` (folds `0..4`)

В каждом run-директории есть:
- `config.json`
- `summary.json`
- `best_by_val_loss.pt`
- `swa_model.pt` (если есть)
- OOF артефакты (`val_probs.npy`, `val_labels.npy`, `val_predictions.csv`) и логи

### 4) Актуальные one-fold checkpoints на новом датасете (feature probes)
Из `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_adaptive_night_run1/phase1_feature_probes/runs/*` включены:

- `000_baseline`
- `003_swa_earlier`
- `004_label_smoothing_low`

Это полезно как warm-start / reference для новых прогонов на `top_new_dataset`.

### 5) Скрипты (чтобы команда запускала тот же пайплайн)
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/adaptive_top_new_night_pipeline.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_top_new_dataset_adaptive_night.sh`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/friend_phase1_tail3_probes.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/train_onefold_no_color_innov_mps.py`

## Важно

- Базовые ImageNet/IN22K веса `timm` (то, что скачивается через `pretrained=True`) **не включены** в этот bundle.
- Они подтягиваются автоматически при первом запуске на машине команды (если есть интернет).
- В bundle включены именно **task-specific дообученные checkpoints**, которые можно использовать для анализа / warm-start / сравнения.

## Рекомендуемое использование

1. Распаковать bundle в корень проекта (чтобы пути `dl_lab1/...` совпали).
2. Для честных сравнений использовать `folds_used_top_new_dataset_aligned_hybrid.csv`.
3. Для быстрых проверок можно стартовать от included one-fold probe checkpoints (`003`, `004`).
