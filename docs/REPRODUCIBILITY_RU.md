# Воспроизведение решения (RU)

Этот документ описывает практические сценарии воспроизведения:
- базового full-CV5 ансамбля (`0.97154 public`);
- дедлайн split-training (командный режим);
- late-stage `mixed old+new` orchestrator (`0.97338 public`);
- Tinder-cleaning pipeline (данные).

## 1. Что нужно заранее

### 1.1. Окружение

- Python `3.10` или `3.11`
- PyTorch + torchvision:
  - Apple Silicon: сборка с `MPS`
  - CUDA GPU: сборка с `CUDA`

Установка общих пакетов:

```bash
pip install -r requirements_common_team_train.txt
```

Обязательные пакеты из файла:
- `numpy`
- `pandas`
- `scikit-learn`
- `pillow`
- `tqdm`
- `timm`
- `albumentations`
- `opencv-python-headless`

Опционально (для части meta-экспериментов):
- `catboost`
- `matplotlib`
- `kaggle`

### 1.2. Данные (не хранятся в git)

См. `data/README.md`.

Ключевой принцип:
- **тест не меняется вообще**
- любые чистки/релэйблы применяются только к `train`

### 1.3. Веса и большие артефакты (не хранятся в git)

См. `weights/README.md`.

Минимально для воспроизведения финальных результатов нужны:
- task-specific checkpoints (old zoo + new zoo)
- финальные CV5 run-артефакты
- при необходимости bundle с `top_new_dataset` и fixed folds

### 1.4. Ускоренный маршрут

Для быстрого старта см. `repro/README.md`:
- `repro/scripts/01_check_layout.sh`
- `repro/scripts/02_reproduce_full_cv5_submission.sh`
- `repro/scripts/03_reproduce_best_mixed_submission.sh`

## 2. Ожидаемая структура путей

Launcher-скрипты рассчитаны на структуру:

```text
repo_root/
├── dl_lab1/
│   ├── scripts/
│   ├── top_new_dataset/                       # положить сюда подготовленный датасет
│   ├── outputs_post_tinder_convnext_cv2_compare/
│   │   └── folds_used_top_new_dataset_aligned_hybrid.csv
│   └── outputs_*                              # будут создаваться при запусках
├── run_me_m2_convnext_small.sh
├── run_friend_gpu_effnetv2s.sh
├── run_friend_m1_deit3_small.sh
└── requirements_common_team_train.txt
```

В этом репозитории `folds_used_top_new_dataset_aligned_hybrid.csv` уже лежит в двух местах:
- `artifacts/folds/...`
- `dl_lab1/outputs_post_tinder_convnext_cv2_compare/...` (для совместимости с launcher'ами)

## 3. Сценарий A: воспроизвести full-CV5 baseline/hеdge (`0.97154`)

Цель:
- получить/проверить сабмит `submission_cv5_all20_lr_geo8_equal.csv`
- это зрелый full-CV5 ансамбль из `20` моделей (`4 модели x 5 фолдов`)

### 3.1. Что нужно

- исходный competition dataset
- подготовленные run-артефакты `outputs_night_model_zoo_cv5/runs/*`
  - с `val_probs.npy`, `val_labels.npy`, конфигами, checkpoint'ами
- test inference weights / checkpoints для всех run'ов

### 3.2. Команда сборки сабмита

Основной скрипт:
- `dl_lab1/scripts/make_submission_from_cv5_all20_lr_tta.py`

Результат-референс в репозитории:
- `artifacts/submissions/cv5_all20/submission_cv5_all20_lr_geo8_equal.csv`
- `artifacts/submissions/cv5_all20/submission_cv5_all20_lr_geo8_equal_meta.json`

### 3.3. Ожидаемая логика

- внутри каждого фолда веса ищутся через `LinearRegression(fit_intercept=False, positive=True)` по `val_probs -> one_hot(y)`
- отрицательные веса обрезаются, затем нормализуются
- TTA на тесте: `geo8`
- фолд-агрегация: `equal`

## 4. Сценарий B: дедлайн split-training (командное распараллеливание)

Цель:
- воспроизвести конфиг `3 модели x 4 фолда x 14 эпох`
- распределить обучение между несколькими машинами (MPS / CUDA / Colab)

### 4.1. Launcher-скрипты

- M2 / ConvNeXt-S: `run_me_m2_convnext_small.sh`
- CUDA / EffNetV2-S: `run_friend_gpu_effnetv2s.sh`
- M1 / DeiT3-S: `run_friend_m1_deit3_small.sh`
- Colab experimental hardening: `run_friend_colab_effnetv2s_classaware_harden.sh`
- Windows/PyCharm helper: `run_friend_gpu_effnetv2s_pycharm.py`

### 4.2. Что делают launcher'ы

Все launcher'ы вызывают:
- `dl_lab1/scripts/top_new_final3_cv5_train_orchestrator.py`

С типовым конфигом:
- `--clean-variant raw`
- `--folds 0,1,2,3`
- `--epochs-override 14`
- `--stage1-epochs-override 9`
- `--resume`

### 4.3. Встроенные решения из adaptive phase1

В дедлайн-CNN конфиге закреплены:
- `no_color`
- `SAM`
- `SWA (start=5)`
- `label_smoothing=0.03`
- `weighted_sampler`
- `mixup + cutmix`

## 5. Сценарий C: late-stage mixed old+new orchestrator (`0.97338`)

Цель:
- воспроизвести лучший public score на текущем цикле (`2026-02-26`)

### 5.1. Что нужно

1. `old` zoo (pre-clean, full CV5):
- `ConvNeXt-S`
- `EffNetV2-S`
- `DeiT3-S`

2. `new` zoo (post-clean, минимум common `fold0,1`; лучше больше):
- `ConvNeXt-S`
- `EffNetV2-S`
- `DeiT3-S`

3. aligned folds CSV:
- `dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv`

4. `top_new_dataset` (после Tinder-cleaning)

### 5.2. Основной скрипт

- `dl_lab1/scripts/mixed_old_new_orchestrator_submit.py`

Он:
- выравнивает OOF по `image_id`
- строит stacked features из `old + new`
- генерирует сабмиты через:
  - `mixed_lr_mse`
  - `mixed_cwr`
- использует `geo8` TTA

### 5.3. Референс-артефакты в репозитории

Папка:
- `artifacts/submissions/mixed_old_new/`

Ключевые файлы:
- `summary.json`
- `submission_mixed_lr_mse_geo8_oof_acc.csv` (best public `0.97338`)
- `submission_mixed_cwr_geo8_oof_acc.csv`
- `submission_mixed_lr_mse_geo8_equal.csv`
- `submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr1.csv`
- `submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr2.csv`

Важно:
- pseudo-CV в `summary.json` отмечен как **оптимистичный** для `old-zoo` (из-за разных исторических fold-разметок);
- финальное решение выбиралось по совокупности `OOF + public + инженерная надёжность`.

## 6. Сценарий D: воспроизведение Tinder-cleaning pipeline

Цель:
- показать воспроизводимый процесс ручной чистки train-данных

### 6.1. Скрипты

- UI: `dl_lab1/scripts/dataset_tinder_review_app.py`
- build confidence cache: `dl_lab1/scripts/build_tinder_confidence_cache.py`
- export actions (с replay `undo`): `dl_lab1/scripts/export_tinder_session_actions.py`
- apply actions to dataset: `dl_lab1/scripts/apply_tinder_actions_to_top_new_dataset.py`

### 6.2. Референс-артефакты в репозитории

Папка:
- `artifacts/manual_review/`

Файлы:
- `tinder_session_export_summary.json`
- `actions_for_apply_manual_actions.csv`
- `per_class_stats.csv`
- `tinder_session_meta.json`

Эти файлы удобно показывать на защите как доказательство воспроизводимости ручной чистки.

## 7. Проверка воспроизводимости и sanity-checks

Перед запуском ночных/долгих прогонов рекомендуется:

1. Проверить, что `dl_lab1/top_new_dataset` существует и содержит `train.csv`, `train/train`, `test.csv`, `sample_submission.csv`.
2. Проверить, что `folds_used_top_new_dataset_aligned_hybrid.csv` доступен по пути из launcher'ов.
3. Сделать smoke-запуск на одном фолде и одной модели.
4. Проверить, что `timm` скачал базовые pretrained weights (если интернет доступен).
5. Зафиксировать версию PyTorch и устройство (`mps` / `cuda`) в логах запуска.

## 8. Финальные шаги перед публикацией

См. `docs/RELEASE_CHECKLIST_RU.md`, но минимум:
- загрузить большие веса/бандлы на Google Drive;
- вписать ссылки в `weights/README.md`;
- создать GitHub Release (не обязательно, но удобно);
- при необходимости расширить post-competition анализ в `docs/EXPERIMENTS_RU.md`.
