# DL Lab 1 Grocery Classification Solution (Release)

Оформленный репозиторий для воспроизведения решения по соревнованию `dl-lab-1-image-classification` (15 классов фруктов/овощей) и подготовки защиты/презентации.

Статус этого релиза:
- зафиксирован по состоянию на `2026-02-26` (дедлайн-спринт);
- лучший public score на текущем цикле: **`0.97338`** (`mixed old+new + LR(MSE) + geo8`);
- официальный private leaderboard score команды `батчсайз не влез`: **`0.95200`** (`5/17`, данные Kaggle от `2026-02-27`).

## Что здесь есть

Репозиторий собран как `lean release`:
- исходники пайплайнов (`dl_lab1/scripts/`, тренеры, оркестраторы);
- launcher-скрипты для распределённого обучения (MPS/CUDA/Colab/PyCharm);
- артефакты для защиты (финальные сабмиты, summary, fixed folds, Tinder-cleaning actions);
- документация для воспроизведения и презентации;
- реестр внешних весов/бандлов (Google Drive).

Репозиторий **не содержит**:
- исходный датасет соревнования;
- большие checkpoints/веса (`.pt/.pth`);
- большие zip-бандлы (хранятся на внешнем диске).

## Ключевые результаты

Финальный late-stage победитель на public (по состоянию на `2026-02-26`):

- `mixed old+new` orchestrator
- blending: `LR(MSE)` (positive convex weights)
- TTA: `geo8` (только геометрия)
- public score: **`0.97338`**

Файл сабмита в репозитории:
- `artifacts/submissions/mixed_old_new/submission_mixed_lr_mse_geo8_oof_acc.csv`

Стабильный full-CV5 хедж:
- `artifacts/submissions/cv5_all20/submission_cv5_all20_lr_geo8_equal.csv`
- public score: `0.97154`

Официальный private leaderboard (`2026-02-27`):
- team: `батчсайз не влез`
- private score: **`0.95200`**
- rank: **`5/17`**

Пост-соревновательный срез по сабмитам:
- лучший private score среди отправленных сабмитов: `0.95515` (`attention_ensemble_submission.csv`, public `0.96798`);
- late-stage ветка `mixed LR 0.97338` показала private `0.94675`, что подтверждает переоптимизацию под public leaderboard.

Подробные таблицы: `docs/EXPERIMENTS_RU.md` и `artifacts/submissions/public_scores_registry.csv`.

## Навигация по ключевым материалам

- эксперименты и результаты: `docs/EXPERIMENTS_RU.md`
- воспроизводимость: `docs/REPRODUCIBILITY_RU.md`
- краткий маршрут воспроизведения: `repro/README.md`
- структура презентации: `docs/PRESENTATION_OUTLINE_RU.md`
- checklist публикации: `docs/RELEASE_CHECKLIST_RU.md`

## Команда и зоны ответственности

Проект велся параллельно по нескольким направлениям: данные, обучение, ансамблирование, проверка гипотез, оформление результатов и воспроизводимости.

- Фёдор Грачёв: основной экспериментальный и интеграционный контур проекта (ключевые тренировочные циклы, оркестраторы и сборка финального решения), а также подготовка воспроизводимого репозитория, презентационных материалов и значимой части Kaggle-проверок/сабмитов.
- Ярослав Кулизнев: активная работа с гипотезами и альтернативными идеями для улучшения пайплайна, обсуждение экспериментальных направлений и проверка части предложений.
- Константин Родионов: параллельные практические задачи по проекту, включая работу с данными и участие в запусках/сопровождении обучения.

## Визуальные материалы для презентации

Color ablation (ключевое открытие: `no_color` лучше):

![Color Ablation](assets/chart_color_ablation.png)

Эволюция ансамбля:

![Ensemble Progress](assets/chart_ensemble_progress.png)

Риск-профиль (public/private shift risk):

![Risk Profile](assets/chart_risk_profile.png)

Ручная чистка (Tinder-style UI):

![Cleaning Preview](assets/cleaning_preview.jpg)

## Структура репозитория

```text
.
├── README.md
├── requirements_common_team_train.txt
├── run_me_m2_convnext_small.sh
├── run_friend_gpu_effnetv2s.sh
├── run_friend_m1_deit3_small.sh
├── run_friend_colab_effnetv2s_classaware_harden.sh
├── run_friend_gpu_effnetv2s_pycharm.py
├── dl_lab1/
│   ├── scripts/                       # оркестраторы, обучение, постпроцессинг, Tinder UI
│   ├── train_top1_mps.py              # исторический базовый trainer
│   ├── train_pair_experts_mps.py      # pair-experts trainer
│   ├── TOP1_pipeline_clean_cv_ensemble.ipynb
│   ├── top_new_dataset/README.md      # точка размещения рабочего датасета
│   ├── outputs_post_tinder_convnext_cv2_compare/
│   │   └── folds_used_top_new_dataset_aligned_hybrid.csv
│   └── outputs_night_model_zoo_cv5/   # небольшие примеры итоговых сабмитов/meta
├── artifacts/
│   ├── folds/
│   ├── manual_review/
│   ├── submissions/
│   └── manifests/
├── docs/
│   ├── EXPERIMENTS_RU.md
│   ├── REPRODUCIBILITY_RU.md
│   ├── PRESENTATION_OUTLINE_RU.md
│   ├── RELEASE_CHECKLIST_RU.md
│   └── archive/PROJECT_ANALYSIS_FULL_SOURCE.md
├── repro/
│   ├── README.md
│   └── scripts/
├── assets/
├── data/README.md
└── weights/README.md
```

## Быстрый старт (для команды / проверки)

### 1) Окружение

Python `3.10` или `3.11`.

Сначала установить PyTorch под ваше железо:
- Apple Silicon (M1/M2): `torch/torchvision` с поддержкой `MPS`
- CUDA GPU: `torch/torchvision` с поддержкой `CUDA`

Потом общие зависимости:

```bash
pip install -r requirements_common_team_train.txt
```

### 2) Подготовить данные и веса

См.:
- `data/README.md`
- `weights/README.md`
- `docs/REPRODUCIBILITY_RU.md`

### 3) Запустить split-training (дедлайн-конфиг)

На M2 (ConvNeXt-S):

```bash
bash run_me_m2_convnext_small.sh
```

На CUDA (EffNetV2-S):

```bash
bash run_friend_gpu_effnetv2s.sh
```

На M1 (DeiT3-S):

```bash
bash run_friend_m1_deit3_small.sh
```

## Подтверждённые улучшения (коротко)

- Переход на `no_color` аугментации (A/B на одном фолде дал крупный прирост).
- `SAM + SWA` в дедлайн-конфиге (`SWA` с ранним стартом `swa_start_epoch=5`).
- `label_smoothing=0.03` для `ConvNeXt-S` в финальном конфиге.
- Tinder-style ручная чистка и relabeling train-данных (создан `top_new_dataset`).
- `LR(MSE)` blending + `geo8` TTA.
- Late-stage смешение `old + new` zoo (главный прирост до `0.97338`).

## Что не сработало / ухудшило (коротко)

- Сильные color-aug (подтверждённый вред в этой задаче).
- `stage4 confidence-aware finetune` в текущей реализации (нет прироста на probe).
- `mixed CWR` как финальный public-сабмит (локально обещал, на public хуже `mixed LR`).
- Pair-experts поверх уже сильного `mixed LR` (не ухудшили, но и не улучшили `0.97338`).
- Partial final3 default ensemble на неполном покрытии фолдов/моделей (`0.95882`, smoke only).

Подробности и таблицы: `docs/EXPERIMENTS_RU.md`.

## Ссылки на веса и большие бандлы

Реестр ссылок и статусов публикации:
- `weights/README.md`

Рекомендуется загрузить:
- `team_bundle_top_new_tinder_plus_ckpts_...zip`
- `team_final3_cv5_split_bundle_...zip`
- отдельные весовые архивы (например, `ConvNeXt-S f0/f1`)

## Полная хронология

Полный рабочий журнал (с локальными путями и всей последовательностью действий):
- `docs/archive/PROJECT_ANALYSIS_FULL_SOURCE.md`

Для публичного чтения и защиты используйте структурированные версии в `docs/`.
