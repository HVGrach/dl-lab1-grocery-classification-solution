# Weights And Large Artifacts (To Be Added)

В этом репозитории большие файлы не хранятся. Сюда нужно добавить ссылки на внешнее хранилище (Yandex Disk / Google Drive) после загрузки.

## Что загрузить (рекомендуемый минимум)

### 1. Team bundle: `top_new_dataset` + task-specific checkpoints

- Имя архива (пример): `team_bundle_top_new_tinder_plus_ckpts_2026-02-26_v1.zip`
- Что внутри:
  - `top_new_dataset` (после Tinder-cleaning)
  - fixed folds CSV
  - CV5 checkpoints (`ConvNeXt-S`, `EffNetV2-S`, `DeiT3-S`)
  - one-fold probes
  - scripts / manifest

Ссылка:
- Yandex Disk: `TODO`
- Google Drive: `TODO`
- Size: `TODO`
- SHA256: `TODO`

### 2. Team split training bundle (для распределённого обучения)

- Имя архива (пример): `team_final3_cv5_split_bundle_2026-02-26_v1.zip`
- Что внутри:
  - `top_new_dataset`
  - `folds_used_top_new_dataset_aligned_hybrid.csv`
  - `dl_lab1/scripts/`
  - launcher'ы `run_*.sh`
  - `requirements_common_team_train.txt`
  - `README_TEAM_FINAL3_CV5_SPLIT.md`

Ссылка:
- Yandex Disk: `TODO`
- Google Drive: `TODO`
- Size: `TODO`
- SHA256: `TODO`

### 3. Отдельные lightweight weight bundles (по необходимости)

Примеры:
- `convnext_small_deadline14f4_weights_f0f1_2026-02-26.zip`

Ссылка:
- Yandex Disk: `TODO`
- Google Drive: `TODO`
- Size: `TODO`
- SHA256: `TODO`

## Замечания

- Для reproducibility лучше давать и **bundle**, и **отдельные weight-only архивы**.
- Права доступа к ссылкам: "доступ по ссылке" без запроса разрешения.
- Если есть несколько версий архива, явно укажите какая является "canonical".

