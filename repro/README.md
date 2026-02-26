# Reproducibility Quick Path

Этот каталог — короткий маршрут воспроизведения без чтения всего проекта.

## Минимальный сценарий (рекомендуется)

1. Подготовить окружение:

```bash
pip install -r requirements_common_team_train.txt
```

2. Проверить структуру репозитория и наличие данных:

```bash
bash repro/scripts/01_check_layout.sh
```

3. Воспроизвести baseline-хедж (`0.97154` public):

```bash
bash repro/scripts/02_reproduce_full_cv5_submission.sh
```

4. Воспроизвести лучший late-stage mixed submission (`0.97338` public):

```bash
bash repro/scripts/03_reproduce_best_mixed_submission.sh
```

## Публикация архивов в Google Drive

Для массовой загрузки и получения публичных ссылок:

```bash
/opt/homebrew/bin/python3.11 repro/scripts/upload_to_google_drive.py \
  --mode oauth \
  --folder-id "<google_drive_folder_id>" \
  --client-secrets google_client_secret.json \
  --make-public \
  --files \
  /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_bundle_top_new_tinder_plus_ckpts_2026-02-26_v1.zip \
  /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1.zip \
  /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/convnext_small_deadline14f4_weights_f0f1_2026-02-26.zip
```

Скрипт создаёт:
- публичные ссылки на файлы;
- `repro/upload_manifest_google_drive.json` с `file_id`/URL/размером.

## Что подготавливается заранее

- `dl_lab1/top_new_dataset` (рабочий train/test layout)
- checkpoint bundles и zoo-артефакты (см. `weights/README.md`)
- fixed folds CSV уже включён в репозиторий:
  - `dl_lab1/outputs_post_tinder_convnext_cv2_compare/folds_used_top_new_dataset_aligned_hybrid.csv`

## Ожидаемые выходы

- CV5 хедж:
  - `dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8_equal.csv`
- Best mixed:
  - `artifacts/submissions/mixed_old_new/submission_mixed_lr_mse_geo8_oof_acc.csv` (reference)
  - или новый CSV в output-папке скрипта `mixed_old_new_orchestrator_submit.py` при полном пересчёте

## Проверка формата сабмита

Формат: `image_id,label`  
Для контроля можно сравнить с:
- `artifacts/submissions/cv5_all20/submission_cv5_all20_lr_geo8_equal.csv`
- `artifacts/submissions/mixed_old_new/submission_mixed_lr_mse_geo8_oof_acc.csv`
