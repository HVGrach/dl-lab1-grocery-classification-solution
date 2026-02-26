# DL Lab 1: Полный технический разбор проекта

Дата фиксации отчёта: 2026-02-18  
Последнее крупное обновление: 2026-02-26 (дедлайн-спринт, mixed old+new orchestrator, лучший public на текущем цикле: `0.97338`)

## 1. Цель проекта и ограничения

Цель: максимизировать качество классификации 15 классов (фрукты/овощи) на Kaggle-соревновании и получить устойчивый результат на private части.

Критичные ограничения:
- Тестовый датасет не изменяется вообще (не удаляем/не добавляем/не чистим тест-файлы).
- Очистка и любые манипуляции применяются только к `train`.
- Модель должна быть устойчивой к сдвигу распределения (не только «выглядеть хорошо» на public LB).
- Обучение ориентировано на Apple Silicon (`MPS`), не `CUDA`.

## 2. Что в итоге построено

Финальная система состоит из 3 уровней:

1) Базовые CV-модели (5-fold каждая):
- `convnext_small.fb_in22k_ft_in1k`
- `tf_efficientnetv2_s.in21k_ft_in1k`
- `resnet50.a1_in1k`

2) Улучшенный логит-ансамбль:
- random search по весам моделей;
- class-bias калибровка логитов на проблемных классах.

3) Мета-ансамбль (stacking):
- `HistGradientBoostingClassifier` на признаках из логитов/вероятностей базовых моделей + признаках pair-experts.

Дополнительно:
- обучены бинарные pair-experts для сложных пар:
  - `Киви vs Картофель` (`6 vs 5`)
  - `Яблоки красные vs Томаты` (`14 vs 12`)
  - `Мандарины vs Апельсин` (`9 vs 0`)

## 3. Данные и очистка train

Скрипт очистки: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/prepare_clean_train.py`

### 3.1. Что делает очистка

- Проверяет изображения по размеру, детализации, группе (`plu`), семантическому выбросу.
- Удаляет явный мусор.
- Отдельно маркирует сомнительные кейсы как `quarantine`.
- Оригинальные train-файлы не удаляются.

### 3.2. Итог очистки (из `summary.json`)

Источник: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/summary.json`

- Всего train строк: `9889`
- Drop: `80`
- Quarantine: `11`
- Strict (drop исключён, quarantine оставлен): `9809`
- Aggressive (drop + quarantine исключены): `9798`

Причины drop:
- `too_small_min_side_lt_56`: `76`
- `manual_semantic_noise`: `5`

### 3.3. Артефакты очистки

- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/train_clean_strict.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/train_clean_aggressive.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/train_clean_manifest.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/train_quarantine.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/cleaned_dataset_bundle.zip`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/cleaning_metadata_only.zip`

## 4. Базовый training pipeline

Основной скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/train_top1_mps.py`

### 4.1. Сплит и валидация

- `5-fold StratifiedKFold`
- Стратификация по `label + plu` (`strat_key = label_plu`)
- OOF-оценка используется как главный внутренний критерий качества.

### 4.2. Что использовано из регуляризации и стабилизации

- `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`
- `WeightedRandomSampler`
- `mixup` (`alpha=0.2`, вероятностно, отключается в последних эпохах)
- `AdamW`
- `Warmup + Cosine LR scheduler`
- `gradient clipping` (`1.0`)
- выбор лучшего чекпоинта по `val_loss` (`best_by_val_loss.pt`)
- TTA (горизонтальный флип) на инференсе.

### 4.3. Аугментации

Train:
- `RandomResizedCrop`
- `HorizontalFlip`
- геометрические (`ShiftScaleRotate`)
- цветовые (`ColorJitter` / `HueSaturationValue` / `RandomBrightnessContrast`)
- деградационные (`GaussianBlur` / `GaussNoise` / `ImageCompression`)
- `CoarseDropout`

Valid/Test:
- `SmallestMaxSize + CenterCrop`
- нормализация ImageNet.

### 4.4. MPS-специфика и исправления

- Обучение целиком на `device=mps`.
- Для стабильности:
  - у `convnext_small`: `channels_last=True`
  - у `effnetv2_s` и `resnet50`: `channels_last=False`
- Это связано с ранее пойманной MPS-ошибкой на backward (`view size is not compatible...`); текущая конфигурация её обходит.

## 5. Результаты базовых моделей и ансамбля

Источники:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/all_model_oof_metrics.json`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/*.json`

### 5.1. OOF по отдельным моделям

| Модель | OOF ACC | OOF F1-macro |
|---|---:|---:|
| convnext_small | 0.946886 | 0.943792 |
| effnetv2_s | 0.946274 | 0.943383 |
| resnet50 | 0.925171 | 0.921223 |

### 5.2. Эволюция ансамбля

| Этап | ACC | F1-macro | ΔACC к предыдущему этапу |
|---|---:|---:|---:|
| Базовый weighted ensemble | 0.956876 | 0.953979 | - |
| После weight search | 0.957080 | 0.954287 | +0.000204 |
| После class-bias | 0.957896 | 0.956048 | +0.000816 |
| Logistic specialists rerank | 0.958304 | 0.956406 | +0.000408 |
| Trained pair-experts rerank | 0.958609 | 0.956774 | +0.000305 |
| Meta HGB stack | 0.973086 | 0.968951 | +0.014477 |

Файлы сабмитов:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/submission_ensemble_oof_optimized.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/submission_ensemble_refined_bias.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/submission_ensemble_refined_specialist.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/submission_ensemble_refined_pair_experts.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/submission_meta_hgb_pair_stack.csv`

## 6. Pair-experts: как обучены и что дали

Скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/train_pair_experts_mps.py`

Режим:
- бинарная классификация (`num_classes=1`, `BCEWithLogitsLoss`)
- `pos_weight` для дисбаланса
- 5-fold CV, OOF + test probabilities
- TTA на инференсе.

Ключевой фикс:
- MPS не поддерживает `float64` для такого пайплайна; таргеты принудительно в `float32`.

Метрики pair-experts:

| Expert | AUC | F1@0.5 | F1@best_thr |
|---|---:|---:|---:|
| kiwi_vs_potato | 0.980517 | 0.872818 | 0.878412 |
| redapple_vs_tomato | 0.987364 | 0.944717 | 0.947880 |
| mandarin_vs_orange | 0.989342 | 0.960358 | 0.962587 |

Вывод:
- сами бинарные эксперты сильные;
- но при интеграции в общий классификатор они применяются только на «сомнительных» кейсах (по top-2 классу и малому margin), поэтому абсолютный прирост общего ACC небольшой, но стабильный.

## 7. Детальный error analysis

Ключевые файлы:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/ensemble_oof_top_confusions.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/hard_class_errors_top3.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/meta_hgb_top_confusions.csv`

### 7.1. Основные пары ошибок (до meta)

Топ путаниц:
- `Мандарины -> Апельсин` (38)
- `Яблоки красные -> Томаты` (19)
- `Картофель -> Лук` (16)
- `Апельсин -> Мандарины` (16)
- `Киви -> Картофель` (12)

### 7.2. Концентрация hard-ошибок по группам (plu)

Из `hard_class_errors_top3.csv`:
- всего hard-ошибок: `128`
- по классам:
  - `Яблоки красные`: `60`
  - `Лук`: `52`
  - `Киви`: `16`
- по группам:
  - `Лук/197`: `45`
  - `Яблоки красные/807`: `38`
  - `Яблоки красные/4271772`: `19`
  - `Киви/78647`: `16`

Это указывает, что часть ошибок имеет выраженный `plu`-сдвиг (групповой домен-шифт), а не только «общеклассовую» природу.

## 8. Аудит риска просадки на private

Скрипт:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/private_risk_audit.py`

Артефакт:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/private_risk_audit.json`

По текущему файлу:
- `meta_cv acc`: `0.9731`
- bootstrap p05 acc: `0.9703`
- repeated stratified holdout min acc: `0.9684`
- group holdout by `plu`:
  - median acc: `0.8604`
  - min acc: `0.4213`
  - runs below 0.85: `3/6`

Интерпретация:
- На IID/почти-IID разбиениях пайплайн очень устойчив.
- На групповых разбиениях по `plu` риск просадки высокий.
- Значит, риск «сильного domain shift на private» реален, и именно он главная угроза (не классическое переобучение по epoch-метрикам).

## 9. Что пробовали и как приняли решения

### 9.1. Принятые решения (в прод-пайплайне)

1. `strict` cleaning как основной train-вариант.
Причина: удаляет явный мусор, не выкидывая много данных.

2. 3 архитектуры + 5 folds (итого 15 чекпоинтов).
Причина: разнообразие inductive bias + устойчивость ансамбля.

3. Критерий сохранения чекпоинта по `val_loss`.
Причина: явно снижает риск брать переобучённые эпохи.

4. `label_smoothing + class_weights + WeightedRandomSampler`.
Причина: стабилизация на дисбалансе и редких классах.

5. Weight search + bias calibration.
Причина: дешёвый и воспроизводимый прирост без дообучения backbone.

6. Pair-experts только на uncertain-кейсах.
Причина: минимизировать побочный вред на уверенных предиктах.

7. Meta-stacking (HGB) на логитах + pair-feature.
Причина: крупнейший прирост OOF из всех подходов.

### 9.2. Подходы, которые пробовали, но не сделали основным

1. Logistic specialist rerank (`build_specialist_rerank.py`).
Причина отказа как основного: прирост есть, но меньше, чем у trained pair-experts, а логика менее прозрачна.

2. Aggressive cleaning как дефолт.
Причина: вариант подготовлен, но не выбран базовым без полноценного A/B по тем же CV-протоколам; риск удалить полезную вариативность.

3. Глобальное применение pair-expert на всех примерах пары.
Причина: ухудшает общую стабильность; оставлен только gated режим (top2 + margin).

4. `channels_last` для всех моделей на MPS.
Причина: для части архитектур приводило к runtime нестабильности; сохранено точечно.

## 10. Kaggle-формат и текущий статус

Формат сабмита:
- как в `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/sample_submission.csv`
- столбцы: `image_id,label`

Снимок leaderboard (получен 2026-02-18 через `kaggle competitions leaderboard ...`):
- лучший score в таблице: `0.95818` (команда `батчсайз не влез`)

Примечание:
- запрос `kaggle competitions submissions` в момент отчёта падал с DNS-ошибкой `api.kaggle.com`, поэтому детальную историю сабмитов в этот момент не выгрузили.

## 11. Полная карта файлов проекта

### 11.1. Основные тренировочные скрипты
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/train_top1_mps.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/train_pair_experts_mps.py`

### 11.2. Пост-обработка ансамбля
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/refine_existing_ensemble.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/build_specialist_rerank.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/apply_trained_pair_experts.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/meta_stack_hgb.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/private_risk_audit.py`

### 11.3. Очистка
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/prepare_clean_train.py`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/unzipped/cleaning/README.md`

### 11.4. Главные результаты
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/meta_hgb_summary.json`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/private_risk_audit.json`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_pair_experts_mps/pair_experts_summary.json`

## 12. Что делать дальше (если цель: стабильно удержать топ на private)

1. Перейти на group-aware meta-CV:
- в stacking-аудите использовать `StratifiedGroupKFold`/group holdout по `plu` как основной критерий отбора.

2. Точечная доработка доминантных проблемных групп:
- `Лук/197`, `Яблоки красные/807`, `Яблоки красные/4271772`, `Киви/78647`.

3. Усилить pair-подход:
- добавить ещё пары из `meta_hgb_top_confusions.csv` (например, `Морковь<->Лук`, `Апельсин<->Мандарины`) и оптимизировать gating под group-CV, а не только under OOF score.

4. Дальше выбирать финальный сабмит не только по public LB:
- gate по нескольким внутренним метрикам (OOF + group-holdout quantiles), чтобы не получить просадку после раскрытия private 70%.

## 13. Интеграция внешних моделей (peer checkpoints)

Внешние чекпоинты:
- `/Users/fedorgracev/Downloads/convnext_tiny_fold1_best.pth`
- `/Users/fedorgracev/Downloads/convnext_tiny_fold4_best.pth`

Скрипт проверки/интеграции:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/eval_blend_peer_convnext_tiny.py`

Что сделано:
1. Реконструирована архитектура из state_dict:
- backbone: `convnext_tiny` (timm), `head=Identity`, avg-pool по spatial.
- `main_classifier` (15 классов), `auxiliary_classifier` (43 классов).

2. Прогон на train/test и анализ:
- Peer-модели дали очень высокие train-метрики (признак сильной подгонки к train).
- `peer-only` на Kaggle показал `0.93855` (хуже базового ансамбля).

3. Безопасная интеграция:
- blending peer-предсказаний с нашим базовым лучшим ансамблем малым весом.
- лучший practical вариант: `beta=0.30` (no-TTA), файл:
  `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/submission_base_plus_peer_convnext_tiny.csv`

Факт по Kaggle:
- `submission_ensemble_oof_optimized.csv`: `0.95818`
- `submission_base_plus_peer_convnext_tiny.csv`: `0.96107` (лучший)
- `submission_peer_convnext_tiny_only.csv`: `0.93855`
- `submission_base_plus_peer_convnext_tiny_tta.csv`: `0.95995`
- `submission_base_plus_peer_convnext_tiny_beta025.csv`: `0.95995`

Вывод:
- Peer-модели стоит добавлять только как слабый компонент бленда.
- В текущем состоянии это дало прирост public `+0.00289` и укрепило top-1 позицию.

## 14. Новое открытие (2026-02-19): цветовые аугментации вредят в этой задаче

Источник A/B-теста:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_color_ablation_onefold_mps/one_fold_ablation_summary.json`

Постановка эксперимента:
- одна и та же модель: `convnext_small`;
- один и тот же split: `fold=0` (из 5-fold);
- одни и те же гиперпараметры (`epochs=12`, `batch_size=16`, `seed=42`);
- менялся только профиль аугментаций:
  - `full`: c color-трансформами (`ColorJitter/HueSaturationValue/RandomBrightnessContrast`);
  - `no_color`: без цветовых сдвигов (оставлены геометрия, окклюзии, деградации).

Результат:
- `full`: `acc=0.94954`, `f1_macro=0.94674`, `best_val_loss=0.57939`
- `no_color`: `acc=0.95872`, `f1_macro=0.95405`, `best_val_loss=0.54543`
- дельта `no_color - full`:
  - `+0.00917 acc`
  - `+0.00731 f1_macro`
  - `-0.03396 val_loss` (лучше)

Интерпретация:
- для текущего датасета сильные цветовые аугментации создают нежелательный сдвиг признаков и ухудшают различение близких классов по цвету;
- гипотеза команды подтверждена экспериментально.

Принятое решение:
1. в основном пайплайне перейти на профиль `no_color` как baseline;
2. цветовые трансформы отключить до отдельного controlled A/B;
3. дальнейшие улучшения строить через геометрию, окклюзии, расписание обучения и ансамблирование, а не через hue/saturation сдвиги.

Технические артефакты:
- обновлён тренировочный скрипт:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/train_top1_mps.py`
  - добавлен аргумент `--aug-profile {full,no_color}`
- добавлен full A/B launcher:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_color_ablation_mps.py`
- добавлен fast one-fold A/B launcher:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_color_ablation_one_fold_mps.py`
- добавлен визуальный дебаг аугментаций на hard-кейсах:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/debug_confusion_augs.py`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top1_mps/analysis/aug_debug/*.png`

## 15. План следующего цикла после отказа от color-aug

1. Подтверждение эффекта на ещё одном фолде:
- повторить one-fold A/B на `fold=1` (те же настройки), чтобы снизить риск случайной удачи на одном split.

2. Переобучение полного ансамбля в режиме `no_color`:
- `convnext_small`, `effnetv2_s`, `resnet50`, 5-fold;
- сравнить против текущего прод-базлайна по OOF и по качеству на hard pairs.

3. Longer fine-tune с двухэтапным LR:
- этап A: эпохи `1..12` (текущий lr);
- этап B: эпохи `13..20` с уменьшением lr в `3-5x`;
- критерий выбора: лучший `val_loss` + OOF итог, а не last epoch.

4. Переподготовка pair-experts в `no_color`-режиме:
- проверить, усилился ли вклад rerank в глобальный ансамбль.

5. Финальный отбор сабмита:
- приоритизировать конфигурации, которые дают лучший баланс:
  - public score,
  - OOF stability,
  - group-holdout risk (по `plu`).

## 16. Внешние проверенные стратегии (papers/challenges) и применимость к нашему проекту

Ниже — не «общие советы», а приёмы, которые уже показывали прирост на fine-grained/product задачах и/или под domain shift.

### 16.1. Что внедрять в первую очередь (высокий ROI)

1. High-resolution fine-tune для fine-grained SKU.
- Основание: Products-10K показывает выраженный прирост при переходе с `224` на `448` для SKU-распознавания.
- Как применить у нас: stage-2 дообучение лучших моделей (`img_size=320/384`) после базового `224`, с пониженным LR.

2. Balanced-subset fine-tune после обучения на полном train.
- Основание: Products-10K отдельно подчёркивает, что краткий дофайнтюн на сбалансированном подмножестве снижает bias long-tail.
- Как применить у нас: 2-4 финальные эпохи на class-balanced subset (ограничение числа примеров на класс).

3. Weight averaging вместо «выбора одной лучшей эпохи».
- Основание: SWA и Model Soup улучшают generalization без увеличения инференс-стоимости.
- Как применить у нас:
  - SWA внутри каждого фолда;
  - post-hoc weight soup для близких конфигов одной архитектуры (например, ConvNeXt no_color run-ы).

4. Регуляризация «плоских минимумов» (SAM).
- Основание: SAM улучшает обобщение и устойчивость к шуму меток.
- Как применить у нас: включать на stage-2 (короткий fine-tune), чтобы не удваивать стоимость полного обучения.

5. Перенос валидационного критерия на worst-group (plu-aware).
- Основание: Group DRO показывает, что средняя accuracy может скрывать провал на атипичных группах.
- Как применить у нас: выбирать финальную модель/бленд не по mean-CV, а по комбинации `mean + worst-group quantile`.

### 16.2. Что внедрять во вторую очередь (средний ROI)

1. CutMix вместо/вместе с mixup в no_color-профиле.
- Основание: CutMix стабильно улучшает классификацию и robustness, часто лучше «чистого» cutout.
- Как применить у нас: low-probability режим (`p~0.2-0.4`), сравнить с текущим mixup-only.

2. Random Erasing как основной occlusion-regularizer.
- Основание: работает как простая и надёжная защита от частичных перекрытий.
- Как применить у нас: аккуратно усилить текущий dropout/erasing блок без цветовых искажений.

3. RandAugment / AugMix в «геометрия+деградации» подмножестве операций.
- Основание: auto-augmentation даёт прирост и устойчивость к corruption shift.
- Важно: исключить/ограничить color-op для нашей задачи (подтверждён вред color-aug).

4. Температурная калибровка для rerank-gating.
- Основание: у современных сетей confidence часто некалиброван; temperature scaling существенно помогает.
- Как применить у нас: калибровать logits на OOF перед порогами specialist/pair rerank.

### 16.3. Что потенциально даёт «рывок», но дороже и рискованнее

1. Полу-supervised дообучение (FixMatch) на псевдо-метках test-like/unlabeled данных.
- Плюс: может резко улучшить representation на малом labeled train.
- Риск: если псевдо-метки шумные — легко ухудшить private.

2. Retrieval-ветка для сложных пар (embedding + kNN) поверх классификатора.
- Основание: DIHE для grocery использует domain-invariant embedding + kNN и работает с 1 reference image на товар.
- Как применить у нас: specialist-модуль для 2-5 самых конфликтных пар классов.

3. Self-supervised pretrain (SimCLR / DINOv2 features) + supervised head.
- Плюс: лучшее начальное представление под domain shift.
- Минус: вычислительно тяжелее, нужен аккуратный протокол сравнения.

### 16.4. Привязка к нашему текущему состоянию

У нас уже подтверждено:
- `no_color` > `full` на same-fold A/B (`+0.00917 acc`).

Следовательно, следующий рациональный маршрут:
1. Полный retrain ансамбля в `no_color`.
2. Stage-2 (12→20 эпох) c LR x(1/3..1/5), плюс SWA.
3. Refit pair-experts в `no_color` + калибровка порогов.
4. Финальный выбор по `public + OOF + group(plu)-risk`.

### 16.5. Источники для раздела 16

- Products-10K: https://arxiv.org/abs/2008.10545
- Freiburg Groceries Dataset: https://arxiv.org/abs/1611.05799
- Domain Invariant Hierarchical Embedding (DIHE): https://arxiv.org/abs/1902.00760
- Retail Product Checkout (RPC): https://arxiv.org/abs/1901.07249
- SWA: https://arxiv.org/abs/1803.05407
- Model Soups: https://arxiv.org/abs/2203.05482
- SAM: https://arxiv.org/abs/2010.01412
- Group DRO: https://arxiv.org/abs/1911.08731
- CutMix: https://arxiv.org/abs/1905.04899
- Random Erasing: https://arxiv.org/abs/1708.04896
- RandAugment: https://arxiv.org/abs/1909.13719
- AugMix: https://arxiv.org/abs/1912.02781
- FixMatch: https://arxiv.org/abs/2001.07685
- Temperature Scaling / calibration: https://arxiv.org/abs/1706.04599
- SimCLR: https://arxiv.org/abs/2002.05709
- DINOv2: https://arxiv.org/abs/2304.07193

## 17. Ночной автопилот (новый цикл, февраль 2026)

### 17.1 Что добавлено в код

1. Обновлён `train_onefold_no_color_innov_mps.py`:
- добавлен выбор устройства `--device {auto,mps,cpu}`;
- добавлено сохранение артефактов для честного ансамбля на одном фолде:
  - `val_logits.npy`
  - `val_probs.npy`
  - `val_labels.npy`

2. Добавлен оркестратор ночного прогона:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py`

Что делает оркестратор:
- запускает набор разнотипных one-fold экспериментов (CNN + Transformer, разные режимы регуляризации);
- пишет единый рейтинг (`run_ranking.csv`);
- автоматически считает ансамбли по вероятностям (`ensemble_candidates.csv`);
- сохраняет итог в `autopilot_summary.json`.

### 17.2 Почему именно такие эксперименты

Подбор сделан по подтверждённым подходам из литературы:
- Mixup: улучшение обобщения и устойчивости (`arXiv:1710.09412`)
- CutMix: прирост точности и robustness (`arXiv:1905.04899`)
- SAM: улучшение generalization через плоские минимумы (`arXiv:2010.01412`)
- SWA: устойчивое улучшение качества и калибровки (`arXiv:1803.05407`)
- ConvNeXt как сильный CNN-базис (`arXiv:2201.03545`)
- EfficientNetV2 как эффективный accuracy/speed-компромисс (`arXiv:2104.00298`)
- DeiT/ViT для альтернативной индуктивной bias-модели (`arXiv:2012.12877`, `arXiv:2010.11929`)

Плюс использованы актуальные рекомендации PyTorch/Albumentations:
- `CrossEntropyLoss(label_smoothing=...)`, class weights;
- `torch.optim.swa_utils` для SWA/EMA;
- аккуратный no-color режим аугментаций с акцентом на геометрию/окклюзии.

### 17.3 Что запускать на ночь

Базовый ночной сценарий (ориентир ~10 часов, зависит от железа):

```bash
/opt/homebrew/bin/python3.11 /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py \
  --preset overnight_10h \
  --base /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1 \
  --clean-variant strict \
  --folds-csv /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps/folds_used.csv \
  --fold-idx 0 \
  --seed 42 \
  --device auto \
  --num-workers 0 \
  --out-root /Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps
```

Утром смотреть:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/run_ranking.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/ensemble_candidates.csv`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/autopilot_summary.json`

### 17.4 Важно по рискам

1. One-fold нужен для быстрого отбора, но финальный вывод по quality делать только после full CV.
2. Если у части моделей отсутствуют веса в кэше timm/HF, первый запуск может тратить время на загрузку.
3. Главный критерий перехода в full-CV: стабильный выигрыш не только по `acc`, но и по `f1_macro` и по ошибкам в hard-pairs.

## 18. Хронологический журнал работ (для защиты лабораторной)

Правило ведения (введено): каждая заметная итерация фиксируется с датой `YYYY-MM-DD`, что пробовали, результат, решение.

### 2026-02-20

1. Завершена ручная доочистка train-данных и собран общий датасет команды:
- рабочий датасет: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/dataset_team_corrected_v1`
- контрольные артефакты чистки: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/*`
- итог: классы/CSV синхронизированы, test не изменялся.

2. Проведена проверка post-clean качества на том же one-fold протоколе:
- запуск: `run_color_ablation_one_fold_mps.py` (`no_color`, `fold=0`, `seed=42`)
- результат: `val_acc=0.95947`, `val_f1_macro=0.95351`.

3. Проведён детальный разбор ошибок и сравнительный аудит old vs new:
- скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/analyze_post_clean_delta.py`
- артефакты: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_onefold_post_clean_compare_mps/analysis_delta/*`
- вывод: основной остаточный риск — пары `Мандарины↔Апельсин`, `Киви↔Картофель`, `Яблоки красные↔Лук/Томаты`.

4. Добавлен ночной автопилот разнотипных пайплайнов (CNN + Transformer):
- новый скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py`
- режимы: `quick`, `overnight_10h`, `full`
- автовыходы: `run_ranking.csv`, `ensemble_candidates.csv`, `autopilot_summary.json`.

5. Обновлён one-fold trainer под автопилот:
- файл: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/train_onefold_no_color_innov_mps.py`
- добавлено:
  - `--device {auto,mps,cpu}`
  - сохранение `val_logits.npy`, `val_probs.npy`, `val_labels.npy` для честного ансамблирования.

6. Зафиксированы источники и rationale для новых экспериментов:
- отчёт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/night_pipeline_research_2026-02-20.md`
- опора: Mixup, CutMix, SAM, SWA, ConvNeXt, EfficientNetV2, DeiT/ViT, рекомендации PyTorch/Albumentations.

### 2026-02-19

1. Подтверждена гипотеза, что color-aug ухудшает задачу:
- A/B `full` vs `no_color` на одном и том же split (`fold=0`)
- прирост `no_color`: `+0.00917 acc`, `+0.00731 f1_macro`, снижение `val_loss`.

2. Принято решение:
- цветовые трансформы не использовать в baseline;
- улучшения строить через геометрию, регуляризацию, расписание LR и ансамбли.

### 2026-02-21

1. Выполнен ночной автопрогон `overnight_10h` (4 разнотипные one-fold модели, `fold=0`, `seed=42`):
- оркестратор: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py`
- лог: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/night.log`
- summary: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/autopilot_summary.json`

2. Индивидуальные результаты моделей:
- `cnn_effnetv2_s_sam_swa`: `acc=0.97211`, `f1_macro=0.96958` (лучший single);
- `cnn_convnext_small_sam_swa`: `acc=0.96947`, `f1_macro=0.96780`;
- `vit_deit3_small_color_safe`: `acc=0.96053`, `f1_macro=0.95544`;
- `vit_base_augreg_lite`: `acc=0.91789`, `f1_macro=0.90801` (явный аутсайдер).
- файл ранжирования: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/run_ranking.csv`

3. Лучший ансамбль (автопоиск весов):
- состав: `effnetv2_s + convnext_small + deit3_small`;
- веса: `[0.4136, 0.3261, 0.2603]`;
- one-fold метрики ансамбля: `acc=0.97526`, `f1_macro=0.97418`.
- файл кандидатов: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_mps/ensemble_candidates.csv`

4. Решение после прогона:
- оставить `vit_base_augreg_lite` вне следующего цикла (даёт просадку);
- в full-CV цикл переносить связку `effnetv2_s + convnext_small + deit3_small`;
- следующий этап: полноценный CV/OOF на выбранной тройке и только после этого финальный Kaggle submit.

5. Что именно дало высокий `val` (консолидированный вывод по текущему циклу):
- отказ от color-аугментаций (`no_color`), сохранение цветовой семантики классов;
- сильные pretrained-бэкбоны с разной индуктивной bias:
  - `tf_efficientnetv2_s.in21k_ft_in1k`
  - `convnext_small.fb_in22k_ft_in1k`
  - `deit3_small_patch16_224.fb_in22k_ft_in1k`;
- регуляризация обучения:
  - `label_smoothing`
  - class weights
  - `mixup/cutmix`
  - `SAM`
  - `SWA` (включая выбор SWA-модели, где она лучше по `val_loss`);
- ручная чистка train-шума (мусор/явные ошибочные классы);
- weighted-ансамбль поверх лучших моделей:
  - лучший one-fold blend: `acc=0.97526`, `f1_macro=0.97418`.

6. Имплементирован гибкий TTA для финального сабмита:
- обновлён скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/make_submission_from_night_best.py`
- добавлены режимы: `--tta-mode {none,flip,geo4,geo8}`
- добавлен лимит количества вьюх: `--tta-views N`
- важный инвариант: только геометрические преобразования (без color-сдвигов), чтобы не повторить деградацию из color-aug этапа.

7. Имплементировано мульти-сид обучение одинаковых архитектур для увеличения разнообразия ансамбля:
- обновлён оркестратор: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py`
- добавлены параметры:
  - `--extra-seeds` (дополнительные сиды, например `133,777`)
  - `--extra-seed-target` (какие модели клонировать по сиду; фильтр по name/group/model)
- для каждого клона сохраняется отдельный run (`..._seed133`, `..._seed777`) и отдельные метрики;
- проверено dry-run сценариями: клоны корректно формируются и запускаются одной командой.

8. Имплементирован режим k-fold cross-validation в оркестраторе:
- файл: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py`
- новые параметры:
  - `--cv-folds 0,1,2,3,4` (явный список фолдов)
  - `--cv-all` (все фолды из `folds_csv` или `range(n_splits)`)
- оркестратор теперь запускает каждую модель на каждом выбранном фолде;
- добавлена CV-сводка по моделям:
  - `/.../cv_model_summary.csv` (`mean/std` по `acc/f1/loss` и суммарное время);
- ансамбли считаются отдельно по каждому фолду, чтобы не смешивать разные валидационные подвыборки.

### 2026-02-22

1. Изменён метод подбора весов ансамбля в `night_model_zoo_autopilot_mps.py`:
- старый вариант: случайный поиск весов (`Dirichlet random search`) с подбором по `val_acc/f1`;
- новый вариант: линейная регрессия (`MSE`) по `one-hot` целям на `val_probs` каждой модели.

2. Техническая реализация нового ансамблирования:
- используется `sklearn.linear_model.LinearRegression(fit_intercept=False, positive=True)`;
- признаки: вероятности моделей, развернутые в матрицу `[N * C, M]`;
- цель: `one-hot(y)` развернутый в вектор `[N * C]`;
- после обучения веса приводятся к convex-комбинации (обрезка `<0` и нормализация суммы к `1`), чтобы сохранять корректный масштаб blended probabilities.

3. Проверка без переобучения (на уже готовых артефактах `outputs_night_model_zoo_cv5`):
- ансамбль пересчитан успешно по сохранённым `val_probs.npy`/`val_labels.npy`;
- генерация `ensemble_candidates.csv` и `per_fold` отчёта не сломалась;
- метод теперь детерминированный (при одинаковых `val_probs` выдаёт те же веса, без разброса от random search).

4. Итоги большого ночного CV-прогона (`outputs_night_model_zoo_cv5`, ~18.5ч):
- режим: `overnight_10h` + `k-fold` по фолдам `0..4`;
- всего запусков: `20` (`4 модели x 5 фолдов`);
- успешно: `20/20`, ошибок запуска: `0`.

5. Средние CV-метрики (5 фолдов):
- `cnn_convnext_small_sam_swa`: `val_acc_mean=0.97209`, `val_f1_macro_mean=0.96885`;
- `cnn_effnetv2_s_sam_swa`: `val_acc_mean=0.97178`, `val_f1_macro_mean=0.96783`;
- `vit_deit3_small_color_safe`: `val_acc_mean=0.96546`, `val_f1_macro_mean=0.96165`;
- `vit_base_augreg_lite`: `val_acc_mean=0.92144`, `val_f1_macro_mean=0.91238` (слабый single, но иногда даёт небольшой ансамблевый буст).

6. Практический вывод после CV5:
- `ConvNeXt-S` и `EffNetV2-S` — основа сильного ансамбля (стабильные и близкие по качеству);
- `DeiT3-Small` слабее как single, но полезен как источник разнообразия ошибок;
- `ViT-Base augreg` как single слабый, в большинстве сценариев неэффективен по времени, вероятный кандидат на исключение или понижение приоритета в следующих циклах.

7. Построен и отправлен в Kaggle full CV-ансамбль из всех `20` моделей (5 фолдов x 4 модели):
- скрипт инференса: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/make_submission_from_cv5_all20_lr_tta.py`
- схема:
  - внутри каждого фолда (`4` модели) веса считаются через `LinearRegression` (`MSE` по `one-hot` целям на `val_probs`);
  - на test считается `geo8` TTA (только геометрия, без color-сдвигов);
  - затем фолд-ансамбли усредняются равными весами (`1/5`).
- артефакты:
  - CSV: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8_equal.csv`
  - meta: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8_equal_meta.json`

8. Практический результат сабмита (public LB):
- `submission_cv5_all20_lr_geo8_equal.csv` -> `0.97154` (public score)
- это выше предыдущего `night_best3_weighted_tta` (`0.96917`).

9. Наблюдение по LR-весам внутри фолдов:
- `vit_base_augreg_lite` получил вес `0.0` во всех 5 фолдах (регрессия сама его отключила);
- основные веса распределяются между `ConvNeXt-S`, `EffNetV2-S` и `DeiT3-S`, что подтверждает предыдущий вывод о полезности `DeiT3` как источника разнообразия и слабости `ViT-Base` в этом сетапе.

10. Подготовка новой итерации ручной чистки датасета:
- создан новый рабочий каталог: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset`;
- исходный архив `dl-lab-1-image-classification.zip` распакован заново в свежую папку;
- первая попытка через системный `unzip` дала ошибки из-за декодирования русских имён (частичная битая распаковка удалена);
- итоговая корректная распаковка выполнена через `Python zipfile` (русские названия классов сохранены);
- проверено наличие структуры (`train/train`, `test_images/test_images`, `train.csv`, `test.csv`, `sample_submission.csv`), классов `15`.

11. Реализован новый advanced meta-stacking пайплайн поверх `CV5` OOF-предсказаний:
- скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/meta_stack_cv5_attention.py`
- поддерживаемые метамодели:
  - `LogisticRegression` (multiclass, вероятности)
  - `CatBoostClassifier` (multiclass, вероятности)
  - `AttentionMetaEnsemble` (transformer-based dynamic stacking по токенам `[модель, probs(15)]`)
- честная OOF-оценка делается по base-fold’ам (`fold_id` из CV5): метамодель обучается на 4 фолдах и валидируется на 1.

12. Smoke-проверка нового meta-скрипта:
- режим: `--methods logreg --skip-test-infer` (без test-инференса, только OOF evaluation)
- результат OOF (`logreg`): `acc=0.97199`, `f1_macro=0.96951`
- подтверждено, что сбор OOF-признаков из `20` run’ов и мета-валидация по `5` фолдам работают корректно.

### 2026-02-25

1. Пересмотр стратегии ручной чистки нового датасета (`top_new_dataset`):
- пользователь отклонил применение `/Users/fedorgracev/Downloads/dataset_clearing.xlsx` к `top_new_dataset` как финальный источник правок;
- причина: файл покрывает только часть классов и отражает более агрессивную (неконсервативную) чистку.

2. Откат `top_new_dataset` к исходному состоянию:
- каталог `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset` удалён;
- датасет распакован заново из `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/dl-lab-1-image-classification.zip` через `Python zipfile`;
- цель: начать консервативную чистку с полностью чистой копии без частичных правок.

3. Статус:
- текущее состояние `top_new_dataset` = свежая распаковка исходного архива;
- предыдущий проход с `dataset_clearing.xlsx` считать отменённым для этой папки.

4. Реализован локальный сайт для ручной чистки датасета в стиле "Tinder" (по классам):
- скрипт приложения: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/dataset_tinder_review_app.py`;
- интерфейс позволяет выбирать класс и проходить все изображения подряд без повторов в рамках сессии;
- управление с клавиатуры: `← = OK`, `→ = подозрительное`, `Backspace/Z = Undo`;
- добавлены раздельные решения: `мусор` и `другой класс (relabel)`;
- результаты сессии сохраняются в `outputs_manual_review/tinder_swipe_review/session_<timestamp>/...`.

5. В сайт добавлена подсказка по модели (проценты ансамбля) для ускорения ручной разметки:
- поверх интерфейса показываются вероятности лучшего ансамбля по текущему изображению (`% по классам`);
- текущий выбранный класс принудительно выводится первым в списке, чтобы быстрее принимать решение;
- проценты визуально вынесены в верхний правый блок (заметный UI-элемент для ручной чистки).

6. Построен confidence-cache для `top_new_dataset` на основе лучшего ансамбля:
- скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/build_tinder_confidence_cache.py`;
- использована гибридная схема:
  - `OOF` вероятности из сохранённых `CV5` артефактов для уже известных train-изображений;
  - `MPS` инференс ансамбля для новых/изменённых путей;
- итоговое покрытие: `9889 / 9889` изображений (`9433 OOF + 456 MPS`);
- cache-файлы:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/confidence_cache/top_new_dataset_ensemble_confidence.csv`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/confidence_cache/top_new_dataset_ensemble_confidence.meta.json`

7. Ручная чистка через новый сайт завершена пользователем (полный проход всех классов):
- пользователь потратил около `2 часов` на ручной просмотр изображений;
- пройдено `15/15` классов, проверено `9889 / 9889` изображений;
- журнал сессии: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/session_20260225_203059`;
- в сессии активно использовался `Undo` (миссклики корректировались без потери консистентности журнала).

8. Экспортированы финальные действия из сессии (с replay `decision + undo`):
- скрипт экспорта: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/export_tinder_session_actions.py`;
- итоговые решения после учёта всех `Undo`:
  - `keep = 9696`
  - `relabel = 139`
  - `trash = 54`
- совместимый `actions.csv` для применения:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/session_20260225_203059/export_final_actions/actions_for_apply_manual_actions.csv`

9. Применены финальные правки к `top_new_dataset` (файлы + метаданные):
- скрипт применения: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/apply_tinder_actions_to_top_new_dataset.py`;
- выполнено `193` файловых операций (`139 relabel + 54 trash`);
- `trash` перемещён в карантин вне `train/train`, чтобы не появлялся как новый "класс" в интерфейсе:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/_manual_quarantine_batches/tinder_session_20260225_203059`
- обновлены:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/train.csv`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/cleaning/manual_review_queue/manifest.csv`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/cleaning/manual_review_queue/queue_summary.json`
- сделаны backup исходных CSV перед перезаписью:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/train.pre_tinder_session_20260225_203059_20260225_225340.backup.csv`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset/cleaning/manual_review_queue/manifest.pre_tinder_session_20260225_203059_20260225_225340.backup.csv`

10. Валидация после применения ручной чистки:
- `top_new_dataset/train.csv`: `9835` строк;
- активных изображений на диске в `top_new_dataset/train/train` (без карантина): `9835`;
- `train.csv` и файловая структура совпадают (`1:1`);
- дубликатов `image_id` нет;
- несоответствий `label` vs папка класса нет (`0` invalid rows).

11. Количественный срез изменений `top_new_dataset` относительно старого рабочего набора (`dataset_team_corrected_v1/strict`):
- старый рабочий train (`strict`): `9496` строк;
- новый `top_new_dataset/train.csv` после ручной чистки: `9835` строк (то есть это не только "очистка", но и более широкий train);
- число уникальных `class/plu`-групп выросло: `69 -> 103`;
- по ключу `plu/filename`:
  - пересечение со старым набором: `9477`;
  - только в старом: `19`;
  - только в новом: `358`;
  - relabel на общих файлах: `88` (путь/класс изменён при сохранении того же файла).

12. Запущен `post-tinder` A/B для `ConvNeXt-S + SAM + SWA` на `top_new_dataset` (ночь `2026-02-25` -> завершение локально `2026-02-26`):
- запуск (CV2) начат на фолдах `0,1`, но по решению пользователя `fold=1` принудительно остановлен и оставлен только `fold=0`;
- оркестратор: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/night_model_zoo_autopilot_mps.py`;
- output root: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42`;
- артефакты завершённого `fold=0`:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42/runs/001_cnn_convnext_small_sam_swa_f0/summary.json`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42/runs/001_cnn_convnext_small_sam_swa_f0/epoch_log.csv`

13. Итог нового `fold=0` (на `top_new_dataset`, `raw`, aligned folds):
- финально выбран `SWA` (`final_model_selected = swa_model`);
- `final_metrics`: `val_loss=0.459683`, `val_acc=0.978680`, `val_f1_macro=0.976709`, `val_errors=42`, `val_size=1970`;
- базовый checkpoint (без SWA) был слабее на этом же фолде: `val_loss=0.460863`, `val_f1_macro=0.975411`.

14. Важно по корректности сравнения:
- прямое сравнение `old_cv5_fold0` vs `new_fold0` по их собственным `summary.json` не является полностью честным A/B (разные датасеты и разные вал-выборки: `1900` vs `1970`).
- поэтому дополнительно выполнен честный replay-сценарий: оба checkpoint'а прогнаны на одном и том же `top_new_dataset/fold0`.

15. Честный A/B на одном и том же `top_new_dataset/fold0` (старый baseline checkpoint vs новый post-tinder checkpoint):
- скрипт сравнения: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/analyze_post_clean_delta.py`
- offline-патч в скрипт внесён отдельно (инференс `pretrained=False`), чтобы исключить сетевые ретраи timm/HF при сравнении checkpoint'ов.

15.1. Если сравнивать `old(best)` vs `new(best_by_val_loss)`:
- `new(best)` слегка хуже:
  - `Δacc = -0.000508`
  - `Δf1_macro = -0.000839`
  - `Δerrors = +1`
- артефакт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/analysis_fold0_old_vs_new_on_top_new_dataset/summary.json`

15.2. Если сравнивать `old(best)` vs `new(final selected = SWA)` (это корректный итоговый A/B для нового run):
- `acc`: паритет (`0.978680 -> 0.978680`);
- `errors`: паритет (`42 -> 42`);
- `f1_macro`: небольшой прирост у нового (`+0.000459`);
- `mean_conf`: небольшой прирост у нового (`0.948237 -> 0.948381`);
- `status_counts`: `1936 unchanged`, `17 improved`, `17 worsened` (перераспределение ошибок без роста общего числа ошибок).
- артефакт:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/analysis_fold0_old_best_vs_new_swa_on_top_new_dataset/summary.json`

16. Локальный разбор изменений ошибок (`old best` vs `new SWA` на одном `top_new_dataset/fold0`):
- положительный net по классам:
  - `Лук` (`+2`)
  - `Бананы`, `Капуста`, `Морковь`, `Огурцы`, `Яблоки красные` (`+1`)
- отрицательный net по классам:
  - `Лимон` (`-4`)
  - `Мандарины` (`-2`)
  - `Апельсин` (`-1`)
- это согласуется с тем, что после правок датасета общий выигрыш пришёл не через уменьшение числа ошибок, а через перераспределение ошибок и улучшение macro-F1.

17. Проба 4-й стадии обучения (confidence-aware stage4 finetune, `fold=0`, MPS):
- новый скрипт: `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/finetune_stage4_confidence_mps.py`
- идея:
  - `hard` (низкая уверенность + train-ошибки checkpoint'а): только аккуратные повороты (`rotate-only`);
  - `easy`: сильные no-color геометрические/crop/occlusion аугментации.
- split перед дообучением:
  - `n_train=7865`
  - `n_hard=1968` (`25%`)
  - `n_easy=5897`
  - все train-ошибки базового checkpoint'а (`5/5`) попали в `hard`
- артефакты stage4-пробы:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42_stage4_probe_f0/summary.json`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/convnext_sam_swa_cv2_seed42_stage4_probe_f0/epoch_log_stage4.csv`

18. Итог stage4-пробы на `fold=0`:
- улучшения нет: `best_stage4_epoch = 0` (сохранён исходный checkpoint без stage4-обновлений);
- `delta_final_minus_base = 0` по всем метрикам (`val_loss`, `val_acc`, `val_f1_macro`, `val_errors`);
- по epoch-логам stage4-эпохи давали худший `val_loss`, чем база:
  - epoch1: `0.269743`
  - epoch2: `0.272384`
  - epoch3: `0.270283`
  - база: `0.254852`
- вывод: текущая конфигурация stage4 (как реализована в этой пробе) неэффективна для `ConvNeXt-S` на `top_new_dataset/fold0`; в ночной full-CV запуск без дополнительной переработки включать не стоит.

19. Подготовлены утилиты для следующего цикла экспериментов (чтобы ускорить ночные прогоны):
- batch stage4 для zoo-run'ов с обновлением `run_ranking.csv`/`cv_model_summary.csv`:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/apply_stage4_to_zoo_runs.py`
- probe-скрипт для `fold=0` (old/new compare + stage4 test):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_post_tinder_fold0_stage4_probe.sh`
- ночной пайплайн (base CV5 -> stage4 on `3x5` runs -> `LR(MSE)+geo8` submit):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_top_new_dataset_cv5_stage4_night.sh`

20. Подготовлен и внедрён адаптивный ночной пайплайн для `top_new_dataset` (идея: сначала быстрый отбор фичей на one-fold, потом full-CV):
- основной скрипт:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/adaptive_top_new_night_pipeline.py`
- wrapper:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_top_new_dataset_adaptive_night.sh`
- архитектура пайплайна:
  - `phase1`: `ConvNeXt-S` one-fold feature probes (greedy-проверка фичей);
  - `phase2`: mini-zoo probe (подбор состава моделей);
  - `phase3`: full CV zoo;
  - `phase4`: `LR(MSE)+TTA` submit;
  - `phase4B`: meta-gating (`logreg/catboost/attn`) и optional submit.

21. В adaptive-пайплайн добавлен автосабмит в Kaggle по порогу внутренней метрики:
- условие: `OOF acc >= 0.97000`;
- лимит отправок: до `2` сабмитов за запуск;
- поддержка `kaggle competitions submissions -v` после отправки;
- это позволило безопасно автоматизировать ночной цикл без ручного контроля каждой стадии.

22. Пройден большой `phase1` (feature probes на `ConvNeXt-S`, `top_new_dataset`, `fold=0`) и получен сильный прикладной сигнал для дедлайн-режима:
- артефакты:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_adaptive_night_run1/phase1_feature_probes/runs/...`
- завершённые ключевые пробы (по `summary.json`):
  - `000_baseline`: `acc=0.97462`, `f1=0.97126`
  - `003_swa_earlier`: `acc=0.97614`, `f1=0.97478` (SWA раньше помогает)
  - `004_label_smoothing_low (0.03)`: `acc=0.97766`, `f1=0.97363` (лучший стабильный сигнал)
  - `008_cutmix_only`: `acc=0.97665`, `f1=0.97282` (сильный, но не лучше лидера)
  - `009_sam_off`: `acc=0.97513`, `f1=0.97329` (хуже -> `SAM` оставляем)
  - `010_swa_off`: `acc=0.97310`, `f1=0.97147` (хуже -> `SWA` оставляем)
- важный нюанс:
  - `005_label_smoothing_high (0.08)` по `final_metrics` хуже, но его `SWA` был очень сильным (`acc=0.977665`, `f1=0.976246`), что показало чувствительность автоселектора к критерию выбора (`val_loss` vs `acc/f1`).
- итоговые решения из `phase1` для дедлайн-конфига:
  - оставить `SAM`, `SWA`, `weighted_sampler`, `mixup+cutmix`, `no_color`;
  - использовать `SWA` раньше (`swa_start_epoch=5`);
  - использовать `label_smoothing=0.03`.

23. Из-за жёсткого дедлайна adaptive-пайплайн не был доведён до полного `phase3/phase4` в изначальном виде:
- `phase1` дал полезные решения;
- дальше был выполнен переход на специальный дедлайн-режим с распределением обучения между несколькими машинами.

24. Собран отдельный train-only оркестратор для командного распараллеливания `3 модели x 5 фолдов` (с последующей адаптацией к `14 эпох / 4 фолда`):
- оркестратор:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/top_new_final3_cv5_train_orchestrator.py`
- wrapper:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/run_top_new_final3_cv5_train.sh`
- поддержка фильтров:
  - `--models`
  - `--folds`
  - `--resume`
  - `--epochs-override`
  - `--stage1-epochs-override`
- в конфиг CNN по умолчанию встроены решения из `phase1`:
  - `SAM on`, `SWA start=5`, `label_smoothing=0.03`, `weighted_sampler on`, `mixup+cutmix`.

25. В тренер и оркестратор добавлена совместимость с `CUDA` (кроме `MPS`), чтобы часть команды могла обучать на GPU:
- изменён тренер:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/train_onefold_no_color_innov_mps.py`
- изменён оркестратор:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/top_new_final3_cv5_train_orchestrator.py`
- дополнительно создан Windows/PyCharm one-click launcher для GPU-друга:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/run_friend_gpu_effnetv2s_pycharm.py`

26. Собран командный split-bundle для распределённого обучения (M2 + GPU + M1 + Colab):
- архив:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1.zip`
- содержимое:
  - `top_new_dataset`
  - `folds_used_top_new_dataset_aligned_hybrid.csv`
  - нужные `scripts/`
  - `run_me_m2_convnext_small.sh`
  - `run_friend_gpu_effnetv2s.sh`
  - `run_friend_m1_deit3_small.sh`
  - `run_friend_colab_effnetv2s_classaware_harden.sh`
  - `run_friend_gpu_effnetv2s_pycharm.py`
  - `requirements_common_team_train.txt`
  - `README_TEAM_FINAL3_CV5_SPLIT.md`
- дедлайн-обновление bundle:
  - переведён на режим `14 epochs / 4 folds (0..3)`.

27. Отдельно подготовлен архив для передачи данных + task-specific checkpoints команде (до старта split-обучения):
- архив:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_bundle_top_new_tinder_plus_ckpts_2026-02-26_v1.zip`
- включает:
  - `top_new_dataset` после Tinder-cleaning,
  - фиксированные folds,
  - CV5 checkpoints (`ConvNeXt-S`, `EffNetV2-S`, `DeiT3-S`),
  - one-fold probes,
  - scripts и manifest.

28. Отдельно упакованы уже обученные веса `ConvNeXt-S` (`fold0`, `fold1`) для быстрой отправки в команду:
- архив:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/convnext_small_deadline14f4_weights_f0f1_2026-02-26.zip`
- включает `best_by_val_loss.pt`, `swa_model.pt`, `summary.json`, `config.json` для `f0/f1`.

29. Реально выполненный дедлайн-прогон `ConvNeXt-S` (M2) на новом датасете:
- конфиг: `14 epochs`, `stage1=9`, `SAM`, `SWA(start=5)`, `label_smoothing=0.03`, `mixup+cutmix`, `weighted_sampler`, `no_color`
- завершены `fold0` и `fold1` (дальше остановлено по времени)
- результаты:
  - `fold0`: `acc=0.976142`, `f1=0.974582`, `val_loss=0.336910`
  - `fold1`: `acc=0.979135`, `f1=0.975845`, `val_loss=0.327294`
- среднее по `f0,f1`:
  - `acc≈0.97764`, `f1≈0.97521`, `val_loss≈0.33210`
- вывод:
  - дедлайн-конфиг подтвердил переносимость сигналов `phase1` в реальное CV-обучение.

30. Запущен дополнительный эксперимент `EffNetV2-S + class-aware final hardening` (идея "сложных изображений" / усиление аугментаций на лёгких классах в конце):
- скрипт запуска (bundle):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/run_friend_colab_effnetv2s_classaware_harden.sh`
- фича реализована в тренере:
  - `--class-aware-harden-last-epochs`
  - `--class-aware-easy-topk`
  - `--class-aware-easy-labels_csv` (опционально)
- `fold0` (MPS, `BATCH_SIZE=16`) завершён:
  - `final(selected=SWA)`: `acc=0.974619`, `f1=0.971644`, `val_loss=0.340002`
- сравнение с обычным `EffNetV2-S` на том же `fold0`:
  - `acc`: паритет (`0.974619`)
  - `f1`: чуть хуже (`-0.000171`)
  - `val_loss`: чуть хуже (`+0.001352`)
- вывод:
  - как single-model апгрейд не подтвердился на `fold0`, но как источник diversity для ансамбля идея остаётся потенциально полезной.

31. Импортированы командные результаты `DeiT3-S` из внешнего архива и встроены в общий bundle:
- импортирован архив:
  - `/Users/fedorgracev/Downloads/outputs_final3_split_deit3_small.zip`
- распаковка в:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_final3_split_deit3_small`
- проверено наличие `4` фолдов (`f0..f3`) и метафайлов оркестратора.

32. Сформирован "канонический" common-zoo (`fold0,1`, 3 модели) для честных быстрых OOF-сравнений оркестраторов:
- используется в локальных OOF-сравнениях новых идей:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_final3_common01_3models_zoo_canonical`
- состав моделей:
  - `cnn_convnext_small_sam_swa_ls03_swa5`
  - `cnn_effnetv2_s_sam_swa_ls03_swa5`
  - `vit_deit3_small_safe`

33. Прогнан дефолтный `LR(MSE)+geo8` оркестратор на partial final3-bundle (неполное покрытие фолдов/моделей):
- output:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_final3_default_ensemble_zoo`
- сабмит:
  - `submission_default_lr_geo8_equal.csv`
- Kaggle public:
  - `0.95882`
- причина просадки:
  - `ConvNeXt/EffNet` были только на `f0,f1`, а на `f2,f3` ансамбль фактически опирался на один `DeiT3`;
  - это был технический smoke-test оркестратора, а не боевой финальный сабмит.

34. Проведён анализ командного ноутбука с meta-идеями (`/Users/fedorgracev/Downloads/meta_model_classifier.ipynb`) и выделены полезные направления:
- сильные идеи:
  - `classwise ridge blend`
  - `CatBoost` на meta-features (`entropy`, `margin`, `pairwise diffs`)
  - attention-meta
- найденные проблемы (важно для воспроизводимости):
  - часть ячеек с in-sample оценкой вместо честного OOF;
  - не полностью воспроизводимый state ноутбука (`переменные из разных веток`);
  - баг в `TransformerMetaLearner`: создание слоя внутри `forward()`.
- итог:
  - идеи признаны полезными (`8/10` по практической ценности), но внедрять их нужно через отдельные скрипты с честной OOF-валидацией.

35. Реализован и прогнан быстрый OOF bakeoff идей из ноутбука (честная CV-проверка на common `fold0,1`):
- новый скрипт:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/meta_oof_bakeoff_notebook_ideas.py`
- output:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_meta_bakeoff_notebook_common01/summary.json`
- результаты:
  - `classwise_ridge_cv`: `acc=0.980940`, `f1=0.979215` (лучший локально)
  - `lr_mse_per_fold`: `acc=0.979924`, `f1=0.977715`
  - `logreg_raw_probs_cv`: `acc=0.976112`, `f1=0.972228`
- вывод:
  - `classwise_ridge` действительно лучше дефолтного `LR(MSE)` на локальном common OOF.

36. В основной meta-оркестратор интегрирован новый метод `CWR` (classwise ridge):
- патч в:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/meta_stack_cv5_attention.py`
- добавлены аргументы:
  - `--cwr-ridge-alpha`
  - `--cwr-classwise-alpha`
- локальный OOF (common `fold0,1`) после интеграции:
  - `attn`: `acc=0.980178`, `f1=0.978222`
  - `logreg`: `acc=0.976112`, `f1=0.973152`
  - `cwr` (из bakeoff, тот же data regime): `acc=0.980940`, `f1=0.979215`
- output meta-run:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_meta_stack_common01/summary.json`
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_meta_stack_common01_cwr_submit/...`

37. `CWR`-сабмит на partial common01-пуле был проверен в Kaggle:
- файл:
  - `submission_meta_cwr_geo8_oof_acc.csv`
- public score:
  - `0.96904`
- вывод:
  - `CWR` как идея рабочая, но на partial-пуле и при ограниченном покрытии моделей/фолдов проиграл финальному `mixed LR`.

38. Реализован отдельный смешанный оркестратор `old + new` (ключевой дедлайн-апгрейд):
- новый скрипт:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/mixed_old_new_orchestrator_submit.py`
- идея:
  - объединить pre-clean zoo (`old`) и post-clean zoo (`new`) как разные источники diversity;
  - выровнять OOF по `image_id` на `top_new_dataset` и текущих `aligned folds`;
  - собрать финальные test submissions через:
    - `mixed_lr_mse`
    - `mixed_cwr`
- важная оговорка:
  - pseudo-CV оптимистичен для old-zoo, т.к. старые модели обучались на другой fold-разметке (до чистки).

39. В mixed-оркестраторе по ходу дедлайна исправлены две технические ошибки:
- dataclass/import issue при динамической загрузке helper-модуля (`sys.modules[...] = mod` перед `exec_module`);
- `NameError: helper is not defined` в агрегаторе test-предсказаний по фолдам (передача `helper` в `aggregate_test_per_model`).
- после патчей прогон завершён с полным использованием кэша `test_probs_tta_geo8.npy`.

40. Итог mixed old+new оркестратора (главный результат дедлайн-спринта):
- output:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_mixed_old_new_orchestrator/summary.json`
- состав пула:
  - `old`: `ConvNeXt-S`, `EffNetV2-S`, `DeiT3-S` (full CV5, pre-clean)
  - `new`: `ConvNeXt-S`, `EffNetV2-S`, `DeiT3-S` (common `fold0,1`, post-clean)
  - всего `6` "модельных источников" в stacked feature space
- pseudo-OOF (aligned intersection `N=3755`, `folds=[0,1]`):
  - `mixed_cwr`: `acc=0.985885`, `f1=0.983327`
  - `mixed_lr_mse`: `acc=0.984820`, `f1=0.982900`
- Kaggle public:
  - `submission_mixed_lr_mse_geo8_oof_acc.csv` -> **`0.97338`** (лучший результат текущего цикла)
  - `submission_mixed_cwr_geo8_oof_acc.csv` -> `0.96972`
- ключевой вывод:
  - смешение `old + new` через `LR(MSE)` оказалось сильнее всех остальных проверенных late-stage оркестраторов.

41. Дополнительно проверен `mixed old+new` с `fold_agg=equal`:
- файл:
  - `submission_mixed_lr_mse_geo8_equal.csv`
- Kaggle public:
  - `0.97338`
- сравнение по меткам с `mixed_lr_mse_geo8_oof_acc`:
  - `0` отличий (`0/2503`), то есть это фактически тот же сабмит.

42. Проверены post-orchestrator low-risk доработки поверх `mixed LR 0.97338`:
- `conservative consensus override` (перезапись `3` строк только там, где `old_cv5` и `mixed_cwr` согласны против `mixed_lr`):
  - файл:
    - `submission_mixed_lr_consensus_override_oldcv5_cwr.csv`
  - Kaggle public:
    - `0.97323` (хуже)
- pair-experts gated override (использованы готовые `test_prob.npy` и старые tuned thresholds/margins):
  - `aggr1` (`7` изменений): `0.97338`
  - `aggr2` (`8` изменений): `0.97338`
- вывод:
  - pair-experts на этом уровне качества не дали прироста, но и не ухудшили score при умеренно агрессивном gated-применении.

43. Важный практический итог по pair-experts:
- идея подтверждена как корректная и полезная в целом (на более ранних/слабых ансамблях давала прирост),
- но на финальном сильном `mixed old+new LR` большинство сложных пар уже решаются хорошо, и потенциал доработки снижается.

44. Финальный late-stage реестр проверенных оркестраторных сабмитов (дедлайн `2026-02-26`):
- `submission_default_lr_geo8_equal.csv` -> `0.95882` (partial final3 smoke, неполный пул)
- `submission_meta_cwr_geo8_oof_acc.csv` -> `0.96904` (`CWR` на partial common01)
- `submission_mixed_cwr_geo8_oof_acc.csv` -> `0.96972` (`CWR` на mixed old+new)
- `submission_mixed_lr_mse_geo8_oof_acc.csv` -> **`0.97338`** (лучший)
- `submission_mixed_lr_mse_geo8_equal.csv` -> `0.97338` (идентичен по меткам)
- `submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr1.csv` -> `0.97338`
- `submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr2.csv` -> `0.97338`

45. Выбор двух фаворитов для финального private-хеджа (до открытия private):
- основной фаворит (максимум public на текущем цикле):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_mixed_old_new_orchestrator/submission_mixed_lr_mse_geo8_oof_acc.csv`
  - rationale: лучший public `0.97338`, сильная diversity-смесь `old+new`, `geo8`, проверенный `LR(MSE)`.
- консервативный/устойчивый хедж:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8_equal.csv`
  - rationale: зрелый full-CV5 all20 pipeline (`0.97154`), меньше late-stage "экспериментальности", полезен как хедж против переоптимизации на public.
- альтернативный "тонкий" хедж (внутри лучшего семейства):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_mixed_old_new_orchestrator/submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr1.csv`
  - rationale: same public, минимальные изменения на спорных парах (`7` строк), потенциально другой private-профиль.

## 19. Финальная Архитектура (версия для защиты)

Ниже описана итоговая архитектура проекта в том виде, в котором она реально использовалась к дедлайну.

### 19.1. Data Layer (train only)

1. Базовая очистка (`strict/aggressive`) + metadata-артефакты:
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/prepare_clean_train.py`

2. Расширенный рабочий набор команды:
- `dataset_team_corrected_v1/strict` (исторический рабочий baseline)

3. Новый `top_new_dataset` (ключевой актив проекта):
- ручная чистка + relabel через Tinder-style UI;
- финальный размер после применения действий: `9835`;
- validated `train.csv == files on disk`, no duplicates, no label/folder mismatch.

### 19.2. Model Layer

1. Базовые pretrained backbone'ы (`timm`, `pretrained=True`, fine-tuning):
- `ConvNeXt-S`
- `EffNetV2-S`
- `DeiT3-S`
- (исторически также: `ResNet50`, `ViT-Base`, peer ConvNeXt-Tiny)

2. Режимы обучения:
- `SAM`
- `SWA` (в том числе ранний старт `swa_start_epoch=5`)
- `label_smoothing`
- `mixup/cutmix`
- `weighted_sampler`
- `no_color` аугментации (важное доменное ограничение)

3. Специалисты:
- бинарные pair-experts для тяжёлых пар (`kiwi/potato`, `redapple/tomato`, `mandarin/orange`)

### 19.3. Ensemble / Orchestrator Layer

1. Классический ансамбль:
- `LR(MSE)` на вероятностях/логитах (`per-fold` и финальный fit)
- `geo8` TTA

2. Meta/advanced layer:
- `HistGradientBoosting` (исторически, сильный прирост в основной фазе проекта)
- `attention` meta-learner (локально сильный OOF на common01)
- `classwise ridge (CWR)` (идея из ноутбука, локально сильная)

3. Mixed old+new orchestrator (финальная late-stage стратегия):
- объединение pre-clean и post-clean zoo через OOF alignment по `image_id`
- финальный победитель на public: `mixed old+new + LR(MSE) + geo8`

### 19.4. Operations / Delivery Layer (важно для защиты)

1. Автоматизация и оркестрация:
- adaptive night pipeline (`phase1 -> phase4`)
- train-only split orchestrator для команды
- mixed old+new orchestrator

2. Командная поставка артефактов:
- zip-бандлы с датасетом/чекпоинтами/скриптами
- cross-platform launchers (MPS/CUDA/Colab/PyCharm)

3. Kaggle delivery:
- автосабмиты и ручные controlled A/B сабмиты
- сравнение нескольких оркестраторов и постпроцессов

## 20. Что сработало / что не сработало / что важно помнить

### 20.1. Что сработало (подтверждено)

- `no_color`-режим аугментаций (ранний A/B и последующие циклы)
- `SWA` (особенно ранний старт `swa_start_epoch=5`)
- `label_smoothing=0.03` на `ConvNeXt-S` в дедлайн-config
- `mixed old+new` blending через `LR(MSE)` (главный late-stage прирост)
- Tinder-style ручная чистка как источник более качественного `top_new_dataset`
- честные OOF bakeoff'ы для быстрой проверки meta-идей (вместо запуска всего notebook "как есть")

### 20.2. Что не сработало (или не дало прироста в финальном режиме)

- `stage4 confidence-aware finetune` (в текущей реализации) на `ConvNeXt-S/fold0`
- `mixed CWR` как финальный public-сабмит (локально выглядел лучше, на public проиграл `mixed LR`)
- pair-experts postprocess на самом сильном `mixed LR` (не ухудшил, но не улучшил `0.97338`)
- partial final3 default ensemble без полного покрытия фолдов/моделей (`0.95882`)

### 20.3. Что осталось перспективным (но не закрыто до дедлайна)

- добавить в `mixed LR` новые источники diversity:
  - `hard-aug EffNet` хотя бы `f0,f1`
  - любые дополнительные независимые модели/сабмиты
- честно проверить `CWR`/`attn` на более полном пуле (не только common `0,1`)
- полноценный `CatBoost` meta на честном OOF (локально пакет не был доступен в финальной фазе)

## 21. Отдельный highlight для защиты: Tinder-style ручная чистка (это реально сильная часть проекта)

Это один из самых сильных и "защитных" элементов проекта, который важно показать отдельно.

### 21.1. Что было сделано

- Собран локальный web-интерфейс ручной разметки в стиле Tinder (`keep / relabel / trash + undo`).
- Пройдено все `9889` изображений вручную.
- Экспортированы финальные действия с replay `decision + undo`.
- Автоматически применены релейблы и перемещение `trash` в карантин.
- После применения выполнена полная валидация структуры датасета.

### 21.2. Почему это важно инженерно

- Это не "косметика", а контролируемая правка train-распределения.
- Есть журнал действий и возможность replay (воспроизводимость).
- Есть backup CSV перед перезаписью.
- Есть separation `active train` vs `_manual_quarantine_batches` (чистая файловая структура).

### 21.3. Что показывать в презентации

- интерфейс (скриншоты/записи из `outputs_manual_review/tinder_swipe_review/...`)
- статистику решений (`keep/relabel/trash`)
- pipeline `session -> export -> apply -> validate`
- сравнение `old best` vs `new SWA` на одном `fold0` (честный A/B)

## 22. Карта артефактов для презентации и доклада (source-of-truth)

Ниже — список файлов/директорий, которые удобно использовать как основу для слайдов, доклада и защиты.

### 22.1. Основной текстовый отчёт

- Этот файл:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/PROJECT_ANALYSIS_FULL.md`

### 22.2. Датасет / ручная чистка / Tinder

- `top_new_dataset`:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/top_new_dataset`
- Tinder session:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/session_20260225_203059`
- exported final actions:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_manual_review/tinder_swipe_review/session_20260225_203059/export_final_actions/actions_for_apply_manual_actions.csv`
- применённый post-tinder compare / честный A/B:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_post_tinder_convnext_cv2_compare/...`

### 22.3. One-fold probes (adaptive phase1)

- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_adaptive_night_run1/phase1_feature_probes/runs/000_baseline`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_adaptive_night_run1/phase1_feature_probes/runs/003_swa_earlier`
- `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_top_new_adaptive_night_run1/phase1_feature_probes/runs/004_label_smoothing_low`
- (плюс остальные probes для таблицы "что пробовали / что приняли / что отклонили")

### 22.4. Командные bundle-артефакты

- dataset + checkpoints bundle:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_bundle_top_new_tinder_plus_ckpts_2026-02-26_v1.zip`
- split training bundle:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1.zip`
- convnext weights bundle (`f0,f1`):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/convnext_small_deadline14f4_weights_f0f1_2026-02-26.zip`

### 22.5. Оркестраторы и meta-идеи

- adaptive night pipeline:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/adaptive_top_new_night_pipeline.py`
- train split orchestrator:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/top_new_final3_cv5_train_orchestrator.py`
- notebook ideas bakeoff:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/meta_oof_bakeoff_notebook_ideas.py`
- mixed old+new orchestrator (финальный ключевой late-stage скрипт):
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/scripts/mixed_old_new_orchestrator_submit.py`
- notebook-наработки команды (анализировались отдельно):
  - `/Users/fedorgracev/Downloads/meta_model_classifier.ipynb`

### 22.6. Финальные/ключевые сабмиты для демонстрации

- лучший late-stage mixed result:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1/dl_lab1/outputs_mixed_old_new_orchestrator/submission_mixed_lr_mse_geo8_oof_acc.csv`
- зрелый historical hedge:
  - `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/outputs_night_model_zoo_cv5/submission_cv5_all20_lr_geo8_equal.csv`

## 23. Финальный статус отчёта (перед разбором private)

Этот отчёт доведён до состояния, пригодного как:
- основной источник для презентации;
- основной источник для письменного доклада/защиты;
- карта артефактов (скрипты, метрики, сабмиты, эксперименты, датасеты);
- база для post-mortem после открытия `private` (что сработало, что переоценил `public`, где был настоящий прирост).

Следующий обязательный шаг после раскрытия private:
- зафиксировать `private`-результаты по финальным кандидатам,
- сравнить `public vs private gap`,
- обновить этот отчёт финальным разделом "Post-Competition Analysis" (ошибки, неожиданные победители, уроки).
