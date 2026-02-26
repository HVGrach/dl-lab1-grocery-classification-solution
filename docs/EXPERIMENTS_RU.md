# Эксперименты, решения и выводы (RU)

Этот файл подготовлен для:
- презентации/защиты;
- README репозитория;
- фиксации того, что пробовали, что улучшило, что ухудшило, и какие шаги следующие.

Все числа и формулировки сведены из `docs/archive/PROJECT_ANALYSIS_FULL_SOURCE.md` (фикс состояния на `2026-02-26`).

## 1. Краткая хронология (важные этапы)

### Этап A. Базовый CV + ансамбль + pair-experts + meta (ранняя фаза)

Базовые single-model OOF:

| Модель | OOF ACC | OOF F1-macro |
|---|---:|---:|
| `convnext_small` | `0.946886` | `0.943792` |
| `effnetv2_s` | `0.946274` | `0.943383` |
| `resnet50` | `0.925171` | `0.921223` |

Эволюция ансамбля (историческая фаза):

| Этап | ACC | F1-macro | ΔACC |
|---|---:|---:|---:|
| Базовый weighted ensemble | `0.956876` | `0.953979` | - |
| После weight search | `0.957080` | `0.954287` | `+0.000204` |
| После class-bias | `0.957896` | `0.956048` | `+0.000816` |
| Logistic specialists rerank | `0.958304` | `0.956406` | `+0.000408` |
| Trained pair-experts rerank | `0.958609` | `0.956774` | `+0.000305` |
| Meta HGB stack | `0.973086` | `0.968951` | `+0.014477` |

Вывод:
- простые постпроцессы (weight search, bias) дают дешёвый стабильный прирост;
- pair-experts дают прирост, но ограниченный (только на uncertain-кейсах);
- meta-stacking дал самый большой OOF-прирост в этой фазе.

## 2. Ключевой A/B: color-aug vs no_color (сильный подтверждённый эффект)

Постановка:
- одна модель: `convnext_small`
- один и тот же split (`fold=0`)
- одинаковые гиперпараметры
- менялся только профиль аугментаций

Результаты:

| Профиль | ACC | F1-macro | best val_loss |
|---|---:|---:|---:|
| `full` (с color transforms) | `0.94954` | `0.94674` | `0.57939` |
| `no_color` | `0.95872` | `0.95405` | `0.54543` |

Дельта `no_color - full`:
- `+0.00917 acc`
- `+0.00731 f1_macro`
- `-0.03396 val_loss` (лучше)

Вывод:
- цветовые сдвиги вредят задаче (ломают цветовую семантику близких классов);
- в baseline оставлен только `no_color`.

## 3. Ночной автопилот и выбор модельного зоопарка

One-fold автопрогон (`overnight_10h`, `fold=0`) показал:

Single-модели:
- `cnn_effnetv2_s_sam_swa`: `acc=0.97211`, `f1_macro=0.96958`
- `cnn_convnext_small_sam_swa`: `acc=0.96947`, `f1_macro=0.96780`
- `vit_deit3_small_color_safe`: `acc=0.96053`, `f1_macro=0.95544`
- `vit_base_augreg_lite`: `acc=0.91789`, `f1_macro=0.90801` (слабый single)

Лучший one-fold ансамбль:
- состав: `EffNetV2-S + ConvNeXt-S + DeiT3-S`
- `acc=0.97526`, `f1_macro=0.97418`

Вывод:
- `ConvNeXt-S` и `EffNetV2-S` — основа;
- `DeiT3-S` полезен как diversity;
- `ViT-Base` чаще аутсайдер.

## 4. Большой CV5-прогон (`20` запусков = `4 модели x 5 фолдов`)

Средние CV-метрики (5 фолдов):

| Модель | ACC mean | F1 mean | Комментарий |
|---|---:|---:|---|
| `ConvNeXt-S` | `0.97209` | `0.96885` | сильный и стабильный |
| `EffNetV2-S` | `0.97178` | `0.96783` | сильный и стабильный |
| `DeiT3-S` | `0.96546` | `0.96165` | слабее single, полезен как diversity |
| `ViT-Base` | `0.92144` | `0.91238` | слабый single |

Итоговый full-CV5 ансамбль:
- внутри фолда: `LR(MSE)` по `val_probs`
- test inference: `geo8` TTA
- агрегация фолдов: равные веса (`1/5`)

Public LB:
- `submission_cv5_all20_lr_geo8_equal.csv` -> **`0.97154`**

Доп. наблюдение:
- `ViT-Base` получил вес `0.0` во всех 5 фолдах (LR сам отключил модель).

## 5. Tinder-style ручная чистка датасета (ключевая инженерная часть)

Что сделано:
- локальный UI для ручного review в стиле Tinder (`keep/relabel/trash + undo`);
- просмотрено вручную `9889 / 9889` изображений;
- экспортирован replay-consistent журнал действий;
- автоматически применены relabel/trash к `top_new_dataset`;
- проведена пост-валидация целостности.

Итоги сессии:
- `keep = 9696`
- `relabel = 139`
- `trash = 54`
- итого файловых операций: `193`

После применения:
- `top_new_dataset/train.csv` = `9835` строк
- активных изображений на диске = `9835`
- `train.csv == файлы на диске` (1:1)
- mismatches `label vs folder` = `0`
- duplicate `image_id` = `0`

Почему это важно:
- это не просто "ручная чистка", а воспроизводимый data-engineering pipeline;
- есть журнал действий и replay;
- есть backup CSV и quarantine для `trash`.

## 6. Post-Tinder A/B (честная проверка на одном и том же `top_new_dataset/fold0`)

Важно:
- прямое сравнение старых/новых `summary.json` нечестно (разные датасеты/val split size);
- поэтому делался отдельный replay на **одном и том же** `top_new_dataset/fold0`.

Корректный итоговый A/B (`old best` vs `new SWA`):
- `acc`: паритет (`0.978680 -> 0.978680`)
- `errors`: паритет (`42 -> 42`)
- `f1_macro`: `+0.000459` у нового
- `mean_conf`: небольшой рост у нового
- `1936` предсказаний unchanged, `17` improved, `17` worsened

Вывод:
- эффект новой очистки на этом probe проявился как перераспределение ошибок и прирост `macro-F1`, а не как падение числа ошибок.

## 7. Adaptive phase1 probes (фича-отбор для дедлайн-конфига)

One-fold probes на `ConvNeXt-S` (`top_new_dataset`, `fold=0`) дали следующие сигналы:

| Probe | ACC | F1 | Вывод |
|---|---:|---:|---|
| `000_baseline` | `0.97462` | `0.97126` | базовая точка |
| `003_swa_earlier` | `0.97614` | `0.97478` | **лучше**, ранний SWA полезен |
| `004_label_smoothing_low (0.03)` | `0.97766` | `0.97363` | **лучший стабильный сигнал** |
| `008_cutmix_only` | `0.97665` | `0.97282` | сильный, но не лидер |
| `009_sam_off` | `0.97513` | `0.97329` | хуже -> `SAM` оставляем |
| `010_swa_off` | `0.97310` | `0.97147` | хуже -> `SWA` оставляем |

Принятые решения в дедлайн-конфиг:
- `SAM = on`
- `SWA = on`, `swa_start_epoch=5`
- `label_smoothing = 0.03`
- `weighted_sampler = on`
- `mixup + cutmix`
- `no_color`

## 8. Что ухудшило / не улучшило (важно для презентации)

### Подтверждённо ухудшало

1. `color-aug` (Hue/Saturation/ColorJitter и т.п.) в этой задаче.
- Подтверждено controlled A/B.
- Сильная деградация по `acc/f1`.

2. Partial final3 default ensemble на неполном пуле моделей/фолдов.
- `submission_default_lr_geo8_equal.csv` -> `0.95882`
- Причина: на части фолдов ансамбль фактически опирался на один `DeiT3`.

### Не подтвердило прирост в финальном режиме

1. `stage4 confidence-aware finetune` (текущая реализация, `ConvNeXt-S/fold0`).
- `best_stage4_epoch = 0`
- stage4-эпохи дали худший `val_loss`, чем база.

2. `mixed CWR` как финальный public-сабмит.
- Локально на common OOF идея сильная.
- На public: `0.96972` < `mixed LR 0.97338`.

3. Pair-experts поверх уже лучшего `mixed LR`.
- `aggr1`: `0.97338`
- `aggr2`: `0.97338`
- То есть полезны как ранний буст, но не добавили прирост на самом сильном late-stage ансамбле.

4. `EffNetV2-S + class-aware final hardening` (probe `fold0`).
- как single-model апгрейд не подтвердился (`acc` паритет, `f1/loss` чуть хуже);
- идея остаётся возможным источником diversity для ансамбля.

## 9. Late-stage orchestrators: финальный сравнительный срез

Проверенные сабмиты (дедлайн `2026-02-26`):

| Submission | Public | Комментарий |
|---|---:|---|
| `submission_default_lr_geo8_equal.csv` | `0.95882` | partial final3 smoke |
| `submission_meta_cwr_geo8_oof_acc.csv` | `0.96904` | `CWR` на partial common01 |
| `submission_mixed_cwr_geo8_oof_acc.csv` | `0.96972` | `CWR` на mixed old+new |
| `submission_mixed_lr_mse_geo8_oof_acc.csv` | **`0.97338`** | лучший |
| `submission_mixed_lr_mse_geo8_equal.csv` | `0.97338` | идентичен по меткам |
| `submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr1.csv` | `0.97338` | pair-experts overlay |
| `submission_mixed_lr_mse_geo8_oof_acc_pairexp_aggr2.csv` | `0.97338` | pair-experts overlay |

Главный вывод:
- late-stage смешение `old + new` zoo через `LR(MSE)` оказалось сильнее всех остальных проверенных вариантов.

## 10. Что сработало (итоговая выжимка для слайда)

Подтверждено в проекте:
- `no_color`-режим аугментаций
- `SWA` (особенно ранний старт)
- `label_smoothing=0.03` в дедлайн-конфиге `ConvNeXt-S`
- `mixed old+new` blending через `LR(MSE)`
- Tinder-style ручная чистка и relabeling train (`top_new_dataset`)
- честные OOF bakeoff-проверки meta-идей (вместо in-sample/ноутбучной иллюзии)

## 11. Идеи, как улучшать дальше (после дедлайна)

Приоритетные направления:

1. Добавить новые источники diversity в `mixed LR`
- например, `hard-aug EffNet` хотя бы `f0,f1`
- дополнительные независимые модели/сабмиты

2. Честно проверить `CWR` / `attn` на более полном пуле
- не только common `fold0,1`, а на большем покрытии моделей/фолдов

3. Полноценный `CatBoost` meta на честном OOF
- идея была перспективной, но не закрыта в дедлайн из-за ограничений окружения

4. Group-aware отбор (по `plu`) при выборе финального сабмита
- не только public + mean-CV
- учитывать `worst-group` / group-holdout quantiles

5. High-resolution stage-2 fine-tune (320/384)
- особенно для fine-grained SKU-like классов

6. Refit pair-experts в новом data regime + калибровка порогов
- полезно именно под group-shift риск, а не только под public.
