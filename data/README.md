# Data Placement (Placeholder)

Исходный датасет соревнования и рабочие копии датасета не хранятся в git.

## Что ожидает код

Для большинства скриптов ожидается наличие папки:

```text
dl_lab1/top_new_dataset/
```

Типичная структура внутри:

```text
dl_lab1/top_new_dataset/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   └── train/
│       ├── <класс_1>/
│       ├── <класс_2>/
│       └── ...
└── test_images/
    └── test_images/
```

## Важно

- Любая чистка / relabeling применяется только к `train`.
- `test` не модифицируется.

## Откуда брать данные

- Исходный архив соревнования: вручную из Kaggle/курса
- Подготовленный `top_new_dataset` (после Tinder-cleaning): из внешнего bundle (см. `weights/README.md`)

