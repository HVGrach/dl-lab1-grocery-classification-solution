# Final Presentation

Готовая презентация для защиты:
- `DL_Lab1_Grocery_Classification_Final_2026-02-27.pptx`

Сопроводительный тайминг и шпаргалка доклада:
- `SPEAKER_NOTES_10MIN_RU.md`

## Как пересобрать презентацию

```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdgcache /opt/homebrew/bin/python3.11 docs/presentation/build_presentation.py
```

Скрипт `build_presentation.py` автоматически:
- генерирует вспомогательные графики в `assets/presentation/`;
- использует основные артефакты из `artifacts/`;
- вставляет актуальные скриншоты Tinder UI из `assets/`;
- формирует итоговый `.pptx` и markdown-ноты.

## Состав

- Основных слайдов: `25`
- Backup-слайдов: `3`
- Общий объём: `28` слайдов
- Базовый хронометраж: `~9 минут` + `~1 минута` на backup/Q&A
