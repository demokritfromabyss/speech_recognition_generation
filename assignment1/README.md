# Assignment 1 — Keyword spotting on Speech Commands

Готовый каркас для задания:
- реализация `LogMelFilterBanks`;
- классификация `yes/no` на `SPEECHCOMMANDS`;
- эксперименты по `n_mels` и `groups`;
- автоматическое сохранение графиков и таблицы результатов.

## Структура

```text
assignment1_full/
├── melbanks.py
├── check_melbanks.py
├── dataset.py
├── model.py
├── train.py
├── requirements.txt
├── report_template.md
├── README.md
└── figures/
```

## Установка

```bash
pip install -r requirements.txt
```

## Проверка слоя LogMelFilterBanks

```bash
python check_melbanks.py
```

Ожидаемо:
- одинаковые формы;
- маленькая `max abs diff`;
- `allclose(atol=1e-5) == True`.

## Запуск всех экспериментов

```bash
python train.py
```

Что произойдёт:
- автоматически скачается `SPEECHCOMMANDS`;
- запустятся эксперименты по `n_mels = [20, 40, 80]`;
- затем по `groups = [1, 2, 4, 8, 16]` при `n_mels = 80`;
- сохранятся JSON-файлы с историей обучения;
- построятся графики в `figures/`;
- сохранится сводная таблица `results_summary.csv`.

## Какие файлы приложить к сдаче

1. Публичный GitHub-репозиторий с кодом.
2. PDF-отчёт на основе `report_template.md`.

## Что нужно вписать в отчёт

После запуска заполните:
- итоговые accuracy;
- среднее время эпохи;
- количество параметров;
- FLOPs;
- краткие выводы по влиянию `n_mels` и `groups`.

## Замечания

- `groups` должно делить `n_mels` без остатка.
- Вход в `Conv1d` имеет форму `(B, n_mels, T)`.
- Используется только бинарная классификация: `yes` и `no`.
