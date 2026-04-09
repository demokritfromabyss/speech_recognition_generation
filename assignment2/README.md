# Assignment 2 — ASR Decoding

Готовый комплект для выполнения второго задания по ASR decoding.

## Что внутри

- `wav2vec2decoder.py` — реализация:
  - greedy decoding
  - beam search decoding
  - beam search + KenLM shallow fusion
  - beam search + second-pass LM rescoring
- `evaluate_assignment2.py` — оценка CER/WER на датасете
- `run_sweeps.py` — свипы по `beam_width`, `temperature`, `alpha/beta`
- `report_template.md` — шаблон отчёта
- `requirements.txt`

## Важно по Python

По заданию нужен Python **>= 3.10 и < 3.14**, потому что `kenlm` несовместим с Python 3.14+.

## Установка

### macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
brew install cmake boost
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install -r requirements.txt
```

### Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
sudo apt-get update && sudo apt-get install -y cmake
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install -r requirements.txt
```

## Как быстро проверить декодер

Скопируйте из оригинального задания папки:
- `examples/`
- `lm/`

И запустите:

```bash
python wav2vec2decoder.py
```

## Как оценить один метод

```bash
python evaluate_assignment2.py \
  --data_dir data/librispeech_test_other \
  --method greedy
```

Пример для shallow fusion:

```bash
python evaluate_assignment2.py \
  --data_dir data/librispeech_test_other \
  --method beam_lm \
  --beam_width 10 \
  --alpha 0.1 \
  --beta 0.5 \
  --lm_model_path lm/3-gram.pruned.1e-7.arpa.gz
```

## Как прогнать свипы

```bash
python run_sweeps.py \
  --librispeech_dir data/librispeech_test_other \
  --earnings_dir data/earnings22_test \
  --lm_model_path lm/3-gram.pruned.1e-7.arpa.gz
```

Результаты сохранятся в папку `results/`.

## Что ещё нужно скачать из оригинального задания

Из репозитория задания положите рядом:

```text
assignment2/
  data/
  examples/
  lm/
  wav2vec2decoder.py
  evaluate_assignment2.py
  run_sweeps.py
  report_template.md
  requirements.txt
```

## Что сдавать

- код
- `report.pdf`
- графики и таблицы результатов
- при желании — CSV/JSON со свипами
