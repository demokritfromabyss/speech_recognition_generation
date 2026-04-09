#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <general_lm_path> <financial_lm_path> <alpha> <beta>"
  exit 1
fi

GENERAL_LM="$1"
FIN_LM="$2"
ALPHA="$3"
BETA="$4"

mkdir -p results

echo "=== Comparing general LM vs financial LM on LibriSpeech ==="
python evaluate_assignment2.py   --data_dir data/librispeech_test_other   --method beam_lm   --beam_width 10   --alpha "${ALPHA}"   --beta "${BETA}"   --lm_model_path "${GENERAL_LM}"   > "results/librispeech_general_lm.json" 2>&1

python evaluate_assignment2.py   --data_dir data/librispeech_test_other   --method beam_lm   --beam_width 10   --alpha "${ALPHA}"   --beta "${BETA}"   --lm_model_path "${FIN_LM}"   > "results/librispeech_financial_lm.json" 2>&1

echo "=== Comparing general LM vs financial LM on Earnings22 ==="
python evaluate_assignment2.py   --data_dir data/earnings22_test   --method beam_lm   --beam_width 10   --alpha "${ALPHA}"   --beta "${BETA}"   --lm_model_path "${GENERAL_LM}"   > "results/earnings_general_lm.json" 2>&1

python evaluate_assignment2.py   --data_dir data/earnings22_test   --method beam_lm   --beam_width 10   --alpha "${ALPHA}"   --beta "${BETA}"   --lm_model_path "${FIN_LM}"   > "results/earnings_financial_lm.json" 2>&1

echo "Done."
