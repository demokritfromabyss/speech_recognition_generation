#!/bin/bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <lm_path> <best_beamlm_alpha> <best_beamlm_beta> <best_rescore_alpha> <best_rescore_beta>"
  exit 1
fi

LM_PATH="$1"
BEST_BEAMLM_ALPHA="$2"
BEST_BEAMLM_BETA="$3"
BEST_RESCORE_ALPHA="$4"
BEST_RESCORE_BETA="$5"

mkdir -p results

echo "=== Best configs on Earnings22 test ==="

python evaluate_assignment2.py   --data_dir data/earnings22_test   --method beam_lm   --beam_width 10   --alpha "${BEST_BEAMLM_ALPHA}"   --beta "${BEST_BEAMLM_BETA}"   --lm_model_path "${LM_PATH}"   > "results/earnings_best_beam_lm.json" 2>&1

python evaluate_assignment2.py   --data_dir data/earnings22_test   --method beam_lm_rescore   --beam_width 10   --alpha "${BEST_RESCORE_ALPHA}"   --beta "${BEST_RESCORE_BETA}"   --lm_model_path "${LM_PATH}"   > "results/earnings_best_rescore.json" 2>&1

echo "=== Temperature sweep on Earnings22: greedy ==="
for T in 0.5 0.8 1.0 1.2 1.5 2.0; do
  echo "greedy temperature=${T}"
  python evaluate_assignment2.py     --data_dir data/earnings22_test     --method greedy     --temperature "${T}"     > "results/earnings_greedy_temp_${T}.json" 2>&1
done

echo "=== Temperature sweep on Earnings22: best shallow fusion ==="
for T in 0.5 0.8 1.0 1.2 1.5 2.0; do
  echo "beam_lm temperature=${T}"
  python evaluate_assignment2.py     --data_dir data/earnings22_test     --method beam_lm     --beam_width 10     --alpha "${BEST_BEAMLM_ALPHA}"     --beta "${BEST_BEAMLM_BETA}"     --temperature "${T}"     --lm_model_path "${LM_PATH}"     > "results/earnings_beam_lm_temp_${T}.json" 2>&1
done

echo "Done."
