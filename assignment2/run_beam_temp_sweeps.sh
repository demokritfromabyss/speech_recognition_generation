#!/bin/bash
set -euo pipefail

mkdir -p results

echo "=== Beam width sweep on LibriSpeech test-other ==="
for BW in 1 3 10 50; do
  echo "beam_width=${BW}"
  python evaluate_assignment2.py     --data_dir data/librispeech_test_other     --method beam     --beam_width "${BW}"     > "results/librispeech_beam_bw_${BW}.json" 2>&1
done

echo "=== Temperature sweep (greedy) on LibriSpeech test-other ==="
for T in 0.5 0.8 1.0 1.2 1.5 2.0; do
  echo "temperature=${T}"
  python evaluate_assignment2.py     --data_dir data/librispeech_test_other     --method greedy     --temperature "${T}"     > "results/librispeech_greedy_temp_${T}.json" 2>&1
done

echo "Done."
