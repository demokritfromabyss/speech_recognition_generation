#!/bin/bash
set -euo pipefail

mkdir -p results

LM_PATH="${1:-lm/3-gram.pruned.1e-7.arpa.gz}"

ALPHAS=(0.01 0.05 0.1 0.5 1.0 2.0 5.0)
BETAS=(0.0 0.5 1.0 1.5)

echo "=== Shallow fusion sweep on LibriSpeech test-other ==="
for A in "${ALPHAS[@]}"; do
  for B in "${BETAS[@]}"; do
    echo "beam_lm alpha=${A} beta=${B}"
    python evaluate_assignment2.py       --data_dir data/librispeech_test_other       --method beam_lm       --beam_width 10       --alpha "${A}"       --beta "${B}"       --lm_model_path "${LM_PATH}"       > "results/librispeech_beam_lm_a_${A}_b_${B}.json" 2>&1
  done
done

echo "=== LM rescoring sweep on LibriSpeech test-other ==="
for A in "${ALPHAS[@]}"; do
  for B in "${BETAS[@]}"; do
    echo "beam_lm_rescore alpha=${A} beta=${B}"
    python evaluate_assignment2.py       --data_dir data/librispeech_test_other       --method beam_lm_rescore       --beam_width 10       --alpha "${A}"       --beta "${B}"       --lm_model_path "${LM_PATH}"       > "results/librispeech_rescore_a_${A}_b_${B}.json" 2>&1
  done
done

echo "Done."
