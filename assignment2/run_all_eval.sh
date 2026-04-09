#!/bin/bash
set -e

LM_PATH="lm/3-gram.pruned.1e-7.arpa.gz"

echo "Running LibriSpeech: greedy"
python evaluate_assignment2.py \
  --data_dir data/librispeech_test_other \
  --method greedy \
  > librispeech_greedy.json 2>&1

echo "Running LibriSpeech: beam"
python evaluate_assignment2.py \
  --data_dir data/librispeech_test_other \
  --method beam \
  > librispeech_beam.json 2>&1

echo "Running LibriSpeech: beam_lm"
python evaluate_assignment2.py \
  --data_dir data/librispeech_test_other \
  --method beam_lm \
  --lm_model_path "$LM_PATH" \
  > librispeech_beam_lm.json 2>&1

echo "Running LibriSpeech: beam_lm_rescore"
python evaluate_assignment2.py \
  --data_dir data/librispeech_test_other \
  --method beam_lm_rescore \
  --lm_model_path "$LM_PATH" \
  > librispeech_rescore.json 2>&1

echo "Running Earnings22: greedy"
python evaluate_assignment2.py \
  --data_dir data/earnings22_test \
  --method greedy \
  > earnings_greedy.json 2>&1

echo "Running Earnings22: beam"
python evaluate_assignment2.py \
  --data_dir data/earnings22_test \
  --method beam \
  > earnings_beam.json 2>&1

echo "Running Earnings22: beam_lm"
python evaluate_assignment2.py \
  --data_dir data/earnings22_test \
  --method beam_lm \
  --lm_model_path "$LM_PATH" \
  > earnings_beam_lm.json 2>&1

echo "Running Earnings22: beam_lm_rescore"
python evaluate_assignment2.py \
  --data_dir data/earnings22_test \
  --method beam_lm_rescore \
  --lm_model_path "$LM_PATH" \
  > earnings_rescore.json 2>&1

echo "Done. Results saved to:"
echo "  librispeech_greedy.json"
echo "  librispeech_beam.json"
echo "  librispeech_beam_lm.json"
echo "  librispeech_rescore.json"
echo "  earnings_greedy.json"
echo "  earnings_beam.json"
echo "  earnings_beam_lm.json"
echo "  earnings_rescore.json"