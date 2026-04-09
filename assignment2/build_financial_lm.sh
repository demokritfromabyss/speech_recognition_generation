#!/bin/bash
set -euo pipefail

CORPUS_PATH="${1:-data/earnings22_train/corpus.txt}"
OUT_DIR="${2:-lm/financial}"
ORDER="${3:-3}"

mkdir -p "${OUT_DIR}"

if ! command -v lmplz >/dev/null 2>&1; then
  echo "Error: lmplz not found in PATH."
  echo "Install KenLM binaries first, for example:"
  echo "  brew install kenlm"
  echo "or build KenLM from source and add bin/ to PATH."
  exit 1
fi

if ! command -v build_binary >/dev/null 2>&1; then
  echo "Error: build_binary not found in PATH."
  echo "Install KenLM binaries first."
  exit 1
fi

ARPA_PATH="${OUT_DIR}/earnings_${ORDER}gram.arpa"
BIN_PATH="${OUT_DIR}/earnings_${ORDER}gram.binary"

echo "Building ${ORDER}-gram ARPA from ${CORPUS_PATH}"
lmplz -o "${ORDER}" < "${CORPUS_PATH}" > "${ARPA_PATH}"

echo "Building binary LM"
build_binary "${ARPA_PATH}" "${BIN_PATH}"

echo "Done."
echo "ARPA:   ${ARPA_PATH}"
echo "Binary: ${BIN_PATH}"
