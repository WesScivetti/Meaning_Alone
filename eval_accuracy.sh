#!/bin/bash
#set -euo pipefail

# ===== USER SETTINGS =====
ROOT_DIR="outputs"               # Where predict results live
RESULT_GLOB="*.tsv"              # TSV pattern
PYTHON_BIN="python3"

# Constructions to evaluate
CONSTRUCTIONS=(
  "letalone"
  "muchless"
  "nottomention"
  "nevermind"
)


USE_LOG_PROBS=false  # true -> add --log_probs
# ===== END SETTINGS =====

echo "Scanning ${ROOT_DIR} for result TSVs..."
echo

LOG_FLAG=()
if [ "${USE_LOG_PROBS}" = "true" ]; then
  LOG_FLAG+=(--log_probs)
fi

# Loop over all predicted TSVs
while IFS= read -r -d '' RESULT_FILE; do
  REV_DIR="$(basename "$(dirname "$RESULT_FILE")")"
  MODEL_DIR="$(basename "$(dirname "$(dirname "$RESULT_FILE")")")"
  OUT_DIR="$(dirname "$RESULT_FILE")"

  echo "=========================================="
  echo "Model:     ${MODEL_DIR}"
  echo "Revision:  ${REV_DIR}"
  echo "TSV:       ${RESULT_FILE}"
  echo "=========================================="

  # Loop over constructions
  for CXN in "${CONSTRUCTIONS[@]}"; do
    ACC_PATH="${OUT_DIR}/accuracy_${CXN}.txt"

    echo "  → Construction: ${CXN}"
    echo "    Output:       ${ACC_PATH}"

    ${PYTHON_BIN} eval.py \
      --accuracy \
      --input_tsv "$RESULT_FILE" \
      --construction "$CXN" \
      "${LOG_FLAG[@]}" \
      > "${ACC_PATH}"
  done

  echo
done < <(find "${ROOT_DIR}" -type f -name "${RESULT_GLOB}" -print0)

echo "All evaluations completed."
