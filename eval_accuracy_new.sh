#!/usr/bin/env bash
#set -euo pipefail

# ===== USER SETTINGS =====
ROOT_DIR="outputs"               # Root where model outputs live
RESULT_GLOB="*.tsv"              # Pattern for model prediction TSV(s) within each checkpoint dir
PYTHON_BIN="python3"

# Models to evaluate (each should correspond to a subdirectory under $ROOT_DIR)
MODELS=(
  "jhu-clsp/ettin-encoder-150m"
  "jhu-clsp/ettin-encoder-400m"
)

# Constructions to evaluate (must match what eval.py expects)
CONSTRUCTIONS=(letalone muchless nottomention nevermind)

USE_LOG_PROBS=false  # true -> add --log_probs
# ===== END SETTINGS =====

LOG_FLAG=()
if [ "${USE_LOG_PROBS}" = "true" ]; then
  LOG_FLAG+=(--log_probs)
fi

# --- Helpers ---

# normalize one numeric token to [0,1] (handles %, >1 interpreted as percent, and "nan")
normalize_token () {
  local token="$1"
  if [[ -z "$token" ]]; then
    echo "NaN"; return
  fi
  # handle NaN-like strings
  if [[ "$token" =~ ^[Nn][Aa][Nn]$ ]]; then
    echo "NaN"; return
  fi
  if [[ "$token" == *% ]]; then
    local n="${token%%%}"
    awk -v v="$n" 'BEGIN{ if (v+0==0 && v!="0") print "NaN"; else printf "%.6f", v/100.0 }'
  else
    awk -v v="$token" 'BEGIN{
      if (v+0==0 && v!="0") { print "NaN"; exit }
      if (v>1) printf "%.6f", v/100.0; else printf "%.6f", v
    }'
  fi
}

# Parse all three accuracies (All, Aligned, Misaligned) from eval.py output
# Returns a space-separated line: "<all> <aligned> <misaligned>"
parse_three_accuracies () {
  local out="$1"
  # Grab the token after '=' from lines that start with "Correct"
  # e.g., "Correct 12 / 16 = 0.75"
  local rhs
  # Use awk to extract RHS after '=' and trim
  mapfile -t rhs < <(printf "%s\n" "$out" \
    | awk -F'=' '/^Correct/{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')
  # Expect 3 lines: All, Aligned, Misaligned (in that order)
  local a_all="${rhs[0]:-NaN}"
  local a_aln="${rhs[1]:-NaN}"
  local a_mis="${rhs[2]:-NaN}"

  a_all="$(normalize_token "$a_all")"
  a_aln="$(normalize_token "$a_aln")"
  a_mis="$(normalize_token "$a_mis")"

  echo "${a_all} ${a_aln} ${a_mis}"
}

# Run eval.py ONCE and return three accuracies (All, Aligned, Misaligned) for a construction
run_eval_three () {
  local input_tsv="$1"
  local cxn="$2"
  local out
  out="$(${PYTHON_BIN} eval.py --accuracy --input_tsv "$input_tsv" --construction "$cxn" "${LOG_FLAG[@]}")"
  parse_three_accuracies "$out"
}

echo
echo "=== Multi-checkpoint accuracy summarizer (with alignment) ==="
echo "Root: ${ROOT_DIR}"
echo

# Iterate explicit model list
for MODEL in "${MODELS[@]}"; do
  MODEL_DIR="${ROOT_DIR}/${MODEL}"

  if [ ! -d "$MODEL_DIR" ]; then
    echo "[WARN] Model directory not found: ${MODEL_DIR} (skipping)"
    echo
    continue
  fi

  SUMMARY_PATH="${MODEL_DIR}/accuracy_summary.tsv"
  # Overwrite summary each run for clarity.
  echo -e "model_name\tcheckpoint\talignment\tletalone_acc\tmuchless_acc\tnottomention_acc\tnevermind_acc" > "$SUMMARY_PATH"

  echo "=========================================="
  echo "Model: ${MODEL}"
  echo "Summary file: ${SUMMARY_PATH}"
  echo "=========================================="

  # Collect checkpoint dirs (one level below model dir), sorted naturally
  mapfile -t CKPT_DIRS < <(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V)

  if [ "${#CKPT_DIRS[@]}" -eq 0 ]; then
    echo "[WARN] No checkpoints found under ${MODEL_DIR}"
    echo
    continue
  fi

  for CKPT_NAME in "${CKPT_DIRS[@]}"; do
    CKPT_PATH="${MODEL_DIR}/${CKPT_NAME}"

    shopt -s nullglob
    TSVS=( "${CKPT_PATH}"/${RESULT_GLOB} )
    shopt -u nullglob

    if [ "${#TSVS[@]}" -eq 0 ]; then
      echo "  [SKIP] ${CKPT_NAME}: no TSVs matching ${RESULT_GLOB}"
      continue
    fi
    if [ "${#TSVS[@]}" -gt 1 ]; then
      echo "  [INFO] ${CKPT_NAME}: multiple TSVs found; using first by lexicographic order:"
      printf '         - %s\n' "${TSVS[@]}"
    fi

    INPUT_TSV="${TSVS[0]}"
    echo "  → Checkpoint: ${CKPT_NAME}"
    echo "    TSV:        ${INPUT_TSV}"

    # For each construction, collect three accuracies (All, Aligned, Misaligned)
    # We'll store them in arrays aligned by index: 0=All, 1=Aligned, 2=Misaligned
    declare -A ACC_ALL ACC_ALN ACC_MIS

    for CXN in "${CONSTRUCTIONS[@]}"; do
      echo -n "      - ${CXN} ... "
      read -r ACC_A ACC_B ACC_C < <(run_eval_three "$INPUT_TSV" "$CXN")
      echo "All=${ACC_A}, Aligned=${ACC_B}, Misaligned=${ACC_C}"
      ACC_ALL["$CXN"]="$ACC_A"
      ACC_ALN["$CXN"]="$ACC_B"
      ACC_MIS["$CXN"]="$ACC_C"
    done

    # Emit three rows: one per alignment
    echo -e "${MODEL}\t${CKPT_NAME}\tAll\t${ACC_ALL[letalone]}\t${ACC_ALL[muchless]}\t${ACC_ALL[nottomention]}\t${ACC_ALL[nevermind]}" >> "$SUMMARY_PATH"
    echo -e "${MODEL}\t${CKPT_NAME}\tAligned\t${ACC_ALN[letalone]}\t${ACC_ALN[muchless]}\t${ACC_ALN[nottomention]}\t${ACC_ALN[nevermind]}" >> "$SUMMARY_PATH"
    echo -e "${MODEL}\t${CKPT_NAME}\tMisaligned\t${ACC_MIS[letalone]}\t${ACC_MIS[muchless]}\t${ACC_MIS[nottomention]}\t${ACC_MIS[nevermind]}" >> "$SUMMARY_PATH"
    echo
  done

  echo "Done: ${SUMMARY_PATH}"
  echo
done

echo "All evaluations completed."
