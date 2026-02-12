#!/bin/bash

# ===== USER SETTINGS =====
HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"

# List of input TSV files
INPUT_FILES=(
  "data/syntactic_tests_aligned.tsv"
)

# Corresponding output filenames (same order as INPUT_FILES)
OUTPUT_FILES=(
  "syntactic_tests_aligned.tsv"
)

# List of models + whether they use causal LM mode
#MODELS=(
#  "jhu-clsp/ettin-encoder-17m false"
#  "jhu-clsp/ettin-encoder-150m false"
#  "jhu-clsp/ettin-encoder-400m false"
#  "jhu-clsp/ettin-encoder-1b false"
#)

MODELS=(
  "jhu-clsp/ettin-encoder-400m false"
  "jhu-clsp/ettin-encoder-1b false"
  "jhu-clsp/ettin-decoder-400m true"
  "jhu-clsp/ettin-decoder-1b true"
)




REVISIONS=(
  "step2999"
  "step8994"
  "step20986"
  "step29979"
  "step59953"
  "step89929"
  "step119903"
  "step149879"
  "step14991"
  "step35974"
  "step41969"
  "step47963"
  "step53957"
  "step65948"
  "step71942"
  "step77938"
  "step83933"
  "step95924"
  "step101918"
  "step107913"
  "step113909"
  "step125897"
  "step131892"
  "step137887"
  "step143882"
)

REVISIONS2=(
  "step5996"
  "step11992"
  "step17988"
  "step23984"
  "step29979"
  "step35974"
  "step41969"
  "step47963"
  "step53957"
  "step59953"
  "step65948"
  "step71942"
  "step77938"
  "step83933"
  "step89929"
  "step95924"
  "step101918"
  "step107913"
  "step113909"
  "step119903"
  "step125897"
  "step131892"
  "step137887"
  "step143882"
  "step149879"
)

BATCH_SIZE=32
REVISION=""

# ===== END SETTINGS =====


#first loop through all models
for ENTRY in "${MODELS[@]}"; do
  MODEL=$(echo $ENTRY | awk '{print $1}')
  CAUSAL=$(echo $ENTRY | awk '{print $2}')


  echo "------------------------------------------"
  echo "running model: $MODEL"

  echo "------------------------------------------"

  # Add causal LM flag if needed
  if [ "$CAUSAL" = "true" ]; then
    CAUSAL_FLAG="--causallm"
  else
    CAUSAL_FLAG=""
  fi

  for REV in "${REVISIONS[@]}"; do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  Revision: $REV"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    # Loop through input/output pairs
    for i in "${!INPUT_FILES[@]}"; do
      INPUT_TSV="${INPUT_FILES[$i]}"
      OUTPUT_TSV="${OUTPUT_FILES[$i]}"

      echo "=========================================="
      echo "Processing dataset: $INPUT_TSV"
      echo "=========================================="



      python3 predict_syntax.py \
        --input_tsv "$INPUT_TSV" \
        --output_tsv "$OUTPUT_TSV" \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        $CAUSAL_FLAG \
        --revision "$REV"
    done


    echo "Finished $REV"
    rm -rf "$HF_CACHE_ROOT/hub/models--${MODEL//\//--}"
    echo "cache clear"

  done
  echo "finished $MODEL"
  rm -rf "$HF_CACHE_ROOT/hub/models--${MODEL//\//--}"
  echo "cache clear"
done

echo "done"