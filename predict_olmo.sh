#!/bin/bash

# ===== USER SETTINGS =====

# List of input TSV files
INPUT_FILES=(
  "data/combined_easy.tsv"
)

# Corresponding output filenames (same order as INPUT_FILES)
OUTPUT_FILES=(
  "combined_easy.tsv"
)

# List of models + whether they use causal LM mode
#MODELS=(
#  "jhu-clsp/ettin-encoder-17m false"
#  "jhu-clsp/ettin-encoder-150m false"
#  "jhu-clsp/ettin-encoder-400m false"
#  "jhu-clsp/ettin-encoder-1b false"
#)

MODELS=(
  "allenai/OLMo-2-1124-7B true"
)

REVISIONS=(
  "step1000-tokens5B"
)

BATCH_SIZE=512
TOPK=2
MASK_STR="[MASK]"
CANDIDATE=""
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



      python3 predict.py \
        --input_tsv "$INPUT_TSV" \
        --output_tsv "$OUTPUT_TSV" \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --topk $TOPK \
        --mask_str "$MASK_STR" \
        $CAUSAL_FLAG \
        --candidate "$CANDIDATE" \
        --revision "$REV"

      python3 predict_blimp.py \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --revision "$REV" \
        $CAUSAL_FLAG

      python3 predict_comps.py \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --revision "$REV" \
        $CAUSAL_FLAG

      python3 predict_ewok.py \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --revision "$REV" \
        $CAUSAL_FLAG


    done
    echo "Finished $REV"
    rm -rf ~/.cache/huggingface/transformers
    rm -rf ~/.cache/huggingface/hub
    echo "cache clear"
  done
  echo "finished $MODEL"
  rm -rf ~/.cache/huggingface/transformers
  rm -rf ~/.cache/huggingface/hub
  echo "cache clear"
done

echo "done"