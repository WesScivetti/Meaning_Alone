#!/bin/bash

# ===== USER SETTINGS =====
HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"

# List of input TSV files
INPUT_FILES=(
  "data/combined_easy.tsv"
)

# Corresponding output filenames (same order as INPUT_FILES)
OUTPUT_FILES=(
  "combined_easy.tsv"
)

# List of models + whether they use causal LM mode

MODELS=(
  "EleutherAI/pythia-12b true"
)


REVISIONS=(
  "step10000"
  "step15000"
  "step20000"
  "step25000"
  "step30000"
  "step35000"
  "step40000"
  "step45000"
  "step50000"
  "step55000"
  "step60000"
  "step65000"
  "step70000"
  "step75000"
  "step80000"
  "step85000"
  "step90000"
  "step100000"
  "step105000"
  "step110000"
  "step115000"
  "step120000"
  "step125000"
  "step130000"

)

BATCH_SIZE=2
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

#      python3 predict_ewok.py \
#        --model_name "$MODEL" \
#        --batch_size $BATCH_SIZE \
#        --revision "$REV" \
#        $CAUSAL_FLAG
#
#      rm -rf "$HF_CACHE_ROOT/transformers"
#      rm -rf "$HF_CACHE_ROOT/hub"
#
#      python3 predict.py \
#        --input_tsv "$INPUT_TSV" \
#        --output_tsv "$OUTPUT_TSV" \
#        --model_name "$MODEL" \
#        --batch_size $BATCH_SIZE \
#        --topk $TOPK \
#        --mask_str "$MASK_STR" \
#        $CAUSAL_FLAG \
#        --candidate "$CANDIDATE" \
#        --revision "$REV"
#
#      rm -rf "$HF_CACHE_ROOT/transformers"
#      rm -rf "$HF_CACHE_ROOT/hub"
#
#      python3 predict_blimp.py \
#        --model_name "$MODEL" \
#        --batch_size $BATCH_SIZE \
#        --revision "$REV" \
#        $CAUSAL_FLAG
#
#      rm -rf "$HF_CACHE_ROOT/transformers"
#      rm -rf "$HF_CACHE_ROOT/hub"
#
#      python3 predict_comps.py \
#        --model_name "$MODEL" \
#        --batch_size $BATCH_SIZE \
#        --revision "$REV" \
#        $CAUSAL_FLAG


    done
    echo "Finished $REV"
#    rm -rf "$HF_CACHE_ROOT/transformers"
#    rm -rf "$HF_CACHE_ROOT/hub"
    echo "cache clear"
  done
  echo "finished $MODEL"
#  rm -rf "$HF_CACHE_ROOT/transformers"
#  rm -rf "$HF_CACHE_ROOT/hub"
  echo "cache clear"
done

echo "done"