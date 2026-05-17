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
  "jhu-clsp/ettin-encoder-150m false"
  "jhu-clsp/ettin-encoder-400m false"
  "jhu-clsp/ettin-encoder-1b false"
  "jhu-clsp/ettin-decoder-150m true"
  "jhu-clsp/ettin-decoder-400m true"
  "jhu-clsp/ettin-decoder-1b true"
)
#
#  evisions = [
#    "step2999",
#    "step8994",
#    "step20986",
#    "step26982",
#    "step29979",
#    "step32976",
#    "step35974",
#    "step38972",
#    "step41969",
#    "step44967",
#    "step47963",
#    "step50960",
#    "step53957",
#    "step56955",
#    "step59953",
#    "step62950",
#    "step65948",
#    "step104916",
#    "step119903"
#]

REVISIONS=(
  "step2999"
  "step5996"
  "step8994"
  "step11992"
  "step14991"
  "step17988"
  "step20986"
  "step23984"
  "step26982"
  "step29979"
  "step32976"
  "step35974"
  "step38972"
  "step41969"
  "step44967"
  "step47963"
  "step50960"
  "step53957"
  "step56955"
  "step59953"
  "step62950"
  "step65948"
  "step68944"
  "step71942"
  "step74940"
  "step77938"
  "step80935"
  "step83933"
  "step86931"
  "step89929"
  "step92926"
  "step95924"
  "step98921"
  "step101918"
  "step104916"
  "step107913"
  "step110911"
  "step113909"
  "step116906"
  "step119903"
  "step122900"
  "step125897"
  "step128895"
  "step131892"
  "step134889"
  "step137887"
  "step140884"
  "step143882"
  "step146880"
  "step149879"
)

REVISIONS2 = (
  "step104916"
  "step107913"
  "step110911"
  "step113909"
  "step116906"
  "step119903"
  "step122900"
  "step125897"
  "step128895"
  "step131892"
  "step134889"
  "step137887"
  "step140884"
  "step143882"
  "step146880"
)

REVISIONS3 = (
""
)


BATCH_SIZE=64
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

      python3 predict_ewok.py \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --revision "$REV" \
        $CAUSAL_FLAG

      rm -rf ~/.cache/huggingface/transformers
      rm -rf ~/.cache/huggingface/hub

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

      rm -rf ~/.cache/huggingface/transformers
      rm -rf ~/.cache/huggingface/hub

      python3 predict_blimp.py \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --revision "$REV" \
        $CAUSAL_FLAG

      rm -rf ~/.cache/huggingface/transformers
      rm -rf ~/.cache/huggingface/hub

      python3 predict_comps.py \
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