#!/bin/bash

# ===== USER SETTINGS =====
HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"

# List of input TSV files
INPUT_FILES=(
  "data/syntactic_tests_combined.tsv"
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
  "EleutherAI/pythia-14m-deduped true"
  "EleutherAI/pythia-70m-deduped true"
  "EleutherAI/pythia-160m-deduped true"
  "EleutherAI/pythia-410m-deduped true"
)

REVISIONS100=(
  "step256"
)

REVISIONS=(
  "step1"
  "step2"
  "step4"
  "step8"
  "step16"
  "step32"
  "step64"
  "step128"
  "step512"
  "step1000"
  "step2000"
  "step3000"
  "step4000"
  "step5000"
  "step6000"
  "step7000"
  "step8000"
  "step9000"
  "step10000"
  "step11000"
  "step12000"
  "step13000"
  "step14000"
  "step15000"
  "step16000"
  "step17000"
  "step18000"
  "step19000"
  "step20000"
  "step21000"
  "step22000"
  "step23000"
  "step24000"
  "step25000"
  "step26000"
  "step27000"
  "step28000"
  "step29000"
  "step30000"
  "step31000"
  "step32000"
  "step33000"
  "step34000"
  "step35000"
  "step36000"
  "step37000"
  "step38000"
  "step39000"
  "step40000"
  "step41000"
  "step42000"
  "step43000"
  "step44000"
  "step45000"
  "step46000"
  "step47000"
  "step48000"
  "step49000"
  "step50000"
  "step51000"
  "step52000"
  "step53000"
  "step54000"
  "step55000"
  "step56000"
  "step57000"
  "step58000"
  "step59000"
  "step60000"
  "step61000"
  "step62000"
  "step63000"
  "step64000"
  "step65000"
  "step66000"
  "step67000"
  "step68000"
  "step69000"
  "step70000"
  "step71000"
  "step72000"
  "step73000"
  "step74000"
  "step75000"
  "step76000"
  "step77000"
  "step78000"
  "step79000"
  "step80000"
  "step81000"
  "step82000"
  "step83000"
  "step84000"
  "step85000"
  "step86000"
  "step87000"
  "step88000"
  "step89000"
  "step90000"
  "step91000"
  "step92000"
  "step93000"
  "step94000"
  "step95000"
  "step96000"
  "step97000"
  "step98000"
  "step99000"
  "step100000"
  "step101000"
  "step102000"
  "step103000"
  "step104000"
  "step105000"
  "step106000"
  "step107000"
  "step108000"
  "step109000"
  "step110000"
  "step111000"
  "step112000"
  "step113000"
  "step114000"
  "step115000"
  "step116000"
  "step117000"
  "step118000"
  "step119000"
  "step120000"
  "step121000"
  "step122000"
  "step123000"
  "step124000"
  "step125000"
  "step126000"
  "step127000"
  "step128000"
  "step129000"
  "step130000"
  "step131000"
  "step132000"
  "step133000"
  "step134000"
  "step135000"
  "step136000"
  "step137000"
  "step138000"
  "step139000"
  "step140000"
  "step141000"
  "step142000"
  "step143000"
)



BATCH_SIZE=100
REVISION=""



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

  for REV in "${REVISIONS100[@]}"; do
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
