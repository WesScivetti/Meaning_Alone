#!/bin/bash


#python3 predict.py \
#  --model_name "jhu-clsp/ettin-encoder-400m" \
#  --batch_size 16 \
#  --revision "" \

MODELS=(
  "jhu-clsp/ettin-encoder-400m"
  "jhu-clsp/ettin-encoder-1b"
  "BERT-base-uncased"
  "BERT-large-uncased"
  "EleutherAI/ModernBERT-Base"
  "EleutherAI/ModernBERT-Large"
  "roberta-base"
  "roberta-large"
  "multibert seed 0"
  "multibert seed 1"
)

for ENTRY in "${MODELS[@]}"; do
  python3 predict.py \
          --input_tsv "data/combined_easy.tsv" \
          --output_tsv "combined_easyPLL.tsv" \
          --model_name "$MODEL" \
          --batch_size 32 \
          --topk 2 \
          --mask_str "[MASK]" \
          --candidate "" \
          --revision ""

  rm -rf ~/.cache/huggingface/transformers
  rm -rf ~/.cache/huggingface/hub
done


echo "done"