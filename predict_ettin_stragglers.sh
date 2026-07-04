#!/bin/bash


python3 predict_ewok.py \
  --model_name "jhu-clsp/ettin-encoder-400m" \
  --batch_size 16 \
  --revision "step137887" \

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_ewok.py \
  --model_name "jhu-clsp/ettin-encoder-400m" \
  --batch_size 16 \
  --revision "step140884" \

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_ewok.py \
  --model_name "jhu-clsp/ettin-decoder-1b" \
  --batch_size 16 \
  --revision "step86931" \
  --causallm

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_ewok.py \
  --model_name "jhu-clsp/ettin-decoder-1b" \
  --batch_size 16 \
  --revision "125897" \
  --causallm

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

echo "Done with ewok------------------------------------------"

python3 predict_blimp.py \
  --model_name "jhu-clsp/ettin-encoder-1b" \
  --batch_size 16 \
  --revision "step107913"

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_blimp.py \
  --model_name "jhu-clsp/ettin-encoder-1b" \
  --batch_size 16 \
  --revision "step131892"

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_blimp.py \
  --model_name "jhu-clsp/ettin-decoder-1b" \
  --batch_size 16 \
  --revision "step146880" \
  --causallm

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

echo "Done with blimp------------------------------------------"


python3 predict_comps.py \
  --model_name "jhu-clsp/ettin-decoder-1b" \
  --batch_size 16 \
  --revision "step83933" \
  --causallm

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_comps.py \
  --model_name "jhu-clsp/ettin-decoder-1b" \
  --batch_size 16 \
  --revision "step113909" \
  --causallm

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

python3 predict_comps.py \
  --model_name "jhu-clsp/ettin-decoder-1b" \
  --batch_size 16 \
  --revision "step143882" \
  --causallm

rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub

#python3 predict_comps.py \
#  --model_name "$MODEL" \
#  --batch_size $BATCH_SIZE \
#  --revision "$REV" \
#  $CAUSAL_FLAG
#
#rm -rf ~/.cache/huggingface/transformers
#rm -rf ~/.cache/huggingface/hub



echo "done"