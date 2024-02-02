#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$MAX_CONTEXT_SIZE" ]; then
  MAX_CONTEXT_SIZE=0
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model nghuyong/ernie-2.0-medium-en \
--pooling mean \
--lr 5e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 8 \
--max-to-keep 3 "$@" \
--use-head-context \
--use-tail-context \
--max-context-size $MAX_CONTEXT_SIZE \
--max-num-desc-tokens 50 \
--use-context-relation \
--use-descriptions \
#--use-link-graph \
#--description-length 15
#--custom-model-init 
