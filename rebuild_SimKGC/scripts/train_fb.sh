#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"

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
--pretrained-model distilbert-base-uncased \
--learning-rate 1e-5  \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--task ${TASK} \
--batch-size 1024 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 2 \
--finetune-t \
--num-epochs 10 \
--num-workers 12 \
--max-num-desc-tokens 50 \
#--max-context-size $MAX_CONTEXT_SIZE \
#--use-head-context \
#--use-tail-context \
#--use-descriptions \
#--use-context-relation \
#--use-link-graph \
#--description-length 15
#--custom-model-init 