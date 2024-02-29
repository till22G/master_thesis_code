#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$MAX_CONTEXT_SIZE" ]; then
  MAX_CONTEXT_SIZE=512
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then 
  DATA_DIR="${DIR}/data/${TASK}"
fi

OUTPUT_DIR=/work/tgalla/relation_freq/least_common_first/FB15k237/

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model prajjwal1/bert-mini \
--learning-rate 1e-5  \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--task ${TASK} \
--batch-size 1024 \
--additive-margin 0.02 \
--use-amp \
--use-self-negatives \
--pre-batch 2 \
--finetune-t \
--num-epochs 10 \
--num-workers 36 \
--max-number-tokens 512 \
--use-head-context \
--use-tail-context \
--max-context-size 3 \
--least-common-first \
--use-descriptions \

OUTPUT_DIR=/work/tgalla/relation_freq/most_common_first/FB15k237/

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model prajjwal1/bert-mini \
--learning-rate 1e-5  \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--task ${TASK} \
--batch-size 1024 \
--additive-margin 0.02 \
--use-amp \
--use-self-negatives \
--pre-batch 2 \
--finetune-t \
--num-epochs 10 \
--num-workers 36 \
--max-number-tokens 512 \
--use-head-context \
--use-tail-context \
--max-context-size 3 \
--most-common-first \
--use-descriptions \