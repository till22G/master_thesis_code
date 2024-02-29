#!/usr/bin/env bash

set -e
set -x

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$MAX_CONTEXT_SIZE" ]; then
  MAX_CONTEXT_SIZE=512
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi

DATA_DIR="${DIR}/data/"$TASK

OUTPUT_DIR=/work/tgalla/relation_freq/most_common_first/WN18RR

python3 -u main.py \
--pretrained-model prajjwal1/bert-mini \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--finetune-t \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--num-workers 12 \
--num-epochs 50 \
--max-number-tokens 512 \
--use-head-context \
--use-tail-context \
--max-context-size 3 \
--most-common-first \
--use-descriptions \

OUTPUT_DIR=/work/tgalla/relation_freq/least_common_first/WN18RR

python3 -u main.py \
--pretrained-model prajjwal1/bert-mini \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--finetune-t \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--num-workers 12 \
--num-epochs 50 \
--max-number-tokens 512 \
--use-head-context \
--use-tail-context \
--max-context-size 3 \
--least-common-first \
--use-descriptions \