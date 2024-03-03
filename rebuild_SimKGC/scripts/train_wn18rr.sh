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

OUTPUT_DIR=/work/tgalla/rerun_distilBERT/WN18RR/without_neighbors

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
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
--num-workers 24 \
--num-epochs 50 \
--max-number-tokens 50 \
--use-descriptions \