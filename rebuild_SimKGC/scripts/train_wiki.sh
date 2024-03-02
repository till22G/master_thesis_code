#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

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


OUTPUT_DIR="/work/tgalla/integrate_context/wiki5m_trans/with_descriptions/tail_context"

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 3e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 3e-5 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 4 \
--num-epochs 1 \
--use-tail-context \
--use-descriptions \
--use-context-relation \
--max-number-tokens 150 \