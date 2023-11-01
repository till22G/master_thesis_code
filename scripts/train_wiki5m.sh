#!/usr/bin/env bash

set -e
set -x

TASK="wiki5m_trans"

DIR="$( cd "$( dirname "$0" )" && cd .. && cd rebuild_SimKGC && pwd)"
echo "working directory: ${DIR}"

cd "$( dirname "$0" )" && cd .. && cd rebuild_SimKGC

DATA_DIR="../data/"$TASK

python3 -u main.py \
--pretrained-model bert-base-uncased \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--max-number-tokens 50 \
--use-descriptions \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--batch-size 512 \
--pre-batch 0 \
--use-inverse-triples \
--use-self-negatives \
--pre-batch-weight 0.05 \
--learning-rate 2e-5 \
--weight-decay 1e-4 \
--warmup 400 \
--use-amp \
--grad-clip 10 \
--num-workers 8 \
--num-epochs 1