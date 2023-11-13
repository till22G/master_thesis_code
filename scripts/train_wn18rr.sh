#!/usr/bin/env bash

set -e
set -x

TASK="wn18rr"

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
--use-neighbors \
--use-descriptions \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--batch-size 1024 \
--pre-batch 0 \
--use-inverse-triples \
--use-self-negatives \
--pre-batch-weight 0.05 \
--learning-rate 5e-5 \
--weight-decay 1e-4 \
--warmup 400 \
--use-amp \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50
