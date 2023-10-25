#!/usr/bin/env bash

set -e

TASK="fb15k237"

DIR="$( cd "$( dirname "$0" )" && cd .. && cd rebuild_SimKGC && pwd)"
echo "working directory: ${DIR}"

cd "$( dirname "$0" )" && cd .. && cd rebuild_SimKGC

python3 -u main.py \
--pretrained-model bert-base-uncased \
--task ${TASK} \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--max-number-tokens 50 \
--use-neighbors \
--use-descriptions \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--batch-size 512 \
--pre-batch 2 \
--use-inverse-triples \
--use-self-negatives \
--pre-batch-weight 0.05 \
--learning-rate 2e-5 \
--weight-decay 1e-4 \
--warmup 400 \
--use-amp \
--grad-clip 10 \
--num-workers 2
