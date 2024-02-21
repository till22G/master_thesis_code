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

OUTPUT_DIR=/work/tgalla/integrate_context/WN18RR/with_relations/head_context/

python3 -u main.py \
--pretrained-model bert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 1e-4 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50 \
--max-num-desc-tokens 50 \
--max-number-tokens 150 \
--use-descriptions \
--use-head-context \
--use-context-relation \
--max-context-size 512

wait

OUTPUT_DIR=/work/tgalla/integrate_context/WN18RR/with_relations/tail_context/

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 1e-4 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50 \
--max-num-desc-tokens 50 \
--max-number-tokens 150 \
--use-descriptions \
--use-tail-context \
--use-context-relation \
--max-context-size 512

wait 

OUTPUT_DIR=/work/tgalla/integrate_context/WN18RR/with_relations/head_and_tail_context/

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 1e-4 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50 \
--max-num-desc-tokens 50 \
--max-number-tokens 150 \
--use-descriptions \
--use-head-context \
--use-tail-context \
--use-context-relation \
--max-context-size 512

wait

OUTPUT_DIR=/work/tgalla/integrate_context/WN18RR/no_relations/head_context/

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 1e-4 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50 \
--max-num-desc-tokens 50 \
--max-number-tokens 150 \
--use-descriptions \
--use-head-context \
--max-context-size 512

wait

OUTPUT_DIR=/work/tgalla/integrate_context/WN18RR/no_relations/tail_context/

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 1e-4 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50 \
--max-num-desc-tokens 50 \
--max-number-tokens 150 \
--use-descriptions \
--use-tail-context \
--max-context-size 512

wait

OUTPUT_DIR=/work/tgalla/integrate_context/WN18RR/no_relations/head_and_tail_context/

python3 -u main.py \
--pretrained-model distilbert-base-uncased \
--model-dir $OUTPUT_DIR \
--task ${TASK} \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--learning-rate 5e-5 \
--warmup 400 \
--t 0.05 \
--finetune-t \
--additive-margin 0.02 \
--weight-decay 1e-4 \
--pre-batch-weight 0.05 \
--use-self-negatives \
--pre-batch 2 \
--use-amp \
--batch-size 1024 \
--grad-clip 10 \
--num-workers 12 \
--num-epochs 50 \
--max-num-desc-tokens 50 \
--max-number-tokens 150 \
--use-descriptions \
--use-head-context \
--use-tail-context \
--max-context-size 512