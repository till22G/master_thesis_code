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


OUTPUT_DIR=/work/tgalla/replication_results/BERT_base_ib/FB15k237

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--learning-rate 1e-5  \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--task ${TASK} \
--batch-size 1024 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 0 \
--finetune-t \
--num-epochs 10 \
--num-workers 32 \
--max-num-desc-tokens 50 \
--use-neighbors \
--use-descriptions

wait 

OUTPUT_DIR=/work/tgalla/replication_results/BERT_base_ib_sn/FB15k237

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--learning-rate 1e-5  \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--task ${TASK} \
--batch-size 1024 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--num-epochs 10 \
--num-workers 32 \
--max-num-desc-tokens 50 \
--use-neighbors \
--use-descriptions

waii


OUTPUT_DIR=/work/tgalla/replication_results/BERT_base_ib_pb/FB15k237

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--learning-rate 1e-5  \
--train-path "$DATA_DIR/train.json" \
--valid-path "$DATA_DIR/valid.json" \
--task ${TASK} \
--batch-size 1024 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 2 \
--finetune-t \
--num-epochs 10 \
--num-workers 32 \
--max-num-desc-tokens 50 \
--use-neighbors \
--use-descriptions

wait 


OUTPUT_DIR=/work/tgalla/replication_results/BERT_base_ib_sn_pb/FB15k237

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
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
--num-workers 32 \
--max-num-desc-tokens 50 \
--use-neighbors \
--use-descriptions