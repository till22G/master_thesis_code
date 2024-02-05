#!/usr/bin/env bash

#set -x
#set -e

model_path="bert"
task="WN18RR"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    task=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

test_path="${DATA_DIR}/test.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
fi

if [ -z "$backup_in_home" ]; then
  backup_in_home=true
fi

neighbor_weight=0.05
rerank_n_hop=2
if [ "${task}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi

python3 -u evaluation.py \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.json" \
--test-path "${test_path}" "$@" 


if [ "${backup_in_home}" = true ]; then
  # copy evaluation results to home directory
  if [ $? -eq 0 ]; then
      echo "Evaluation completed successfully."
  fi

  # find saved evlauation metrics
  MODEL_DIR=$(dirname "${model_path}")
  pattern="*.txt"
  results=$(find "$MODEL_DIR" -type f -name "$pattern")

  if [ $? -eq 0 ]; then
      echo "Files found:"
      echo "$results"
  else
      echo "No files found matching the pattern '$pattern' in '$directory_path'."
  fi

  model_dir_name="${MODEL_DIR#*/}"; 
  model_dir_name="${model_dir_name#*/}" 
  model_dir_name="${model_dir_name#*/}" 
  backup_path="/home/tgalla/backup_results/${model_dir_name}"

  if [ ! -d "$backup_path" ]; then
    echo "Creating backup directory"
    mkdir -p "$backup_path"
  fi

  echo "Copying result files to ${backup_path}"
  for file in $results; do
    cp "$file" "$backup_path"
  done


fi