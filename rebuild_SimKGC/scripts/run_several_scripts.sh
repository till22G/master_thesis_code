#!/usr/bin/env bash

terminate_scripts() {
    echo "Terminating scripts..."
    # Send termination signal to the currently running background process
    kill -TERM "$child_pid"
    exit
    echo "All scripts terminated successfully"
}

trap terminate_scripts SIGTERM

# Function to run a script and store its PID
run_training() {
    MAX_CONTEXT_SIZE="$1" OUTPUT_DIR="$2" bash "$3"
}

run_eval(){
    bash "$1" "$2" "$3"
}

run_training_wiki() {
    MAX_CONTEXT_SIZE="$1" OUTPUT_DIR="$2" bash "$3" "$4"
}

run_eval_wiki(){
    bash "$1" "$2"
}

#run_training_wiki 500 "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_all/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait
run_training_wiki 5 "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_5/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait
run_training_wiki 4 "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_4/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait
run_training_wiki 3 "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_3/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait
run_training_wiki 2 "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_2/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait
run_training_wiki 1 "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_1/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait

export CUDA_VISIBLE_DEVICES=0,1

run_training_wiki 0 "../distilBERT_new_descriptions_with_head_and_tail_context_and_relation_0/wiki5m_trans" "scripts/train_wiki.sh" "wiki5m_trans" & wait

export CUDA_VISIBLE_DEVICES=0

run_eval_wiki "scripts/eval_wiki5m_trans.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_5/wiki5m_trans/model_best.mdl" & wait
run_eval_wiki "scripts/eval_wiki5m_trans.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_4/wiki5m_trans/model_best.mdl" & wait
run_eval_wiki "scripts/eval_wiki5m_trans.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_3/wiki5m_trans/model_best.mdl" & wait
run_eval_wiki "scripts/eval_wiki5m_trans.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_2/wiki5m_trans/model_best.mdl" & wait
run_eval_wiki "scripts/eval_wiki5m_trans.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_1/wiki5m_trans/model_best.mdl" & wait
run_eval_wiki "scripts/eval_wiki5m_trans.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_and_relation_0/wiki5m_trans/model_best.mdl" & wait

echo "done"