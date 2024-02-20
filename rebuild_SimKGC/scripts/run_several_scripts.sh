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

bash scripts/eval.sh /work/tgalla/replication_results/BERT_base_ib/_wiki5m_trans/best_model_checkpoint.mdl wiki5m_trans & wait
bash scripts/eval.sh /work/tgalla/replication_results/BERT_base_ib_pb/_wiki5m_trans/best_model_checkpoint.mdl wiki5m_trans & wait
bash scripts/eval.sh /work/tgalla/replication_results/BERT_base_ib_sn/_wiki5m_trans/best_model_checkpoint.mdl wiki5m_trans & wait

echo "done"