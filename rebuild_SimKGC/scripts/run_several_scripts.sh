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

#bash scripts/eval.sh /work/tgalla/integrate_context/WN18RR/with_relations/head_context/model_checkpoint_50.mdl & wait
#bash scripts/eval.sh /work/tgalla/integrate_context/WN18RR/with_relations/tail_context/model_checkpoint_50.mdl & wait
#bash scripts/eval.sh /work/tgalla/integrate_context/WN18RR/with_relations/head_and_tail_context/model_checkpoint_50.mdl & wait

bash scripts/eval.sh /work/tgalla/integrate_context/WN18RR/no_relations/head_context/model_checkpoint_50.mdl & wait
bash scripts/eval.sh /work/tgalla/integrate_context/WN18RR/no_relations/tail_context/model_checkpoint_50.mdl & wait
bash scripts/eval.sh /work/tgalla/integrate_context/WN18RR/no_relations/head_and_tail_context/model_checkpoint_50.mdl & wait

#bash scripts/eval.sh /work/tgalla/integrate_context/FB15k237/with_relations/head_context/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/integrate_context/FB15k237/with_relations/tail_context/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/integrate_context/FB15k237/with_relations/head_and_tail_context/model_checkpoint_10.mdl FB15k237 & wait

#bash scripts/eval.sh /work/tgalla/integrate_context/FB15k237/no_relations/head_context/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/integrate_context/FB15k237/no_relations/tail_context/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/integrate_context/FB15k237/no_relations/head_and_tail_context/model_checkpoint_10.mdl FB15k237 & wait

echo "done"