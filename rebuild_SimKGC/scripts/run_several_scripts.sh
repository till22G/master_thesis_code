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

#bash scripts/eval.sh /work/tgalla/context_amount/WN18RR/with_descriptions/with_relations/head_and_tail_context/max_n_1/model_checkpoint_50.mdl WN18RR & wait
#bash scripts/eval.sh /work/tgalla/context_amount/WN18RR/with_descriptions/with_relations/head_and_tail_context/max_n_3/model_checkpoint_50.mdl WN18RR & wait
#bash scripts/eval.sh /work/tgalla/context_amount/WN18RR/with_descriptions/with_relations/head_and_tail_context/max_n_5/model_checkpoint_50.mdl WN18RR & wait
#bash scripts/eval.sh /work/tgalla/context_amount/WN18RR/with_descriptions/with_relations/head_and_tail_context/max_n_512/model_checkpoint_50.mdl WN18RR & wait

#bash scripts/eval.sh /work/tgalla/integrate_context_512/FB15k237/no_relations/head_context/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/integrate_context_512/FB15k237/no_relations/tail_context/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/context_amount/FB15k237/no_relations/head_and_tail_context/n_5/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/context_amount/FB15k237/no_relations/head_and_tail_context/n_10/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/context_amount/FB15k237/no_relations/head_and_tail_context/n_20/model_checkpoint_10.mdl FB15k237 & wait
#bash scripts/eval.sh /work/tgalla/context_amount/FB15k237/no_relations/head_and_tail_context/n_50/model_checkpoint_10.mdl FB15k237 & wait


bash scripts/train_fb.sh

wait

bash scripts/train_wn18rr.sh

echo "done"