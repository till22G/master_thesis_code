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


# Run each script sequentially with different OUTPUT_DIR values
run_training 500 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_all/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_all/FB15k237/model_best.mdl" "FB15k237" & wait
run_training 5 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_5/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_5/FB15k237/model_best.mdl" "FB15k237" & wait
run_training 4 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_4/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_4/FB15k237/model_best.mdl" "FB15k237" & wait
run_training 3 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_3/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_3/FB15k237/model_best.mdl" "FB15k237" & wait
run_training 2 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_2/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_2/FB15k237/model_best.mdl" "FB15k237" & wait
run_training 1 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_1/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_1/FB15k237/model_best.mdl" "FB15k237" & wait
run_training 0 "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_0/FB15k237" "scripts/train_fb.sh" & wait
run_eval "scripts/eval.sh" "../2_distilBERT_new_descriptions_with_head_and_tail_context_no_relation_0/FB15k237/model_best.mdl" "FB15k237" & wait

echo "done"