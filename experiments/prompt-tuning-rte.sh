export MODELS_NAME="bert-base-uncased"
export TASK_NAME=super_glue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=PROMPT_TUNING

max_seq_length=256
bs=32
max_steps=30000
weight_decay=1e-5
num_virtual_tokens=40
prune_ratio=0.5

for lr in 1e-3; do
    python run_comparison.py \
        --model_name_or_path $MODELS_NAME \
        --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-PROMPT-$num_virtual_tokens-token-exp \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --max_seq_length $max_seq_length \
        --output_dir $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-PROMPT-$num_virtual_tokens-token-exp/ \
        --overwrite_output_dir \
        --seed 42 \
        --learning_rate $lr \
        --evaluation_strategy epoch \
        --num_train_epochs 20 \
        --weight_decay $weight_decay \
        --save_strategy='no' \
        --peft_type $PEFT_TYPE \
        --num_virtual_tokens $num_virtual_tokens \
        --prune_type xprompt-tuning \
        --prune_ratio $prune_ratio;
done;

for lr in 1e-3; do
    python run_comparison.py \
        --model_name_or_path $MODELS_NAME \
        --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-PROMPT-$num_virtual_tokens-token-exp \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --max_seq_length $max_seq_length \
        --output_dir $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-PROMPT-$num_virtual_tokens-token-exp/ \
        --overwrite_output_dir \
        --seed 42 \
        --learning_rate $lr \
        --evaluation_strategy epoch \
        --num_train_epochs 20 \
        --weight_decay $weight_decay \
        --save_strategy='no' \
        --peft_type $PEFT_TYPE \
        --num_virtual_tokens $num_virtual_tokens \
        --prune_type rprompt-tuning \
        --prune_ratio $prune_ratio;
done;
