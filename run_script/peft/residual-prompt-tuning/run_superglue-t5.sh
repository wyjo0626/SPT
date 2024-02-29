export MODELS_NAME="t5-base t5-large"
export TASK_NAME=super_glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=RESIDUAL_PROMPT_TUNING

max_seq_length=256
bs=16
max_steps=30000
lrs="3e-3 1e-2 3e-2 1e-1 3e-1"
weight_decay=0.01
seed=42
init_type=RANDOM_UNIFORM
virtual_token=10

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in boolq cb rte wic wsc multirc; do
    for lr in $lrs; do
      if test "$DATASET_NAME" = "multirc"; then max_seq_length=348; fi
      python run.py \
        --model_name_or_path $MODEL_NAME \
        --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$seed-$PEFT_TYPE-$virtual_token-token \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --max_seq_length $max_seq_length \
        --output_dir checkpoints/PEFT/$PEFT_TYPE/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$seed-$virtual_token-token/ \
        --overwrite_output_dir \
        --seed $seed \
        --learning_rate $lr \
        --save_strategy steps \
        --evaluation_strategy steps \
        --max_steps $max_steps \
        --eval_steps 1000 \
        --save_steps 1000 \
        --warmup_steps 500 \
        --weight_decay $weight_decay \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --peft_type $PEFT_TYPE \
        --init_type $init_type \
        --num_virtual_tokens $virtual_token;
    done;
  done;
done;