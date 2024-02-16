export MODELS_NAME="bert-base-uncased bert-large-uncased gpt2-medium gpt2-large t5-base t5-large"
export TASK_NAME=glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=PREFIX_TUNING

lr=1e-3
max_seq_length=256
bs=32
max_steps=30000
weight_decay=1e-5
k_shot=10
virtual_tokens_list="20"

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in cola mrpc rte stsb wnli mnli qnli qqp sst2; do
    for var in $virtual_tokens_list; do
      python run.py \
        --model_name_or_path $MODEL_NAME \
        --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$PEFT_TYPE-$var-token-$k_shot-shot \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --max_seq_length $max_seq_length \
        --output_dir checkpoints/PEFT/$PEFT_TYPE/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$var-token-$k_shot-shot/ \
        --overwrite_output_dir \
        --seed 42 \
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
        --k_shot_example $k_shot \
        --peft_type $PEFT_TYPE \
        --num_virtual_tokens $var;
    done;
  done;
done;