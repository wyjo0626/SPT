export MODELS_NAME="bert-base-uncased"
export TASK_NAME=super_glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=XPROMPT_TUNING

max_seq_length=256
bs=32
max_steps=30000
weight_decay=1e-5
virtual_tokens_list="80"

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in boolq; do
    for lr in 1e-3 1e-2; do
      for var in $virtual_tokens_list; do
        if test "$DATASET_NAME" = "multirc"; then max_seq_length=348; fi

        python run.py \
          --model_name_or_path $MODEL_NAME \
          --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$PEFT_TYPE-$var-token \
          --task_name $TASK_NAME \
          --dataset_name $DATASET_NAME \
          --do_train \
          --do_eval \
          --do_predict \
          --per_device_train_batch_size $bs \
          --per_device_eval_batch_size $bs \
          --max_seq_length $max_seq_length \
          --output_dir checkpoints/PEFT/$PEFT_TYPE/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$var-token/ \
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
          --peft_type $PEFT_TYPE \
          --num_virtual_tokens $var;
      done;
    done;
  done;
done;