export MODELS_NAME="bert-base-uncased t5-base bert-large-uncased t5-large"
export TASK_NAME=super_glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=RESIDUAL_PROMPT_TUNING

lr=0.3
max_seq_length=256
bs=32
epochs=20
weight_decay=1e-5
seed=42
virtual_tokens_list="10 100"

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in boolq cb rte wic wsc copa; do
    for var in $virtual_tokens_list; do
      if [ "$DATASET_NAME" = "boolq" ] || [ "$DATASET_NAME" = "rte" ] || [ "$DATASET_NAME" = "wic" ]; then
        epochs=20
      else
        epochs=40
      fi
      python run.py \
        --model_name_or_path $MODEL_NAME \
        --run_name EXP1-$TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$PEFT_TYPE-$var-token-$seed \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --max_seq_length $max_seq_length \
        --output_dir checkpoints/EXP1/$PEFT_TYPE/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$var-token-$seed/ \
        --overwrite_output_dir \
        --seed $seed \
        --learning_rate $lr \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --num_train_epochs $epochs \
        --warmup_steps 500 \
        --weight_decay $weight_decay \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --peft_type $PEFT_TYPE \
        --num_virtual_tokens $var \
        --encoder_bottleneck 250;
    done;
  done;
done;