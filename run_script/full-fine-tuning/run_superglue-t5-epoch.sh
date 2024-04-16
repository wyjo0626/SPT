export MODELS_NAME="t5-base t5-large"
export TASK_NAME=super_glue
export CUDA_VISIBLE_DEVICES=0

max_seq_length=256
bs=16
lrs="1e-5 1e-4 1e-3"
weight_decay=0.01
seed=42

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in boolq cb rte wic wsc multirc; do
    for lr in $lrs; do
      if test "$DATASET_NAME" = "multirc"; then max_seq_length=348; else max_seq_length=256; fi
      if test "$DATASET_NAME" = "multirc"; then epochs=10; else epochs=20; fi
      python run.py \
        --model_name_or_path $MODEL_NAME \
        --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$seed \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --max_seq_length $max_seq_length \
        --output_dir checkpoints/FFT/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$seed/ \
        --overwrite_output_dir \
        --seed $seed \
        --learning_rate $lr \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --num_train_epochs $epochs \
        --warmup_steps 500 \
        --weight_decay $weight_decay \
        --load_best_model_at_end \
        --save_total_limit 1;
    done;
  done;
done;