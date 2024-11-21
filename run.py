import logging
import os
import sys
import numpy as np
from datasets import concatenate_datasets

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from models.utils import get_trainer
from utils.arguments import get_args
from utils.general import colorstr, colorformat, emojis
from tasks.utils import *

os.environ["WANDB_PROJECT"] = "peft"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    logger.info(colorstr('bright_blue', 'bold', '*** Train Start ***'))
    
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    metrics = train_result.metrics
    
    # trainer.save_model() # Saves the tokenizer too far easy upload
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


def evaluate(trainer):
    logger.info(colorstr('bright_blue', 'bold', '*** Evaluate Start ***'))
    
    metrics = trainer.evaluate()
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer, predict_dataset):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")
    else:
        logger.info(colorstr('bright_blue', 'bold', '*** Predict Start ***'))
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="test")
        
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    
    args = get_args()
    
    model_args, data_args, training_args, peft_args = args
    
    if peft_args.tokenizer_name_or_path is None:
        peft_args.tokenizer_name_or_path = (
            model_args.tokenizer_name if model_args.tokenizer_name is not None else  model_args.model_name_or_path
        )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity_error()
    # transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"{colorstr('bright_yellow', 'bold', 'Training/evaluation parameters')} {training_args}")
    logger.info(f"\n\n{colorstr('bright_yellow', 'bold', 'Data Parameter')} {data_args}")
    logger.info(f"\n\n{colorstr('bright_yellow', 'bold', 'Model parameters')} {model_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    
    if data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.dataset import GlueDataset
        dataset = GlueDataset
    elif data_args.task_name.lower() == "super_glue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.dataset import SuperGlueDataset
        dataset = SuperGlueDataset
    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.dataset import QADataset
        dataset = QADataset
    elif data_args.task_name.lower() == "others":
        assert data_args.dataset_name.lower() in OTHERS_DATASETS
        from tasks.others.dataset import OthersDataset
        dataset = OthersDataset
    else:
        raise NotImplementedError("Task {} is not implemented. Please choose a task from: {}".format(data_args.task_name, ", ".join(TASKS)))
    
    set_seed(training_args.seed)
    
    trainer, dataset = get_trainer(model_args, data_args, training_args, peft_args, dataset)

    if data_args.long_tail is not None and data_args.task_name.lower() in ["super_glue", "glue"] and len(dataset.raw_datasets["train"].features["label"].names) == 2:
        def label_decode(label, tokenizer):
            label = np.array(label)
            label[label < 0] = tokenizer.pad_token_id
            label = tokenizer.decode(label, skip_special_tokens=True)
            return label

        train_data = dataset.train_dataset
        predict_data = dataset.predict_dataset
        tokenizer = trainer.tokenizer

        train_data_1 = train_data.filter(lambda example: label_decode(example['labels'], tokenizer) == '1')
        train_data_0 = train_data.filter(lambda example: label_decode(example['labels'], tokenizer) == '0')
        test_data_1 = predict_data.filter(lambda example: label_decode(example['labels'], tokenizer) == '1')
        test_data_0 = predict_data.filter(lambda example: label_decode(example['labels'], tokenizer) == '0')
        
        logger.info(f"Train label Counts ['1' : {len(train_data_1)}, '0' : {len(train_data_0)}]")
        logger.info(f"Test label Counts ['1' : {len(test_data_1)}, '0' : {len(test_data_0)}]")
        
        if len(test_data_1) > len(test_data_0):
            target_data = train_data_1
            untarget_data = train_data_0
            logger.info(f"Label 1 is targeted!")
        else:
            target_data = train_data_0
            untarget_data = train_data_1
            logger.info(f"Label 0 is targeted!")

        if data_args.long_tail == 10:
            target_data = target_data.select(range(0, round(len(untarget_data) / 9)))
        elif data_args.long_tail == 20:
            target_data = target_data.select(range(0, round(len(untarget_data) / 4)))
        elif data_args.long_tail == 30:
            target_data = target_data.select(range(0, round(len(untarget_data) * 3 / 7)))

        if len(test_data_1) > len(test_data_0):
            logger.info(f"Long-tailed Train label Counts ['1' : {len(target_data)}, '0' : {len(untarget_data)}]")
        else:
            logger.info(f"Long-tailed Train label Counts ['1' : {len(untarget_data)}, '0' : {len(target_data)}]")

        balanced_data = concatenate_datasets([target_data, untarget_data]).shuffle(seed=training_args.seed)

        dataset.train_dataset = balanced_data
        trainer.train_dataset = balanced_data

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
    
    if peft_args.peft_type is not None:
        trainer.model.active_peft_config.inference_mode = True
    
    if training_args.do_eval:
        evaluate(trainer)
    
    if training_args.do_predict:
        predict(trainer, dataset.predict_dataset)