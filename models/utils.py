import logging
import functools

from enum import Enum
from copy import deepcopy
from typing import Optional

from utils.general import colorstr, colorformat, emojis
from .training import BaseTrainer, BaseSeq2SeqTrainer

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    TaskType
)

from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    ResidualPromptTuningConfig,
    BitFitConfig,
    XPromptTuningConfig,
    RPromptTuningConfig,
    CPromptTuningConfig,
    EPromptTuningConfig,
)

AUTO_MODEL = {
    TaskType.SEQ_CLS: AutoModelForSequenceClassification,
    TaskType.SEQ_2_SEQ_LM: AutoModelForSeq2SeqLM,
    TaskType.CAUSAL_LM: AutoModelForCausalLM,
    TaskType.TOKEN_CLS: AutoModelForTokenClassification,
    TaskType.QUESTION_ANS: AutoModelForQuestionAnswering,
}

AUTO_PEFT = {
    PeftType.ADALORA: AdaLoraConfig,
    PeftType.ADAPTION_PROMPT: AdaptionPromptConfig,
    PeftType.LORA: LoraConfig,
    PeftType.P_TUNING: PromptEncoderConfig,
    PeftType.PREFIX_TUNING: PrefixTuningConfig,
    PeftType.PROMPT_TUNING: PromptTuningConfig,
    PeftType.RESIDUAL_PROMPT_TUNING: ResidualPromptTuningConfig,
    PeftType.BITFIT: BitFitConfig,
    PeftType.XPROMPT_TUNING: XPromptTuningConfig,
    PeftType.RPROMPT_TUNING: RPromptTuningConfig,
    PeftType.CPROMPT_TUNING: CPromptTuningConfig,
    PeftType.EPROMPT_TUNING: EPromptTuningConfig,
}

logger = logging.getLogger(__name__)


def get_model(model_args, peft_args, task_type: TaskType, tokenizer, dataset):
    logger.info(f"{colorstr('bright_blue', 'bold', '*** Model Initialization Start ***')}")
    model_class = AUTO_MODEL[task_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        num_labels=dataset.num_labels
    )
    
    all_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"***** {colorstr('bright_yellow', 'bold', 'orig total param')} is {colorstr('bright_yellow', 'bold', f'{all_param}')} *****")
    
    if peft_args.peft_type is not None:
        # Mapping peft config
        peft_class = AUTO_PEFT[peft_args.peft_type]
        peft_dict = {}
        
        peft_set = set(peft_class.__dataclass_fields__.keys())
        peft_args_set = set(peft_args.__dataclass_fields__.keys())
        common_set = peft_set & peft_args_set
        
        for key in common_set:
            peft_dict[key] = peft_args.__getattribute__(key)
        
        peft_config = peft_class(task_type=task_type, **peft_dict)
        logger.info(f"{colorstr('bright_yellow', 'bold', 'Peft parameters')} {peft_config}")
        
        model = get_peft_model(model, peft_config)
        model.peft_config[model.active_adapter].inference_mode = False
        
        peft_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"***** {colorstr('bright_yellow', 'bold', 'peft total param')} is {colorstr('bright_yellow', 'bold', f'{peft_param}')} *****")
        
        model.print_trainable_parameters()
    
    return model


def get_trainer(model_args, data_args, training_args, peft_args, Dataset):
    logger.info(f"{colorstr('bright_blue', 'bold', '*** Trainer Initialization Start ***')}")
    
    if any(k in model_args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        padding_side=padding_side,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # TODO: check if this is correct
    
    dataset = Dataset(data_args, model_args, training_args, tokenizer)
    
    if any(x in model_args.model_name_or_path for x in ["bert", "roberta", "albert"]):
        logger.info(f"Loading encoder model from {model_args.model_name_or_path}.")
        task_type = TaskType.SEQ_CLS
    elif any(x in model_args.model_name_or_path for x in ["t5"]):
        logger.info(f"Loading seq2seq model from {model_args.model_name_or_path}.")
        task_type = TaskType.SEQ_2_SEQ_LM
    elif any(x in model_args.model_name_or_path for x in ["gpt"]): # TODO : add 라마 추가하기
        logger.info(f"Loading decoder model from {model_args.model_name_or_path}.")
        task_type = TaskType.CAUSAL_LM
    else:
        raise NotImplementedError
    
    model = get_model(model_args, peft_args, task_type, tokenizer, dataset)
    
    if task_type == TaskType.SEQ_CLS:
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator
        )
    elif task_type == TaskType.SEQ_2_SEQ_LM or task_type == TaskType.CAUSAL_LM:
        trainer = BaseSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )
    
    return trainer, dataset.predict_dataset
