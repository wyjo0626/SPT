from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments
from peft import (
    PeftType, 
    PromptTuningInit,
    XPromptTuningInit, 
    LoftQConfig, 
    PromptEncoderReparameterizationType, 
    ResidualPromptTuningInit,
    ResidualPromptTuningReparameterizationType
)

from tasks.utils import *


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: Optional[str] = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        }
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    add_prefix: bool = field(
        default=False,
        metadata={
            "help": "Whether add the prefix before each example, typically using the name of the dataset."
        }
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    split_validation_test: Optional[bool] = field(
        default=True,
        metadata={"help": "If set, for the datasets which do not have the test set, we use validation set as their"
                    "test set and make a validation set from either splitting the validation set into half (for smaller"
                    "than 10K samples datasets), or by using 1K examples from training set as validation set (for larger"
                    " datasets)."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={"help": "The specific prompt string to use"}
    )
    k_shot_example: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of examples to use for the k-shot learning."
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )


@dataclass
class DynamicTrainingArguments(Seq2SeqTrainingArguments):
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to get the predictions."}
    )
    generation_max_length: Optional[int] = field(
        default=20,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

@dataclass
class DynamicPeftArguments:
    peft_type: Optional[PeftType] = field(
        default=None,
        metadata={
            "help": "Training with Peft or Fine-Tuning"
        }
    )
    # PromptLearningConfig
    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    
    # PromptTuningConfig
    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    
    # PrefixTuningConfig
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )
    
    # PromptEncoderConfig
    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(
        default=PromptEncoderReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the prompt encoder"},
    )
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the prompt encoder"},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the prompt encoder"},
    )
    
    # LoraConfig
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    ),
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    ),
    
    # AdaptionPromptConfig
    adapter_len: int = field(default=None, metadata={"help": "Number of adapter tokens to insert"})
    adapter_layers: int = field(default=None, metadata={"help": "Number of adapter layers (from the top)"})
    
    # AdaLoraConfig
    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Intial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    
    # ResidualPromptTuningConfig
    residual_prompt_tuning_init: Union[ResidualPromptTuningInit, str] = field(
        default=ResidualPromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    residual_prompt_tuning_init_text: Optional[str] = field(default=None, metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"},
    )
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"},
    )
    tokenizer_kwargs: Optional[dict] = field(default=None,metadata={
            "help": ("The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if prompt_tuning_init is `TEXT`"),},
    )
    encoder_reparameterization_type: Union[str, ResidualPromptTuningReparameterizationType] = field(
        default=ResidualPromptTuningReparameterizationType.MLP, metadata={"help": "How to reparameterize of the prompt."}
    )
    encoder_bottleneck_size: int = field(default=800, metadata={"help": "The bottleneck size of the mlp."})
    encoder_num_layers: int = field(default=1, metadata={"help": "The number of layers of the mlp."})
    encoder_dropout: int = field(default=0.0, metadata={"help": "The dropout of the mlp."})
    encoder_layer_norm: bool = field(default=True, metadata={"help": "Set this the False if you don't use layer normalization"})
    encoder_separate: bool = field(default=True, metadata={"help": "Use separate MLP for each prompt tokens"})
    residual: bool = field(default=True, metadata={"help": "Set this the False if you don't use residual connection."})
    
    # BitFitConfig
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    ),
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    ),
    
    # XPromptTuningConfig
    xprompt_tuning_init: Union[XPromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    xprompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if prompt_tuning_init is "
                "`TEXT`"
            ),
        },
    )
    prune_steps: int = field(
        default=4,
        metadata={
            "help": "Pruning as much as steps during training"
        }
    )
    token_ratio: int = field(
        default=0.5,
        metadata={"help": "The ratio to prune for soft prompt tokens"}
    )
    piece_ratio: int = field(
        default=0.5,
        metadata={"help": "The ratio to prune for soft prompt piece"}
    )
    token_piece: int = field(
        default=16,
        metadata={"help": "Separate the embedding vector in k pieces"}
    )


def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DynamicTrainingArguments, DynamicPeftArguments))
    
    args = parser.parse_args_into_dataclasses()
    
    return args