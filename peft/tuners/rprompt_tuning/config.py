# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PromptLearningConfig
from peft.utils import PeftType


class RPromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class RPromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`RPromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_kwargs (`dict`, *optional*):
            The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if `prompt_tuning_init` is
            `TEXT`.
    """
    rprompt_tuning_init: Union[RPromptTuningInit, str] = field(
        default=RPromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    rprompt_tuning_init_text: Optional[str] = field(
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
    prune_step: int = field(
        default=15000,
        metadata={
            "help": "Pruning is performed at this step, followed by rewinding in the remaining step"
        }
    )
    token_pieces: int = field(
        default=16,
        metadata={"help": "Separate the embedding vector in k pieces"}
    )
    token_ratio: float = field(
        default=0.5,
        metadata={"help": "The ratio to prune for soft prompt tokens"}
    )
    piece_ratio: float = field(
        default=0.5,
        metadata={"help": "The ratio to prune for soft prompt piece"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.RPROMPT_TUNING

        if self.tokenizer_kwargs and (self.prompt_tuning_init != RPromptTuningInit.TEXT):
            raise ValueError(
                f"tokenizer_kwargs only valid when using prompt_tuning_init='{RPromptTuningInit.TEXT.value}'."
            )
