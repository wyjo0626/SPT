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


@dataclass
class XPromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`XPromptEmbedding`].
    
    Args:
        prune_step (`int`): Pruning is performed at this step, followed by rewinding in the remaining step.
        token_pieces (`int`): Separate the embedding vector in k pieces.
        token_ratio (`int`): The ratio to prune for soft prompt tokens.
        piece_ratio (`int`): The ratio to prune for soft prompt piece
    """
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
        self.peft_type = PeftType.XPROMPT_TUNING
