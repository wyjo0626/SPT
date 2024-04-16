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
from typing import Dict, Optional, Union

from peft.config import PromptLearningConfig
from peft.utils import PeftType


class ResidualPromptTuningReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"


@dataclass
class ResidualPromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a ['ResidualPromptEmbedding']
    
    Args:
        encoder_reparameterization_type (`ResidualPromptTuningReparameterizationType`): How to reparameterize of the prompt.
        encoder_bottleneck_size (`int`): The bottleneck size of the mlp.
        encoder_num_layers (`int`): The number of layers of the mlp.
        encoder_dropout (`int`): The dropout of the mlp.
        encoder_layer_norm (`bool`): Set this the False if you don't use layer normalization.
        encoder_separate (`bool`): Use separate MLP for each prompt tokens.
        residual (`bool`): Set this the False if you don't use residual connection.
    """
    encoder_reparameterization_type: Union[str, ResidualPromptTuningReparameterizationType] = field(
        default=ResidualPromptTuningReparameterizationType.MLP,
        metadata={"help": "How to reparameterize of the prompt."}
    )
    encoder_bottleneck_size: int = field(
        default=400,
        metadata={"help": "The bottleneck size of the mlp."}
    )
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the mlp."}
    )
    encoder_dropout: int = field(
        default=0.0,
        metadata={"help": "The dropout of the mlp."}
    )
    encoder_layer_norm: bool = field(
        default=True,
        metadata={"help": "Set this the False if you don't use layer normalization"}
    )
    encoder_separate: bool = field(
        default=False,
        metadata={"help": "Use separate MLP for each prompt tokens"}
    )
    residual: bool = field(
        default=True,
        metadata={"help": "Set this the False if you don't use residual connection."}
    )
    
    def __post_init__(self):
        self.peft_type = PeftType.RESIDUAL_PROMPT_TUNING
