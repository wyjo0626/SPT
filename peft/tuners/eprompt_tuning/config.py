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


class EPTReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"


class EPTActivationType(str, enum.Enum):
    NONE = "NONE"
    RELU = "RELU"
    TANH = "TANH"
    SIGM = "SIGM"


class EPTResidualType(str, enum.Enum):
    NONE = "NONE"
    INPUT = "INPUT"
    EXP = "EXP"


@dataclass
class EPromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`ExplosivePromptEmbedding`].

    Args:
        ept_reparameterization_type (`EPTReparameterizationType`): How to reparameterize the ept encoder
        ept_nonlinearity (`EPTActivationType`): the type of activation function
        ept_hidden_size (`int`): The hidden size of ept encoder
        ept_num_layers (`int`): The number of layers of the ept encoder
        ept_dropout (`float`): The dropout of the ept encoder
        ept_layer_norm (`bool`): Set this the True if you use layer normalization
        ept_residual (`bool`): Set this the True if you use residual connection
    """
    ept_reparameterization_type: Union[str, EPTReparameterizationType] = field(
        default=EPTReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the ept encoder"},
    )
    ept_nonlinearity: Union[EPTActivationType, str] = field(
        default=EPTActivationType.NONE,
        metadata={"help": "the type of activation function."}
    )
    ept_residual: Union[EPTResidualType, str] = field(
        default=EPTResidualType.NONE,
        metadata={"help": "the type of residual connection."}
    )
    ept_hidden_size: int = field(
        default=1,
        metadata={"help": "The hidden size of ept encoder"},
    )
    ept_num_layers: int = field(
        default=1,
        metadata={"help": "The number of layers of the ept encoder"},
    )
    ept_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the ept encoder"},
    )
    ept_bias: bool = field(
        default=False,
        metadata={"help": "Set this the True if you use layer bias"}
    )
    ept_layer_norm: bool = field(
        default=False,
        metadata={"help": "Set this the True if you use layer normalization"}
    )
    
    def __post_init__(self):
        self.peft_type = PeftType.EPROMPT_TUNING