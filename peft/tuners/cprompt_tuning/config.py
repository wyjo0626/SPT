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
from typing import Optional, Union, List

from peft.config import PromptLearningConfig
from peft.utils import PeftType


class CPromptTuningActivation(str, enum.Enum):
    RELU = "RELU"
    TANH = "TANH"
    SIGM = "SIGM"


@dataclass
class CPromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        output_embeddings (`int`): The output channel arguments to use for nn.Conv1d initialization.
        conv_out_channels (`List[int]`): List of convolution layer out_channels to create convolution.
                For example, [50, 40, 20]"
                If you don't add convolution layer, then only add 1x1 convolution.
        conv_kernel_sizes (`List[int]`): List of convolution layer kernel to create convolution.
                For example, [3, 5, 7]
                If you don't add convolution layer, then only add 1x1 convolution.
        conv_bias (`bool`): Set this the False if you don't add bias to conv layers.
        conv_pool (`bool`): Set this the False if you don't add max pooling to conv layers.
        encoder_nonlinearity (`CPromptTuningActivation`): The type of activation function.
        encoder_layer_norm (`bool`): Set this the False if you don't use layer normalization.
        encoder_dropout (`float`): Set this 0.0 if you don't use dropout.
        encoder_bottleneck (`int`): The type of bottleneck size.
        encoder_num_modules (`int`): The number of modules of the mlp.
        encoder_residual (`int`): Set this the False, if you don't add residual connection.
    """
    output_embeddings: Optional[int] = field(
        default=10,
        metadata={
            "help": "The output channel arguments to use for nn.Conv1d initialization."
        }
    )
    conv_out_channels: Optional[Union[List[int]]] = field(
        default=None,
        metadata={
            "help": "List of convolution layer out_channels to create convolution."
            "For example, [50, 40, 20]"
            "If you don't add convolution layer, then only add 1x1 convolution."
        }
    )
    conv_kernel_sizes: Optional[Union[List[int]]] = field(
        default=None,
        metadata={
            "help": "List of convolution layer kernel to create convolution."
            "For example, [3, 5, 7]"
            "If you don't add convolution layer, then only add 1x1 convolution."
        }
    )
    conv_bias: bool = field(
        default=True,
        metadata={
            "help": "Set this the False if you don't add bias to conv layers."
        }
    )
    conv_pool: bool = field(
        default=True,
        metadata={
            "help": "Set this the False if you don't add max pooling to conv layers."
        }
    )
    conv_nonlinearity: Union[CPromptTuningActivation, str] = field(
        default=CPromptTuningActivation.RELU,
        metadata={
            "help": "The type of activation function."
        }
    )
    conv_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Set this the False if you don't use layer normalization."
        }
    )
    conv_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Set this 0.0 if you don't use dropout."
        }
    )
    conv_residual: int = field(
        default=True,
        metadata={
            "help": "Set this the False, if you don't add residual connection."
        }
    )

    def __post_init__(self):
        self.peft_type = PeftType.CPROMPT_TUNING

        if isinstance(self.conv_out_channels, list):
            if not isinstance(self.conv_kernel_sizes, list):
                raise ValueError(f"convolution layers list is not matched with kernel size list")
            if len(self.conv_out_channels) != len(self.conv_kernel_sizes):
                raise ValueError(f"convolution layer list is not matched with {self.conv_out_channels}-{self.conv_kernel_sizes}")
