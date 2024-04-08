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


class CPromptTuningMixture(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    RESIDUAL_PROMPT_TUNING = "RESIDUAL_PROMPT_TUNING"


class CPromptTuningConvolutionType(str, enum.Enum):
    DEFAULT = "DEFAULT"
    CONN = "CONN"


class CPromptTuningReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"


@dataclass
class CPromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        output_embeddings (`int`): The output channel arguments to use for nn.Conv1d initialization.
        conv_out_channels (`List[int]`): List of convolution layer out_channels to create convolution.
                For example, [50, 40, 20]"
                If you don't add convolution layer, then only add 1x1 convolution.
        conv_kernel_sizes (`List[str]`): List of convolution layer kernel to create convolution.
                For example, ['bottleneck', 3, 1]
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
    conv_type: Union[str, CPromptTuningConvolutionType] = field(
        default=CPromptTuningConvolutionType.DEFAULT,
        metadata={"help": "Convolutional Layer type"}
    )
    output_embeddings: Optional[int] = field(
        default=10,
        metadata={
            "help": "The output channel arguments to use for nn.Conv1d initialization."
        }
    )
    conv_out_channels: Union[List[int]] = field(
        default=None,
        metadata={
            "help": "List of convolution layer out_channels to create convolution."
            "For example, [50, 40, 20]"
            "If you don't add convolution layer, then only add 1x1 convolution."
        }
    )
    conv_kernel_sizes: Union[List[str]] = field(
        default=None,
        metadata={
            "help": "List of convolution layer kernel or bottleneck to create convolution."
            "For example, [3, 5, 7, 'bottleneck']"
        }
    )
    conv_bias: bool = field(
        default=False,
        metadata={
            "help": "Set this the False if you don't add bias to conv layers."
        }
    )
    conv_pool: bool = field(
        default=False,
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
    prompt_tuning_type: Union[str, CPromptTuningMixture] = field(
        default=CPromptTuningMixture.PROMPT_TUNING,
        metadata={"help": "prompt tuning type"}
    )
    encoder_reparameterization_type: Union[str, CPromptTuningReparameterizationType] = field(
        default=CPromptTuningReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the prompt encoder"}
    )
    encoder_nonlinearity: Union[CPromptTuningActivation, str] = field(
        default=CPromptTuningActivation.RELU,
        metadata={
            "help": "The type of activation function."
        }
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
    encoder_residual: bool = field(
        default=True,
        metadata={"help": "Set this the False if you don't use residual connection."}
    )

    def __post_init__(self):
        self.peft_type = PeftType.CPROMPT_TUNING

        self.conv_kernel_sizes = [int(item) if isinstance(item, str) and item.isdigit() else item for item in self.conv_kernel_sizes ]
        if isinstance(self.conv_out_channels, list):
            if not isinstance(self.conv_kernel_sizes, list):
                raise ValueError(f"convolution layers list is not matched with kernel size list")
            if len(self.conv_out_channels) != len(self.conv_kernel_sizes):
                raise ValueError(f"convolution layer list is not matched with {self.conv_out_channels}-{self.conv_kernel_sizes}")
            if self.conv_out_channels[-1] != self.output_embeddings:
                raise ValueError(f"covolution last layer [{self.conv_out_channels[-1]}] is not matched with out embeddings {self.output_embeddings}")
