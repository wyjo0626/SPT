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
import math

import torch
import torch.nn as nn

from .config import CPromptTuningActivation, CPromptTuningConfig
from peft.tuners.tuners_utils import BaseEmbedding


class CPromptEmbedding(BaseEmbedding):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`CPromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import CPromptEmbedding, CPromptTuningConfig

    >>> config = CPromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     init_type="TEXT",
    ...     init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> cprompt_embedding = CPromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `out_embeddings`, `token_dim`)
    """
    def __init__(self, config, word_embeddings):
        super().__init__(config, word_embeddings)
        
        out_virtual_tokens = config.output_embeddings * config.num_transformer_submodules
        
        if config.inference_mode:
            self.reduced_embedding = torch.nn.Embedding(out_virtual_tokens, self.token_dim)
        
        self.token_mask = torch.ones(out_virtual_tokens)
        self.out_virtual_tokens = out_virtual_tokens
        self.config = config
        
        if not config.inference_mode:
            
            in_channels = self.total_virtual_tokens
            conv_layers = []
            
            if config.conv_out_channels is not None:
                
                for n in range(len(config.conv_out_channels)):
                    kernel_size = config.conv_kernel_sizes[n]
                    out_channel = config.conv_out_channels[n] * config.num_transformer_submodules
                    
                    if kernel_size % 2 == 0:
                        raise ValueError("kernel size must be odd to keep the embedding token dimension.")

                    conv_layers.append(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2,
                            bias=config.conv_bias,
                        )
                    )
                    
                    in_channels = out_channel
                    
                    if config.conv_pool:
                        conv_layers.append(
                            nn.MaxPool1d(
                                kernel_size=kernel_size,
                                stride=1,
                                padding=kernel_size // 2
                            )
                        )
            
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=out_virtual_tokens,
                    kernel_size=1,
                    stride=1,
                    bias=config.conv_bias,
                )
            )
            
            module = [nn.Linear(self.token_dim, config.encoder_bottleneck)]
            
            if config.encoder_nonlinearity == CPromptTuningActivation.RELU:
                module.append(nn.ReLU())
            elif config.encoder_nonlinearity == CPromptTuningActivation.TANH:
                module.append(nn.Tanh())
            elif config.encoder_nonlinearity == CPromptTuningActivation.SIGM:
                module.append(nn.Sigmoid())
            
            module.append(nn.Linear(config.encoder_bottleneck, self.token_dim))
            
            if config.encoder_dropout > 0:
                module.append(nn.Dropout(p=config.dropout))
            if config.encoder_layer_norm:
                module.append(nn.LayerNorm(self.token_dim))
            
            num_modules = config.encoder_num_modules
            if num_modules > 2:
                encoder_num_modules_default = CPromptTuningConfig.encoder_num_modules
                warnings.warn(
                    f"Since only MLP 1 and 2 layers were used in Residual Prompt Tuning (Zhengxiang et al.),"
                    f"for MLP, the argument `encoder_num_layers` is ignored."
                    f"Exactly {encoder_num_modules_default} MLP layers are used"
                )
                num_modules = encoder_num_modules_default
            if num_modules == 2:
                module = module + module
            
            self.module = nn.Sequential(*module)
            self.conv_layers = nn.Sequential(*conv_layers)
    
    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        prompt_embeddings = self.conv_layers(prompt_embeddings)
        if self.config.encoder_residual:
            return self.module(prompt_embeddings) + prompt_embeddings
        else:
            return self.module(prompt_embeddings)

    def create_reduced_embedding(self):
        self.reduced_embedding = torch.nn.Embedding(self.out_virtual_tokens, self.token_dim)
