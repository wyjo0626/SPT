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
import torch
import torch.nn as nn

from .config import EPromptTuningConfig, EPTReparameterizationType, EPTActivationType
from peft.tuners.tuners_utils import BaseEmbedding


class EPTEmbedding(BaseEmbedding):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for ept.

    Args:
        config ([`EPromptTuningConfig`]): The configuration of the prompt encoder.

    Example:

    ```py
    >>> from peft import EPromptTuningConfig, EPTEmbedding

    >>> config = EPromptTuningConfig(
    ...     peft_type="EPT",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=1,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     ept_reparameterization_type="MLP",
    ...     ept_hidden_size=1,
    ... )

    >>> ept_encoder = EPTEmbedding(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt encoder.
        - **mlp_head** (`torch.nn.Sequential`) -- The MLP head of the prompt encoder if `inference_mode=False`.
        - **lstm_head** (`torch.nn.LSTM`) -- The LSTM head of the prompt encoder if `inference_mode=False` and
        `encoder_reparameterization_type="LSTM"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
        prompt encoder.
        - **encoder_type** (Union[[`PromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt
          encoder.


    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """
    def __init__(self, config, word_embeddings):
        super().__init__(config, word_embeddings)
        
        self.encoder_type = config.ept_reparameterization_type
        nonlinear_type = config.ept_nonlinearity
        
        hidden_size = config.ept_hidden_size
        num_layers = config.ept_num_layers
        dropout = config.ept_dropout
        layer_norm = config.ept_layer_norm
        bias = config.ept_bias
        self.residual = config.ept_residual
        
        if not config.inference_mode:
            
            if nonlinear_type == EPTActivationType.RELU:
                nonlinear = nn.ReLU()
            elif nonlinear_type == EPTActivationType.TANH:
                nonlinear = nn.Tanh()
            elif nonlinear_type == EPTActivationType.SIGM:
                nonlinear = nn.Sigmoid()
            else:
                nonlinear = None
            
            self.encoder = []
            
            if self.encoder_type == EPTReparameterizationType.LSTM:
                # LSTM
                self.lstm_head = nn.LSTM(
                    input_size=self.token_dim,
                    hidden_size=self.token_dim // 2,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True,
                    bias=bias
                )
                
                self.encoder.append(nn.Linear(self.token_dim, hidden_size, bias=bias))
                if nonlinear: self.encoder.append(nonlinear)
                self.encoder.append(nn.Linear(hidden_size, self.token_dim, bias=bias))
            elif self.encoder_type == EPTReparameterizationType.TRANSFORMER:
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=2, dropout=dropout, activation=nonlinear)
                self.encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=num_layers))
            elif self.encoder_type == EPTReparameterizationType.MLP:
                self.encoder.append(nn.Linear(self.token_dim, hidden_size, bias=bias))
                if nonlinear: self.encoder.append(nonlinear)
                self.encoder.append(nn.Linear(hidden_size, self.token_dim, bias=bias))
                
                if dropout > 0: self.encoder.append(nn.Dropout(p=dropout))
                if layer_norm: self.encoder.append(nn.LayerNorm(self.token_dim))
                if num_layers > 2:
                    tmp = self.encoder
                    for i in range(num_layers):
                        self.encoder = self.encoder + tmp
            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM")
        
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        
        if self.encoder_type == EPTReparameterizationType.LSTM:
            output_embeds = self.encoder(self.lstm_head(input_embeds)[0])
        else:
            output_embeds = self.encoder(input_embeds)
        
        if self.residual:
            return output_embeds + input_embeds
        else:
            return output_embeds
