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
# Based on https://github.com/arazd/ResidualPrompts
import warnings

import torch

from .config import (
    ResidualPromptTuningConfig,
    ResidualPromptTuningReparameterizationType
)
from peft.tuners.tuners_utils import BaseEmbedding


class ResidualMLP(torch.nn.Module):
    """
    MLP class for prompt re-parameterization. MLP can have a residual connection.
    
    args:
        config ([`ResidualPromptTuningConfig`]): The configuration of the residual prompt embedding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.bottleneck_size = config.encoder_bottleneck_size
        self.encoder_type = config.encoder_reparameterization_type
        self.num_layers = config.encoder_num_layers
        self.dropout = config.encoder_dropout
        self.residual = config.residual
        
        if self.encoder_type == ResidualPromptTuningReparameterizationType.MLP:
            layers = [
                torch.nn.Linear(self.input_size, self.bottleneck_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.bottleneck_size, self.output_size),
            ]
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(p=config.encoder_dropout))
            if config.encoder_layer_norm:
                layers.append(torch.nn.LayerNorm(self.output_size))
            if self.num_layers > 2:
                encoder_num_layers_default = ResidualPromptTuningConfig.encoder_num_layers
                warnings.warn(
                    f"Since only MLP 1 and 2 layers were used in Residual Prompt Tuning (Zhengxiang et al.),"
                    f"for {self.encoder_reparameterization_type.value}, the argument `encoder_num_layers` is ignored."
                    f"Exactly {encoder_num_layers_default} MLP layers are used"
                )
                self.num_layers = encoder_num_layers_default
            if self.num_layers == 2:
                layers = layers + layers
            
            self.encoder = torch.nn.Sequential(*layers)
        elif self.encoder_type == ResidualPromptTuningReparameterizationType.LSTM:
            self.lstm_head = torch.nn.LSTM(
                input_size=self.token_dim,
                hidden_size=self.token_dim // 2,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=True,
                batch_first=True
            )
            self.mlp_head = nn.Sequential(
                torch.nn.Linear(self.token_dim, self.token_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.token_dim, self.token_dim)
            )
        elif self.encoder_type == ResidualPromptTuningReparameterizationType.TRANSFORMER:
            self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=2, dropout=self.dropout)
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM")

    def forward(self, input_embeds):
        if self.encoder_type == ResidualPromptTuningReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        else:
            output_embeds = self.encoder(input_embeds)
        
        if self.residual:
            return output_embeds + input_embeds
        else:
            return output_embeds


class ResidualPromptEmbedding(BaseEmbedding):
    """
    The model to encode virtual tokens into prompt embeddings with residual connections.
    Implements residual prompt-tuning as described in https://arxiv.org/pdf/2305.03937v1.pdf
    
    Args:
        config ([`ResidualPromptTuningConfig`]): The configuration of the residual prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.
    
    **Attributes**:
        - **embedding** (`torch.nn.Module`) -- The embedding layer of the prompt embedding.
    
    Example:
    
    '''py
    >>> from peft import ResidualPromptEmbedding, ResidualPromptTuningConfig
    
    >>> config = ResidualPromptTuningConfig(
    ...     peft_type="RESIDUAL_PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     init_type="TEXT",
    ...     init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ...     encoder_reparameterization_type="MLP",
    ...     encoder_bottleneck_size=800,
    ...     encoder_hidden_size=768,
    ...     encoder_num_layers=1,
    ...     encoder_dropout=0.1,
    ...     residual=True,
    ...     layer_norm=True,
    ... )
    
    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = ResidualPromptEmbedding(config, t5_model.shared)
    '''
    
    Input Shape: (`batch_size`, `total_virtual_tokens`)
    
    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """
    def __init__(self, config, word_embeddings):
        super().__init__(config, word_embeddings)
        
        self.separate = config.encoder_separate
        
        if not config.inference_mode:
            if self.separate:
                self.mlp = {}
                self.mlp = torch.nn.ModuleDict()
                for i in range(self.total_virtual_tokens):
                    self.mlp[str(i)] = ResidualMLP(config)
            else:
                self.mlp = ResidualMLP(config)
    
    def forward(self, indices):
        # Just get embeddings
        input_embeds = self.embedding(indices)

        if self.separate:
            output_embeds = []
            for i in range(self.total_virtual_tokens):
                embeds = self.mlp[str(i)](input_embeds[i:i+1])
                output_embeds.append(embeds)
            output_embeds = torch.concat(output_embeds)
        else:
            output_embeds = self.mlp(input_embeds)
        
        return output_embeds
