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

from .config import (
    CPromptTuningActivation, 
    CPromptTuningConfig,  
    CPromptTuningMixture,
    CPromptTuningConvolutionType, 
    CPromptTuningReparameterizationType,
)
from peft.tuners.tuners_utils import BaseEmbedding


class Bottleneck(nn.Module):
    """
    Bottleneck class for convolutional prompt. Bottleneck can have a residual connection.
    
    args:
        in_channels (`int`)
        out_channels (`int`)
        config ([`CPromptTuningConfig`]): The configuration of the convolutional prompt embedding.
    """
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=config.conv_bias,
        )
        self.norm1 = nn.LayerNorm(config.token_dim)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=config.conv_bias,
        )
        self.norm2 = nn.LayerNorm(config.token_dim)
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=config.conv_bias,
        )
        self.norm3 = nn.LayerNorm(config.token_dim)
        self.relu = nn.ReLU()
        self.config = config
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        if self.config.conv_residual:
            out += identity
        out = self.relu(out)
        
        return out


class ConvolutionLayer(nn.Module):
    """
    Convolution layer class for prompt tuning.
    
    args:
        config ([`CPromptTuningConfig`]): The configuration of the convolutional prompt tuning 
    """
    def __init__(self, in_channels, config):
        super().__init__()
        
        self.config = config
        
        if config.conv_type == CPromptTuningConvolutionType.CONN:
            with open("./peft/tuners/cprompt_tuning/top_words.txt", "r") as f:
                from transformers import AutoTokenizer
                
                tokenizer_kwargs = config.tokenizer_kwargs or {}
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
                top_words = f.read()
                top_token_ids = tokenizer(top_words, add_special_tokens=False)["input_ids"]
                num_text_tokens = len(top_token_ids)
                if num_text_tokens > self.total_virtual_tokens:
                    top_token_ids = top_token_ids[:self.total_virtual_tokens]
                elif num_text_tokens < self.total_virtual_tokens:
                    num_reps = math.ceil(self.total_virtual_tokens / num_text_tokens)
                    top_token_ids = top_token_ids * num_reps
                top_token_ids = top_token_ids[:self.total_virtual_tokens]
                top_token_ids = torch.LongTensor(top_token_ids).to(word_embeddings.weight.device)
                top_embedding_weights = word_embeddings(top_token_ids).detach().clone()
                top_embedding_weights = top_embedding_weights.to(torch.float32)
                self.top_embedding_weights = top_embedding_weights.cuda()
        
        if config.conv_type == CPromptTuningConvolutionType.CONN:
            conv_dict = nn.ModuleDict()
            orders = []
        elif config.conv_type == CPromptTuningConvolutionType.DEFAULT:
            conv_layers = []
        
        for n in range(len(config.conv_out_channels)):
            kernel_size = config.conv_kernel_sizes[n]
            out_channels = config.conv_out_channels[n] * config.num_transformer_submodules
            
            if isinstance(kernel_size, str) and kernel_size != "bottleneck":
                raise ValueError("kernel must be bottleneck")
            elif isinstance(kernel_size, int) and kernel_size % 2 == 0:
                raise ValueError("kernel size must be odd to keep the embedding token dimension.")
            
            if kernel_size == "bottleneck":
                bottleneck = Bottleneck(in_channels, out_channels, config)
                conv_layers.append(bottleneck)
            else:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=config.conv_bias,
                    )
                )
            
                if config.conv_dropout > 0:
                    conv_layers.append(nn.Dropout(p=config.conv_dropout))
                if config.conv_layer_norm:
                    conv_layers.append(nn.LayerNorm(config.token_dim))
                if config.conv_nonlinearity == CPromptTuningActivation.RELU:
                    conv_layers.append(nn.ReLU())
                elif config.conv_nonlinearity == CPromptTuningActivation.TANH:
                    conv_layers.append(nn.Tanh())
                elif config.conv_nonlinearity == CPromptTuningActivation.SIGM:
                    conv_layers.append(nn.Sigmoid())
            
                in_channels = out_channels
            
                if config.conv_pool:
                    conv_layers.append(
                        nn.MaxPool1d(
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2
                        )
                    )
        
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, prompt_embeddings):
        # just get convolutional embeddings
        prompt_embeddings = self.conv_layers(prompt_embeddings)
        return prompt_embeddings


class PTuningEncoder(nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.
    """
    def __init__(self, config):
        super().__init__()
        
        input_size = config.token_dim
        output_size = config.token_dim
        hidden_size = config.encoder_hidden_size
        self.encoder_type = config.encoder_reparameterization_type
        
        if self.encoder_type == CPromptTuningReparameterizationType.LSTM:
            lstm_dropout = config.encoder_dropout
            num_layers = config.encoder_num_layers
            # LSTM
            self.lstm_head = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                bidirectional=True,
                batch_first=True,
            )
            self.mlp_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size * 2, output_size)
            )
        elif self.encoder_type == CPromptTuningReparameterizationType.MLP:
            encoder_num_layers_default = CPromptTuningConfig.encoder_num_layers
            if config.encoder_num_layers != encoder_num_layers_default:
                warnings.warn(
                    f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "
                    f"Exactly {encoder_num_layers_default} MLP layers are used."
                )
            layers = [
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            ]
            self.mlp_head = torch.nn.Sequential(*layers)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    def forward(self, prompt_embeddings):
        if self.encoder_type == CPromptTuningReparameterizationType.LSTM:
            prompt_embeddings = self.mlp_head(self.lstm_head(prompt_embeddings)[0])
        elif self.encoder_type == CPromptTuningReparameterizationType.MLP:
            prompt_embeddings = self.mlp_head(prompt_embeddings)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")
        return prompt_embeddings


class ResidualMLP(torch.nn.Module):
    """
    MLP class for prompt re-parameterization. MLP can have a residual connection.
    
    args:
        config ([`ResidualPromptTuningConfig`]): The configuration of the residual prompt embedding.
    """
    def __init__(self, config):
        super().__init__()
        token_dim = config.token_dim
        input_size = token_dim
        output_size = token_dim
        bottleneck_size = config.encoder_bottleneck_size
        num_layers = config.encoder_num_layers
        self.encoder_type = config.encoder_reparameterization_type
        self.config = config
        
        if self.encoder_type == CPromptTuningReparameterizationType.LSTM:
            self.lstm_head = torch.nn.LSTM(
                input_size=token_dim,
                hidden_size=token_dim // 2,
                num_layers=config.encoder_dropout,
                dropout=config.encoder_dropout,
                bidirectional=True,
                batch_first=True
            )
            self.mlp_head = nn.Sequential(
                torch.nn.Linear(token_dim, token_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(token_dim, token_dim)
            )
        elif self.encoder_type == CPromptTuningReparameterizationType.MLP:
            layers = [
                torch.nn.Linear(input_size, bottleneck_size),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_size, output_size)
            ]
            if config.encoder_dropout > 0:
                layers.append(torch.nn.Dropout(p=config.encoder_dropout))
            if config.encoder_layer_norm:
                layers.append(torch.nn.LayerNorm(output_size))
            if num_layers > 2:
                encoder_num_layers_default = CPromptTuningConfig.encoder_num_layers
                warnings.warn(
                    f"Since only MLP 1 and 2 layers were used in Residual Prompt Tuning (Zhengxiang et al.),"
                    f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored."
                    f"Exactly {encoder_num_layers_default} MLP layers are used"
                )
                num_layers = encoder_num_layers_default
            if num_layers == 2:
                layers = layers + layers
            
            self.encoder = torch.nn.Sequential(*layers)
        elif self.encoder_type == CPromptTuningReparameterizationType.TRANSFORMER:
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=token_dim, nhead=2, dropout=config.encoder_dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM")

    def forward(self, prompt_embeddings):
        identity = prompt_embeddings
        
        if self.encoder_type == CPromptTuningReparameterizationType.LSTM:
            prompt_embeddings = self.mlp_head(self.lstm_head(prompt_embeddings)[0])
        else:
            prompt_embeddings = self.encoder(prompt_embeddings)
        
        if self.config.encoder_residual:
            return prompt_embeddings + identity
        else:
            return prompt_embeddings


class ResidualPromptTuningEncoder(nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings with residual connections.
    """
    def __init__(self, config, out_virtual_tokens):
        super().__init__()
        
        self.separate = config.encoder_separate
        self.out_virtual_tokens = out_virtual_tokens
        
        if self.separate:
            self.mlp = torch.nn.ModuleDict()
            for i in range(self.out_virtual_tokens):
                self.mlp[str(i)] = ResidualMLP(config)
        else:
            self.mlp = ResidualMLP(config)
    
    def forward(self, input_embeds):
        if self.separate:
            prompt_embeddings = []
            for i in range(self.out_virtual_tokens):
                embeds = self.mlp[str(i)](input_embeds[i:i+1])
                prompt_embeddings.append(embeds)
            prompt_embeddings = torch.concat(prompt_embeddings)
        else:
            prompt_embeddings = self.mlp(input_embeds)
        
        return prompt_embeddings


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

    Output Shape: (`batch_size`, `output_embeddings`, `token_dim`)
    """
    def __init__(self, config, word_embeddings):
        super().__init__(config, word_embeddings)
        
        out_virtual_tokens = config.output_embeddings * config.num_transformer_submodules
        
        if config.inference_mode:
            self.reduced_embedding = torch.nn.Embedding(out_virtual_tokens, self.token_dim)
        
        self.token_mask = torch.ones(config.output_embeddings)
        self.out_virtual_tokens = out_virtual_tokens
        self.config = config
        
        # for convolution fixed Reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if not config.inference_mode:
            
            in_channels = self.total_virtual_tokens
            self.conv_layers = ConvolutionLayer(in_channels, config)
            
            if config.prompt_tuning_type == CPromptTuningMixture.P_TUNING:
                self.module = PTuningEncoder(config)
            elif config.prompt_tuning_type == CPromptTuningMixture.RESIDUAL_PROMPT_TUNING:
                self.module = ResidualPromptTuningEncoder(config, out_virtual_tokens)
    
    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        prompt_embeddings = self.conv_layers(prompt_embeddings)
        
        if self.config.prompt_tuning_type in [
            CPromptTuningMixture.P_TUNING,
            CPromptTuningMixture.RESIDUAL_PROMPT_TUNING
        ]:
            prompt_embeddings = self.module(prompt_embeddings)
        
        return prompt_embeddings

    def create_reduced_embedding(self):
        self.reduced_embedding = torch.nn.Embedding(self.out_virtual_tokens, self.token_dim)
