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

from .config import CPromptTuningInit, CPromptTuningActivation


class CPromptEmbedding(nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """
    def __init__(self, config, word_embeddings):
        super().__init__()
        
        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = nn.Embedding(total_virtual_tokens, config.token_dim)
        self.token_mask = torch.ones(config.output_embeddings)
        if config.prompt_tuning_init == CPromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer_kwargs = config.tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = nn.Parameter(word_embedding_weights)
        
        self.conv = nn.Conv1d(
            in_channels=total_virtual_tokens, 
            out_channels=config.output_embeddings,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        
        layers = [nn.Linear(config.token_dim, config.bottleneck)]
        
        if config.nonlinearity == CPromptTuningActivation.RELU:
            layers.append(nn.ReLU())
        elif config.nonlinearity == CPromptTuningActivation.TANH:
            layers.append(nn.Tanh())
        elif config.nonlinearity == CPromptTuningActivation.SIGM:
            layers.append(nn.Sigmoid())
        
        layers.append(nn.Linear(config.bottleneck, config.token_dim))
        
        if config.dropout > 0:
            layers.append(nn.Dropout(p=config.dropout))
        if config.layer_norm:
            layers.append(nn.LayerNorm(config.token_dim))
        
        self.module = nn.Sequential(*layers)
    
    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.conv(self.embedding(indices))
        prompt_embeddings = self.module(prompt_embeddings) + prompt_embeddings
        return prompt_embeddings
