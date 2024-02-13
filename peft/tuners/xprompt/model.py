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
import logging

from utils.general import colorstr, colorformat, emojis

from .config import XPromptTuningInit


logger = logging.getLogger(__name__)


class XPromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings that can prune the negative tokens
    Implements xprompts as described in https://aclanthology.org/2022.emnlp-main.758.pdf

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

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
    token_prefix : str = "token-level"
    piece_prefix : str = "piece-level"
    
    def __init__(self, config, word_embeddings):
        super().__init__()
        
        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        self.token_mask = torch.ones(total_virtual_tokens)
        self.piece_mask = torch.ones(total_virtual_tokens, config.token_dim)
        if config.token_dim % config.token_pieces > 0:
            raise ValueError("The number of token_pieces for the token dimension does not perfectly divide.")
        
        self.to_prune = {
            self.token_prefix: set(), 
            self.piece_prefix: {}
        }
        self.kept_prune = {
            self.token_prefix: set(range(total_virtual_tokens)), 
            self.piece_prefix: {
                f"{self.token_prefix}:{token}": set(range(config.token_pieces)) 
                for token in range(total_virtual_tokens)
            }
        }
        
        
        if config.xprompt_tuning_init == XPromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer_kwargs = config.tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.xprompt_tuning_init_text
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
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
        
        self.config = config
        self.total_virtual_tokens = total_virtual_tokens
    
    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices) * self.piece_mask.to(indices.device)
        return prompt_embeddings

    def batch_token_importance(self):
        embed = self.embedding.weight
        d_embed = embed.grad
        mask = self.token_mask.to(embed.device)
        
        if d_embed is None:
            raise ValueError("Only embedding with gradient can be calculated of importance")
        
        # Take the absolute dot
        importance = torch.einsum(
            "ew,ew->e",
            [embed, d_embed],
        )
        
        importance *= mask
        importance = importance.abs().detach()
        
        return importance, mask
    
    def batch_piece_importance(self):
        split_size = self.config.token_dim // self.config.token_pieces
        split_embed = torch.split(self.embedding.weight, split_size, dim=1)
        split_d_embed = torch.split(self.embedding.weight.grad, split_size, dim=1)
        mask = self.piece_mask[:, torch.arange(self.config.token_dim) % split_size == 0].to(
            self.embedding.weight.device
        )
        
        importance = torch.stack(
            [torch.einsum("li,li->l", piece, grad) 
             for piece, grad in zip(split_embed, split_d_embed)], dim=1
        )
        
        importance *= mask
        importance = importance.abs().detach()
        
        return importance, mask

    def estimate_token_importance(self, trainer, current_step):
        """Train the model for one epoch to prune the negative token"""
        
        # only prune the token in target step.
        if current_step != self.config.prune_step:
            return
        
        logger.info(f"*** {colorstr('cyan', 'bold', 'Pruning Tokens')} ***")
        
        model = trainer.model
        is_training = model.training
        device = self.embedding.weight.device
        
        model.eval()
        trainer.optimizer.zero_grad()
        
        embed_importance = torch.zeros(self.total_virtual_tokens).to(device)
        denoms = embed_importance.clone()
        
        for idx, data in enumerate(trainer.get_train_dataloader()):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]
            
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            self.embedding.weight.retain_grad()
            output.loss.backward()
            
            importance, denom = self.batch_token_importance() 
            
            embed_importance += importance
            denoms += denom
        
        if is_training:
            model.train()
        
        # token Normalize
        embed_importance /= denoms
        profile = self.token_profile(embed_importance)
        
        # prune the tokens
        for desc, _ in profile:
            idx = int(desc.split(":")[1])
            self.to_prune[self.token_prefix].add(idx)
            self.kept_prune[self.token_prefix].remove(idx)
            self.to_prune[self.piece_prefix][desc] = self.kept_prune[self.piece_prefix].pop(desc)
            self.token_mask[idx] = 0
            self.piece_mask[idx] = 0
        
        logger.info(f"*** {colorstr('cyan', 'bold', 'token importance scores')} to be removed ***\n"
                    f"{list(f'{token[1].item():.5f}' for token in profile)}\n"
                    f"{list(token for token, _ in profile)}\n")
        logger.info(f"*** {colorstr('cyan', 'bold', 'pruned')} ***\n{self.to_prune[self.token_prefix]}")
        logger.info(f"*** {colorstr('cyan', 'bold', 'kept')} ***\n{self.kept_prune[self.token_prefix]}\n")
        logger.info(f"*** {colorstr('cyan', 'bold', 'token mask')} ***\n{self.token_mask}")
    
    def estimate_piece_importance(self, trainer, current_step):
        """Train the model for one epoch to prune the negative token"""
        
        # only prune the token in target step.
        if current_step != self.config.prune_step:
            return
        
        logger.info(f"*** {colorstr('cyan', 'bold', 'Pruning Pieces')} ***")
        
        model = trainer.model
        is_training = model.training
        device = self.embedding.weight.device
        
        model.eval()
        trainer.optimizer.zero_grad()
        
        embed_importance = torch.zeros(self.total_virtual_tokens, self.config.token_pieces).to(device)
        denoms = embed_importance.clone()
        
        for idx, data in enumerate(trainer.get_train_dataloader()):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]
            
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            self.embedding.weight.retain_grad()
            output.loss.backward()
            
            importance, denom = self.batch_piece_importance() 
            
            embed_importance += importance
            denoms += denom
        
        if is_training:
            model.train()
        
        # piece Normalize
        embed_importance /= denoms
        profile = self.piece_profile(embed_importance)
        
        # prune the pieces
        split_size = self.config.token_dim // self.config.token_pieces
        for token, _ in profile.items():
            if self.to_prune[self.piece_prefix].get(token) is None: self.to_prune[self.piece_prefix][token] = set()
            for desc, _ in profile[token]:
                idx = int(desc.split(":")[1])
                self.to_prune[self.piece_prefix][token].add(idx)
                self.kept_prune[self.piece_prefix][token].remove(idx)
                self.piece_mask[int(token.split(":")[1])][idx*split_size:(idx+1)*split_size] = 0
        
        logger.info(f"*** {colorstr('cyan', 'bold', 'piece importance scores')} to be removed ***\n")
        logger.info(f"*** {colorstr('cyan', 'bold', 'pruned')} ***\n{self.to_prune[self.piece_prefix]}")
        logger.info(f"*** {colorstr('cyan', 'bold', 'kept')} ***\n{self.kept_prune[self.piece_prefix]}\n")
    
    def token_profile(self, importance):
        target_token_length = int(self.total_virtual_tokens * self.config.token_ratio)
        
        # create profile
        profile = {
            f"{self.token_prefix}:{token}": importance[token]
            for token in range(len(importance))
        }
        # select kept tokens
        profile = {k: v for k, v in profile.items() if int(k.split(":")[1]) in self.kept_prune[self.token_prefix]}
        # sort tokens
        profile = sorted(profile.items(), key=lambda x: x[1])[:target_token_length]
        
        return profile

    def piece_profile(self, importance):
        target_piece_length = int(self.config.token_pieces * self.config.piece_ratio)
        
        # create profile
        profile = {
            f"{self.token_prefix}:{token}": {
                f"{self.piece_prefix}:{piece}": importance[token][piece]
                for piece in range(len(importance[token]))
            }
            for token in range(len(importance))
        }
        # select kept tokens
        profile = {k: v for k, v in profile.items() if int (k.split(":")[1]) in self.kept_prune[self.token_prefix]}
        for k, v in profile.items():
            profile[k] = sorted(profile[k].items(), key=lambda x: x[1])[:target_piece_length]
        
        return profile
