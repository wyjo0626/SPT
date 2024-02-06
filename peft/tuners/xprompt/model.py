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
        
        self.config = config
        
        self._crnt_virtual_tokens = config.num_virtual_tokens
        self.embedding = torch.nn.Embedding(self.orig_total_virtual_tokens, config.token_dim)
        self.token_mask = torch.ones(self.orig_total_virtual_tokens)
        # self.piece_mask = torch.nn.ones()
        
        self.to_prune = {
            self.token_prefix: set(), 
            # "piece-level": set(),
            }
        self.kept_prune = {
            self.token_prefix: set(range(self.orig_total_virtual_tokens)), 
            # "piece-level": set(),
        }
        
        # prune steps and rewinding step.
        self.total_steps = config.prune_steps + 1
        self.crnt_prune_step = 0
        
        if config.xprompt_tuning_init == XPromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer_kwargs = config.tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.xprompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > self.orig_total_virtual_tokens:
                init_token_ids = init_token_ids[:self.orig_total_virtual_tokens]
            elif num_text_tokens < self.orig_total_virtual_tokens:
                num_reps = math.ceil(self.orig_total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:self.orig_total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
    
    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)[:, list(self.kept_prune[self.token_prefix]), :]
        return prompt_embeddings

    def batch_token_importance(self):
        embed = self.embedding.weight
        d_embed = embed.grad
        mask = self.token_mask.to(embed.device)
        
        if d_embed is None:
            raise ValueError("Only embedding with gradient can be calculated of importance")
        
        # Take the absolute dot
        importance = torch.einsum(
            "lv,lv->l",
            [embed, d_embed],
        )
        
        importance *= mask
        importance = importance.abs().detach()
        
        return importance, mask

    def estimate_token_importance(self, trainer, total_steps, crnt_steps):
        """Train the model for one epoch to prune the negative token"""
        prune_steps = total_steps // self.total_steps
        self.crnt_prune_step = crnt_steps // prune_steps
        # only prune the token in prune steps, not 0 and total steps (rewinding steps)
        if (crnt_steps % prune_steps != 0 or 
            self.crnt_prune_step == 0 or
            self.crnt_prune_step == self.total_steps):
            return
        
        logger.info(f"*** {colorstr('cyan', 'bold', 'Pruning Tokens')} ***")
        
        model = trainer.model
        device = self.embedding.weight.device
        is_training = model.training
        
        model.eval()
        trainer.optimizer.zero_grad()
        
        # piece_mask = torch.ones(total_virtual_tokens).to(device)
        embed_importance = {
            self.token_prefix: torch.zeros(self.orig_total_virtual_tokens).to(device)
            # "piece-level": torch.zeros(total_virtual_tokens).to(device)
        }
        denoms = {attn_type: val.clone() for attn_type, val in embed_importance.items()}
        profile = {}
        
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
            
            embed_importance[self.token_prefix] += importance
            denoms[self.token_prefix] += denom
        
        if is_training:
            model.train()
        
        # Normalize
        embed_importance[self.token_prefix] /= denoms[self.token_prefix]
        profile[self.token_prefix] = self.get_profile(embed_importance[self.token_prefix], self.token_prefix)
        
        # prune the tokens
        for desc in profile[self.token_prefix]:
            idx = int(desc[0].split(":")[1])
            self.to_prune[self.token_prefix].add(idx)
            self.kept_prune[self.token_prefix].remove(idx)
            self.token_mask[idx] = 0
        
        logger.info(f"*** {colorstr('cyan', 'bold', 'token importance scores')} to be removed ***\n"
                    f"{list(f'{token[1].item():.5f}' for token in profile[self.token_prefix])}\n"
                    f"{list(token for token, _ in profile[self.token_prefix])}\n")
        logger.info(f"*** {colorstr('cyan', 'bold', 'pruned')} ***\n{self.to_prune}")
        logger.info(f"*** {colorstr('cyan', 'bold', 'kept')} ***\n{self.kept_prune}\n")
        
        self._crnt_virtual_tokens = int(self._crnt_virtual_tokens * self.config.token_ratio)
    
    def get_profile(self, importance, prefix):
        shp = importance.shape
        
        # target token length
        target_token_length = int(self.crnt_total_virtual_tokens * self.config.token_ratio)
        
        if len(shp) == 1:
            # token profile
            num_token = shp[0]
        
            # create profile
            profile = {
                f"{prefix}:{token}": importance[token]
                for token in range(num_token)
            }
            # select kept tokens
            profile = {k: v for k ,v in profile.items() if int(k.split(":")[1]) in self.kept_prune[self.token_prefix]}
            # sort tokens
            profile = sorted(profile.items(), key=lambda x: x[1])[:target_token_length]
        else:
            # piece profile
            None
        
        return profile
    crnt
    @property
    def crnt_total_virtual_tokens(self):
        # get (ininitalized or pruned) virtual token length
        return self._crnt_virtual_tokens * self.config.num_transformer_submodules

    @property
    def crnt_virtual_tokens(self):
        # get current virtual token length
        return self._crnt_virtual_tokens

    @property
    def orig_total_virtual_tokens(self):
        # get original virtual token length
        return self.confiug.num_virtual_tokens * self.config.num_transformer_submodules
