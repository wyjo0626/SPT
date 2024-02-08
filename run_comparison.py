import os

from models.utils import get_trainer
from utils.arguments import ModelArguments, DataTrainingArguments, DynamicTrainingArguments, DynamicPeftArguments
# from uitls.general import colorstr, colorformat, emojis
from tasks.utils import *

import datasets
import numpy as np
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments, set_seed

from dataclasses import dataclass, field
from typing import Optional, Union, List


os.environ["WANDB_PROJECT"] = "peft"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class PruneArguments:
    prune_type: Optional[str] = field(
        default=None,
    ),
    prune_ratio: Optional[float] = field(
        default=0.5
    )


def importance_score(embed, mask):
    d_embed = embed.grad
    # Take the absolute dot
    importance = torch.einsum(
        "li,li->l",
        [embed, d_embed],
    )
    importance *= mask
    importance = importance.abs().detach()
    
    return importance, mask

def get_profile(importances, prefix):
    n_tokens = importances.size(0)
    return {
        f"{prefix}:{token}": importances[token]
        for token in range(n_tokens)
    }

def parse_pruning_descriptors(descriptors):
    to_prune = {
        "token": set(),
    }
    
    for descriptor in descriptors:
        embed_type, idx = descriptor.split(":")
        idx = int(idx)
        to_prune[embed_type].add(idx)
    
    return to_prune

def mask_heads(mask, to_prune):
    for embed_type in to_prune:
        for idx in to_prune[embed_type]:
            mask[idx] = 0


def preprocess(model, data, prune_type=None):
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    labels = data["labels"]
    
    batch_size = len(input_ids)
    num_virtual_tokens = model.active_peft_config.num_virtual_tokens
    
    prompt_encoder = model.prompt_encoder[model.active_adapter]
    prompt_tokens = (
        model.prompt_tokens[model.active_adapter]
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(prompt_encoder.embedding.weight.device)
    )
    
    inputs_embeds = model.word_embeddings(input_ids)
    if prune_type is None:
        prompts = prompt_encoder(prompt_tokens).to(inputs_embeds.dtype)
        prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens).to(attention_mask.device)
    elif prune_type == "xprompt-tuning":
        prompts = prompt_encoder(prompt_tokens).to(inputs_embeds.dtype)
        prefix_attention_mask = model.embed_mask.expand(batch_size, num_virtual_tokens).to(attention_mask.device)
    elif prune_type == "rprompt-tuning":
        prompts = prompt_encoder(prompt_tokens)[:, list(model.kept_prune)].to(attention_mask.device)
        prefix_attention_mask = torch.ones(batch_size, int(num_virtual_tokens * round(1.0 - model.prune_ratio, 5))).to(attention_mask.device)
    
    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
    inputs_embeds  = torch.cat((prompts, inputs_embeds), dim=1)

    return inputs_embeds, attention_mask, labels


def evaluate(model, optimizer, trainer, prune_type=None):
    model.eval()
    optimizer.zero_grad()
    
    acc = torch.tensor([])
    loss = torch.tensor([])
    
    for idx, data in enumerate(trainer.get_eval_dataloader()):
        inputs_embeds, attention_mask, labels = preprocess(model, data, prune_type)
        
        output = model.base_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        preds = output.logits.detach().cpu()
        
        acc = torch.cat((acc, 100. * (np.argmax(preds, axis=1) == labels.cpu())), dim=-1)
        loss = torch.cat((loss, torch.tensor([output.loss.detach().cpu()])), dim=-1)
    
    return loss.mean().item(), acc.mean().item()


def train(model, trainer, optimizer, training_args, prune_type=None):
    model = model.cuda()
    total_epochs = int(training_args.num_train_epochs)
    
    print(f"{total_epochs} epochs start")
    for i in range(total_epochs):
        model.train()
        
        for idx, data in enumerate(trainer.get_train_dataloader()):
            inputs_embeds, attention_mask, labels = preprocess(model, data, prune_type)
            
            output = model.base_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            
        loss, acc = evaluate(model, optimizer, trainer, prune_type)
        print(f"{i+1} epochs - loss : {loss}   acc : {acc}")


def estimate_token_importance(model, trainer, optimizer, prune_type=None):
    learned_embedding = model.prompt_encoder[model.active_adapter].embedding
    device = learned_embedding.weight.device
    orig_embed_mask = torch.ones(learned_embedding.weight.shape[0]).to(device)
    
    embed_importance = {
        "token-level": torch.zeros(learned_embedding.weight.shape[0]).to(device),
        "piece-level": torch.zeros(learned_embedding.weight.shape).to(device),
    }
    denoms = {attn_type: val.clone() for attn_type, val in embed_importance.items()}
    
    model.eval()
    optimizer.zero_grad()
    
    for idx, data in enumerate(trainer.get_train_dataloader()):
        inputs_embeds, attention_mask, labels = preprocess(model, data, prune_type)
        
        output = model.base_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        learned_embedding.weight.retain_grad()
        output.loss.backward()
        
        if prune_type is None:
            importance, denom = importance_score(learned_embedding.weight, orig_embed_mask)
        elif prune_type == "xprompt-tuning":
            importance, denom = importance_score(learned_embedding.weight, model.embed_mask)
        elif prune_type == "rprompt-tuning":
            importance, denom = importance_score(learned_embedding.weight, orig_embed_mask)
        
        embed_importance["token-level"] += importance
        denoms["token-level"] += denom
    
    
    if hasattr(model, "embed_mask") or hasattr(model, "kept_prune"):
        return embed_importance
    
    profile = get_profile(embed_importance["token-level"], "token")
    
    sorted_profiles = sorted(
        profile.items(),
        key=lambda x: x[1],
    )
    
    total_tokens = len(sorted_profiles)
    target_tokens = int(total_tokens * model.prune_ratio)
    
    to_prune_profile = []
    for n, _ in sorted_profiles[:target_tokens]:
        to_prune_profile.insert(0, n)
    
    to_prune = parse_pruning_descriptors(to_prune_profile)
    mask_heads(orig_embed_mask, to_prune)

    kept_prune = set()

    for i in range(learned_embedding.weight.shape[0]):
        if i not in to_prune['token']:
            kept_prune.add(i)
    
    model.embed_mask = orig_embed_mask
    model.kept_prune = kept_prune
    
    return embed_importance


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DynamicTrainingArguments, DynamicPeftArguments, PruneArguments))
    
    args = parser.parse_args_into_dataclasses()
    
    model_args, data_args, training_args, peft_args, prune_args = args
    
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    if data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.dataset import GlueDataset
        dataset = GlueDataset
    elif data_args.task_name.lower() == "super_glue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.dataset import SuperGlueDataset
        dataset = SuperGlueDataset
    else:
        raise NotImplementedError("Task {} is not implemented. Please choose a task from: {}".format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)
    
    trainer, predict_dataset = get_trainer(model_args, data_args, training_args, peft_args, dataset)
    
    model = trainer.model
    model.prune_ratio = prune_args.prune_ratio
    prune_type = prune_args.prune_type
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        weight_decay=training_args.weight_decay,
        lr=training_args.learning_rate)
    
    train(model, trainer, optimizer, training_args, None)
    score = estimate_token_importance(model, trainer, optimizer)
    loss, acc = evaluate(model, optimizer, trainer, prune_type)
    print(f"{prune_type}, after pruned - loss : {loss}   acc : {acc}")
    
    train(model, trainer, optimizer, training_args, prune_type)
    loss, acc = evaluate(model, optimizer, trainer, prune_type)
    print(f"rewinding - loss : {loss}   acc : {acc}")