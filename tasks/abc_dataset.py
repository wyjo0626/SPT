from abc import ABC, abstractmethod
import logging
import functools
import enum
import numpy as np

from typing import Callable, List, Mapping
from transformers import (
    AutoTokenizer,
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction
)
import torch
from datasets import load_dataset

from utils.general import colorstr, colorformat, emojis

logger = logging.getLogger(__name__)


class AbstractDataset(ABC):
    task = NotImplemented
    name = NotImplemented
    data_args = NotImplemented
    training_args = NotImplemented
    model_args = NotImplemented
    tokenizer = NotImplemented
    
    raw_datasets = NotImplemented
    processed_dataset = NotImplemented
    tokenized_dataset = NotImplemented
    preprocessor: Callable = NotImplemented
    postprocessor: Callable = NotImplemented
    prefix = NotImplemented
    metrics = NotImplemented
    metrics_name = NotImplemented
    
    train_dataset = NotImplemented
    eval_dataset = NotImplemented
    predict_dataset = NotImplemented
    
    labels_list = None
    num_labels = None
    label2id = None
    id2label = None
    
    max_target_length = NotImplemented
    max_seq_length = NotImplemented
    data_collator = NotImplemented
    column_names = ['source', 'target']
    
    def __init__(self, data_args, model_args, training_args, tokenizer):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        
        self.task = data_args.task_name
        self.name = data_args.dataset_name
        
        if self.task in ['glue', 'super_glue']:
            self.raw_datasets = load_dataset(self.task, self.name)
        else:
            raise NotImplementedError

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = 'max_length'
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False
        
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger then the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            self.data_collator = None
        
        if self.name in POST_PROCESSOR:
            post_processor = POST_PROCESSOR[self.name](tokenizer, data_args.ignore_pad_token_for_loss)
        else:
            post_processor = PostProcessor(tokenizer, data_args.ignore_pad_token_for_loss)
        
        self.postprocessor = post_processor.process
    
    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning(f"n_obs is set to {n_obs}")
        return n_obs
    
    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.training_args.seed)
        return torch.randperm(num_samples, generator=generator).tolist()
    
    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)
    
    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if "validation" in split :
            return indices[:validation_size]
        else:
            return indices[validation_size:]
    
    def get(self, split_key, split, n_obs=None, split_validation_test=False, is_small=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and is_small == True and split != "train":
            dataset = self.tokenized_dataset[split_key["validation"]]
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset) // 2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K and
        # For larger datasets (n_samples > 100K), we divide training set into 10K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and is_small == False and split != "test":
            dataset = self.tokenized_dataset[split_key["train"]]
            if len(dataset) > 100000:
                validation_size = 10000
            else:
                validation_size = 1000
            indices = self.get_split_indices(split, dataset, validation_size=validation_size)
            dataset = self.subsample(dataset, n_obs, indices)
        elif split_validation_test and is_small == False and split == "test":
            dataset = self.tokenized_dataset[split_key["validation"]]
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        else:
            dataset = self.tokenized_dataset[split_key[split]]
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        
        return dataset
            
    
    def set_max_target_length(self, default_max_length):
        if self.labels_list is not None:
            self.max_target_length = max([len(self.tokenizer.encode(label)) for label in self.labels_list])
        self.max_target_length = default_max_length
    
    def seq2seq_format(self,
                       sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix] + sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                'task': self.name,}
    
    def preprocess_dataset(self):
        self.processed_dataset = self.raw_datasets.map(
            functools.partial(self.preprocessor, add_prefix=self.data_args.add_prefix),
            remove_columns=self.raw_datasets['train'].column_names
        )
    
    def tokenize_dataset(self):
        if any(x in self.model_args.model_name_or_path for x in ["bert", "roberta", "albert"]):
            logger.info(f"Loading encoder model from {self.model_args.model_name_or_path}")
            tokenize_function = self.encoder_preprocess_function
            self.compute_metrics = self.compute_metrics_encoder
        elif any(x in self.model_args.model_name_or_path for x in ["t5"]):
            logger.info(f"Loading seq2seq model from {self.model_args.model_name_or_path}")
            tokenize_function = self.seq2seq_preprocess_function
            self.compute_metrics = self.compute_metrics_seq2seq
        elif any(x in self.model_args.model_name_or_path for x in ["gpt", "llama"]):
            logger.info(f"Loading decoder model from {self.model_args.model_name_or_path}")
            tokenize_function = self.decoder_preprocess_function
            self.compute_metrics = self.compute_metrics_decoder
            self.training_args.generation_max_length = self.max_seq_length + self.training_args.generation_max_length
        else:
            raise NotImplementedError
        
        self.training_args.metric_for_best_model = self.metrics_name[0]
        self.training_args.greater_is_better = True
        
        self.tokenized_dataset = self.processed_dataset.map(
            functools.partial(tokenize_function),
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
        )
    
    @abstractmethod
    def preprocess_k_shot_dataset(self):
        # Preprocess for k-shot learning after preprocess_dataset
        ...
    
    @abstractmethod
    def split_dataset(self):
        # Split Dataset train, evaluate, pedict
        ...
    
    def print_dataset_numbers(self):
        if self.training_args.do_train:
            logger.info(f"{colorstr('bright_green', 'bold', 'number of train dataset')} : {len(self.train_dataset)}")
        if self.training_args.do_eval:
            logger.info(f"{colorstr('bright_green', 'bold', 'number of eval dataset')}  : {len(self.eval_dataset)}")
        if self.training_args.do_predict:
            logger.info(f"{colorstr('bright_green', 'bold', 'number of test dataset')}  : {len(self.predict_dataset)}")
    
    @abstractmethod
    def set_metrics(self):
        ...
    
    # Preprocessing tokenize datasets
    def encoder_preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples['source'],
                                      max_length=self.max_seq_length,
                                      padding=self.padding,
                                      truncation=True)
        # Setup the tokenizer for targets
        if hasattr(self, "is_regression") and self.is_regression:
            labels = torch.tensor([self.round_stsb_target(float(i)) for i in examples["target"]])
        else:
            labels = torch.tensor([int(i) for i in examples['target']])
        model_inputs['labels'] = labels
        return model_inputs
    
    def seq2seq_preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples['source'],
                                      max_length=self.max_seq_length,
                                      padding=self.padding,
                                      truncation=True)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['target'],
                                    max_length=self.max_target_length,
                                    padding=self.padding,
                                    truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == 'max_length' and self.data_args.ignore_pad_token_for_loss:
            labels['input_ids'] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']
            ]
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def decoder_preprocess_function(self, examples):
        batch_size = len(examples['source'])
        inputs = [f"{x} Label : " for x in examples['source']]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(examples['target'])
        
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.data_args.max_seq_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.data_args.max_seq_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            if (self.task == "glue" or self.task == "super_glue") and self.data_args.max_seq_length - len(sample_input_ids) < 0:
                # Some data have a sequence longer than data_args.max_seq_length, 
                # the label might not be processed correctly, 
                # such as max_seq_length=3, label["input_ids"] = [-100, -100, -100, -100, 101, 1015, 102, 0]
                labels["input_ids"][i] = label_input_ids[len(sample_input_ids) - self.data_args.max_seq_length:]
            else:
                labels["input_ids"][i] = [-100] * (self.data_args.max_seq_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.data_args.max_seq_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.data_args.max_seq_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.data_args.max_seq_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics_encoder(self, p: EvalPrediction):
        preds, labels = p
        
        num_logits = preds.shape[-1]
        if num_logits == 1:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        
        result = {}
        for metric in self.metrics:
            result.update(metric(predictions=preds, labels=labels))
        return result
    
    def compute_metrics_seq2seq(self, p: EvalPrediction):
        preds, labels = p
        decoded_preds, decoded_labels = self.postprocessor(preds=preds, labels=labels, data_info=None)
        
        result = {}
        for metric in self.metrics:
            result.update(metric(predictions=decoded_preds, labels=decoded_labels))
        return result
    
    def compute_metrics_decoder(self, p: EvalPrediction):
        output_sequences, labels = p
        preds = output_sequences[:, self.data_args.max_seq_length:]
        decoded_preds, decoded_labels = self.postprocessor(preds=preds, labels=labels, data_info=None)

        result = {}
        for metric in self.metrics:
            result.update(metric(predictions=decoded_preds, labels=decoded_labels))
        return result


class PostProcessor(ABC):
    """Postprocess the predictions and labels to make them suitable for evaluation."""
    def __init__(self, tokenizer, ignore_pad_token_for_loss):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
    
    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels 


class MultiRC(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        preds = [{"group": info["group"], "value":pred} for info, pred in zip(data_info, preds)]
        labels = [{"group": info["group"], "value": label} for info, label in zip(data_info, labels)] 
        return preds, labels 


class Record(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        # labels = [info["answers"] for info in data_info]
        return preds, labels 


POST_PROCESSOR = {
    # "multirc": MultiRC,
    # "record": Record
}
