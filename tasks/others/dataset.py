import re
import evaluate
import numpy as np
import logging
import functools
import collections

from tasks.abc_dataset import AbstractDataset
from utils.general import colorstr, colorformat, emojis
from utils.qa_utils import pad_punctuation
from utils import metrics


task_to_keys = {
    "scitail": ("sentence1", "sentence2", "gold_label"),        #   23,596  no test
    "yelp": ("text", "label"),                                  #  560,000  no eval
    "amazon": ("title", "content", "label"),                    # 360,0000  no eval
    "winogrande": ("sentence", "option1", "option2", "answer"), #   40,398  no test
    "paws": ("sentence1", "sentence2", "label")                 #   49,401  no test
}

task_to_metrics = {
    "scitail": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "yelp": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "amazon": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "winogrande": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "paws": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
}

small_datasets_without_all_splits = [
    "scitail", "winogrande", "paws",
]

large_datasets_without_all_splits = [
    "yelp", "amazon"
]

logger = logging.getLogger(__name__)


class OthersDataset(AbstractDataset):
    def __init__(self, data_args, model_args, training_args, tokenizer):
        super().__init__(data_args, model_args, training_args, tokenizer)
        
        # labels
        self.num_labels = 2
        self.labels_list = ["0", "1"]
        
        if self.name == "scitail":
            self.label2id = {"entailment": "0", "neutral": "1"}
        else:
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
        
        # Set max_target_length by label_list
        self.max_target_length = self.set_max_target_length(training_args.generation_max_length)
        
        # Preprocessing the raw_datasets
        if self.name == "yelp":
            self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
        elif self.name in ["scitail", "amazon", "paws"]:
            self.sentence1_key, self.sentence2_key, self.sentence3_key = task_to_keys[data_args.dataset_name]
        else:
            self.sentence1_key, self.sentence2_key, self.sentence3_key, self.sentence4_key = task_to_keys[data_args.dataset_name]
        
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if self.name == "scitail":
            self.label2id = {"entailment": "0", "neutral": "1"}
        else:
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}
        
        # Check dataset
        logger.info(f"{colorstr('bright_yellow', 'bold', 'Check Dataset')}")
        print(f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                f"{colorstr('bright_magenta', 'bold', self.sentence2_key)} : {self.raw_datasets['train'][0][self.sentence2_key]}"
                )
        if self.name in ["scitail", "amazon", "paws"]:
            print(f"{colorstr('bright_magenta', 'bold', self.sentence3_key)} : {self.raw_datasets['train'][0][self.sentence3_key]}")
        if self.name == "winogrande":
            print(f"{colorstr('bright_magenta', 'bold', self.sentence4_key)} : {self.raw_datasets['train'][0][self.sentence4_key]}")
        
        # Set metrics
        self.set_metrics()
        
        # Preprocess format
        self.preprocess_dataset()
        
        if data_args.k_shot_example is not None:
            self.preprocess_k_shot_dataset()
        
        # Tokenize
        self.tokenize_dataset()
        
        # Split Dataset
        self.split_dataset()
        
        # Print number of datasets
        self.print_dataset_numbers()

    def preprocessor(self, example, add_prefix=True):
        extra_fields = {}
        
        if self.name == "yelp":
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key]]
            tgt_texts = [str(example[self.sentence2_key])]
        elif self.name == "scitail":
            src_texts = ["premise:", example[self.sentence1_key],
                         "hypothesis:", example[self.sentence2_key]]
            tgt_texts = [self.label2id[example[self.sentence3_key]]]
        elif self.name == "amazon":
            src_texts = ["sentence:", "<title> {0} <context> {1}".format(
                example[self.sentence1_key], example[self.sentence2_key])]
            tgt_texts = [str(example[self.sentence3_key])]
        elif self.name == "paws":
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key],
                         self.sentence2_key + ":", example[self.sentence2_key]]
            tgt_texts = [str(example[self.sentence3_key])]
        elif self.name == "winogrande":
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key],
                         "option0:", example[self.sentence2_key],
                         "option1:", example[self.sentence3_key]]
            tgt_texts = [
                str(int(example[self.sentence4_key]) - 1) if example[self.sentence4_key].isdigit() else ""
            ]
        
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

    def preprocess_k_shot_dataset(self):
        class_num_dct = {}
        
        for _, value in enumerate(self.label2id.values()):
            class_num_dct[str(value)] = 0
        
        num_example_per_class = self.data_args.k_shot_example // len(class_num_dct)
        if self.data_args.k_shot_example % len(class_num_dct) > 0: num_example_per_class += 1
        shuffled_train_dataset = self.processed_dataset["train"].shuffle(seed=self.training_args.seed)
        index_lst = []
        
        for i, data in enumerate(shuffled_train_dataset):
            if sum(class_num_dct.values()) == self.data_args.k_shot_example:
                break
            
            label = data["target"]
            if class_num_dct[label] < num_example_per_class and sum(class_num_dct.values()) < self.data_args.k_shot_example:
                class_num_dct[label] += 1
                index_lst.append(i)
        
        if self.name in large_datasets_without_all_splits:
            if self.name in ["scitail", "paws"]:
                split_key = {"train": "train", "validation": "validation", "test": "test"}
            elif self.name in ["yelp", "amazon"]:
                split_key = {"train": "train", "validation": "test"}
            else:
                split_key = {"train": "train", "validation": "validation"}
            
            self.processed_dataset[split_key["test"]] = self.processed_dataset[split_key["validation"]]
            
            if len(self.processed_dataset[split_key["train"]]) > 100000:
                validation_size = 10000
            else:
                validation_size = 1000
            indices = self.shuffled_indices(self.processed_dataset[split_key["train"]])[:validation_size]
            valid_dataset = self.subsample(self.processed_dataset[split_key["train"]], None, indices)
            self.processed_dataset[split_key["validation"]] = valid_dataset
        
        self.processed_dataset["train"] = shuffled_train_dataset.select(index_lst)

    def split_dataset(self):
        is_small = None
        if self.name in small_datasets_without_all_splits:
            is_small = True
        elif self.name in large_datasets_without_all_splits:
            is_small = False
        
        if self.name in ["scitail", "paws"]:
            dct = {"train": "train", "validation": "validation", "test": "test"}
        elif self.name in ["yelp", "amazon"]:
            dct = {"train": "train", "validation": "test"}
        else:
            dct = {"train": "train", "validation": "validation"}
        
        # Training
        if self.training_args.do_train:
            self.train_dataset = self.get(split_key=dct,
                                          split="train", 
                                          n_obs=self.data_args.max_train_samples,
                                          split_validation_test=self.data_args.split_validation_test,
                                          is_small=is_small,
                                          is_few=self.data_args.k_shot_example)
        
        # Evaluation
        if self.training_args.do_eval:
            self.eval_dataset = self.get(split_key=dct,
                                         split="validation",
                                         n_obs=self.data_args.max_eval_samples,
                                         split_validation_test=self.data_args.split_validation_test,
                                         is_small=is_small,
                                         is_few=self.data_args.k_shot_example)
        
        if self.training_args.do_predict:
            self.predict_dataset = self.get(split_key=dct,
                                            split="test",
                                            n_obs=self.data_args.max_predict_samples,
                                            split_validation_test=self.data_args.split_validation_test,
                                            is_small=is_small,
                                            is_few=self.data_args.k_shot_example)

    def set_metrics(self):
        self.metrics_name = task_to_metrics[self.name]["name"]
        self.metrics = task_to_metrics[self.name]["metrics"]
