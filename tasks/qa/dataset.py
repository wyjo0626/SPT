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
    "squad": ("question", "context", "answers"),                                        #  87,599   no test
    "newsqa": ("question", "context", "answers"),                                       #  74,160   no test
    "searchqa": ("question", "context", "answers"),                                     # 117,384   no test
    "hotpotqa": ("question", "context", "answers"),                                     #  72,928   no test
    "nq": ("question", "context", "answers"),                                           # 104,071   no test
    "drop": ("question", "passage", "answers_spans"),                                   #  77,400   no test
    "piqa": ("goal", "sol1", "sol2", "label"),                                          #  16,113   no test
    "commonsense_qa": ("question", "choices", "answerKey"),                             #   9,741   no test
    "social_i_qa": ("question", "context", "answerA", "answerB", "answerC", "label"),   #  33,410   no test
}

task_to_metrics = {
    "squad": {"name": ["f1", "em"], "metrics": [metrics.squad]},
    "newsqa": {"name": ["f1", "em"], "metrics": [metrics.squad]},
    "searchqa": {"name": ["f1", "em"], "metrics": [metrics.squad]},
    "hotpotqa": {"name": ["f1", "em"], "metrics": [metrics.squad]},
    "nq": {"name": ["f1", "em"], "metrics": [metrics.squad]},
    "drop": {"name": ["f1", "em"], "metrics": [metrics.squad]},
    "piqa": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "commonsense_qa": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "social_i_qa": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
}

small_datasets_without_all_splits = [
    "squad", "newsqa", "hotpotqa", "drop", "piqa", "commonsense_qa", "social_i_qa"
]

large_datasets_without_all_splits = [
    "searchqa", "nq"
]

logger = logging.getLogger(__name__)


class QADataset(AbstractDataset):
    def __init__(self, data_args, model_args, training_args, tokenizer):
        super().__init__(data_args, model_args, training_args, tokenizer)
        
        # labels
        if self.name == "piqa":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif self.name == "commonsense_qa":
            self.num_labels = 5
            self.label_list = ["A", "B", "C", "D", "E"]
        elif self.name == "social_i_qa":
            self.num_labels = 3
            self.label_list = ["0", "1", "2"]

        # Set max_target_length by label_list
        self.set_max_target_length(training_args.generation_max_length)
        
        # Preprocessing the raw_datasets
        if self.name in [
            "squad", "newsqa", "searchqa", "hotpotqa", "nq", "drop", "commonsense_qa"
        ]:
            self.sentence1_key, self.sentence2_key, self.sentence3_key = task_to_keys[self.name]
        elif self.name == "piqa":
            self.sentence1_key, self.sentence2_key, self.sentence3_key, \
                self.sentence4_key = task_to_keys[self.name]
        elif self.name == "social_i_qa":
            self.sentence1_key, self.sentence2_key, self.sentence3_key, \
                self.sentence4_key, self.sentence5_key, self.sentence6_key = task_to_keys[self.name]

        if self.name in ["piqa", "commonsense_qa", "social_i_qa"]:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
        
        # Check dataset
        logger.info(f"{colorstr('bright_yellow', 'bold', 'Check Dataset')}")
        print(f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                f"{colorstr('bright_magenta', 'bold', self.sentence2_key)} : {self.raw_datasets['train'][0][self.sentence2_key]}\n"
                f"{colorstr('bright_magenta', 'bold', self.sentence3_key)} : {self.raw_datasets['train'][0][self.sentence3_key]}"
                )
        if self.name in ["piqa", "social_i_qa"]:
            print(f"{colorstr('bright_magenta', 'bold', self.sentence4_key)} : {self.raw_datasets['train'][0][self.sentence4_key]}")
        if self.name == "social_i_qa":
            print(f"{colorstr('bright_magenta', 'bold', self.sentence5_key)} : {self.raw_datasets['train'][0][self.sentence5_key]}\n"
                    f"{colorstr('bright_magenta', 'bold', self.sentence6_key)} : {self.raw_datasets['train'][0][self.sentence6_key]}"
                    )

        # Set metrics
        self.set_metrics()
        
        # Preprocess format
        self.preprocess_dataset()
        
        # Tokenize
        self.tokenize_dataset()
        
        # Split Dataset
        self.split_dataset()
        
        # Print number of datasets
        self.print_dataset_numbers()
    
    def preprocessor(self, example, add_prefix=True):
        extra_fields = {}
        
        if self.name in ["squad", "newsqa", "searchqa", "hotpotqa", "nq", "drop"]:
            src_texts = [self.sentence1_key + ":", pad_punctuation(example[self.sentence1_key]),
                         self.sentence2_key + ":", pad_punctuation(example[self.sentence2_key])]
            
            if self.name == "squad":
                answer = pad_punctuation(example[self.sentence3_key]["text"][0]).split("\t")
            elif self.name in ["newsqa", "searchqa", "hotpotqa", "nq"]:
                answer = pad_punctuation(example[self.sentence3_key][0]).split("\t")
            elif self.name == "drop":
                answer = pad_punctuation(example[self.sentence3_key]["spans"][0])
            
            tgt_texts = [answer] if type(answer) == str else answer
        elif self.name == "commonsense_qa":
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key], 
                         self.sentence2_key + "1:", example[self.sentence2_key]["text"][0],
                         self.sentence2_key + "2:", example[self.sentence2_key]["text"][1],
                         self.sentence2_key + "3:", example[self.sentence2_key]["text"][2],
                         self.sentence2_key + "4:", example[self.sentence2_key]["text"][3],
                         self.sentence2_key + "5:", example[self.sentence2_key]["text"][4],
                         ]
            answer = example[self.sentence3_key]
            if answer == "": answer = "A"
            tgt_texts = [str(self.label2id[answer])]
        elif self.name == "piqa":
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key], 
                         "choiece1:", example[self.sentence2_key],
                         "choiece2:", example[self.sentence3_key]]
            tgt_texts = [str(example[self.sentence4_key])]
        elif self.name == "social_i_qa":
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key],
                         self.sentence2_key + ":", example[self.sentence2_key],
                         "|| choice0:", example[self.sentence3_key],
                         "|| choice1:", example[self.sentence4_key],
                         "|| choice2:", example[self.sentence5_key],
                         ]
            tgt_texts = [str(int(example[self.sentence6_key]) - 1)]
        
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields)
    
    def preprocess_k_shot_dataset(self):
        None
    
    def split_dataset(self):
        is_small = None
        if self.name in small_datasets_without_all_splits:
            is_small = True
        elif self.name in large_datasets_without_all_splits:
            is_small = False
        
        dct = {"train": "train", "validation": "validation", "test": "test"}
        
        # Training
        if self.training_args.do_train:
            self.train_dataset = self.get(split_key=dct,
                                          split="train", 
                                          n_obs=self.data_args.max_train_samples,
                                          split_validation_test=self.data_args.split_validation_test,
                                          is_small=is_small)
        
        # Evaluation
        if self.training_args.do_eval:
            self.eval_dataset = self.get(split_key=dct,
                                         split="validation",
                                         n_obs=self.data_args.max_eval_samples,
                                         split_validation_test=self.data_args.split_validation_test,
                                         is_small=is_small)
        
        if self.training_args.do_predict:
            self.predict_dataset = self.get(split_key=dct,
                                            split="test",
                                            n_obs=self.data_args.max_predict_samples,
                                            split_validation_test=self.data_args.split_validation_test,
                                            is_small=is_small)

    def set_metrics(self):
        self.metrics_name = task_to_metrics[self.name]["name"]
        self.metrics = task_to_metrics[self.name]["metrics"]
