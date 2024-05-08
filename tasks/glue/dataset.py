import evaluate
import numpy as np
import logging
import functools

from tasks.abc_dataset import AbstractDataset
from utils.general import colorstr, colorformat, emojis
from utils import metrics

task_to_keys = {
    "cola": ("sentence", None),         #   8,551   no test
    "mnli": ("premise", "hypothesis"),  # 392,702   no test
    "mrpc": ("sentence1", "sentence2"), #   3,668
    "qnli": ("question", "sentence"),   # 104,743   no test
    "qqp": ("question1", "question2"),  # 363,846   no test
    "rte": ("sentence1", "sentence2"),  #   2,490   no test
    "sst2": ("sentence", None),         #  67,349   no test
    "stsb": ("sentence1", "sentence2"), #   5,749   no test
    "wnli": ("sentence1", "sentence2"), #     635   no test
}

task_to_metrics = {
    "cola": {"name": ["matthews_correlation"], "metrics": [metrics.matthews_corrcoef]},
    "mnli": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "mnli_mismatched": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "mnli_matched": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "mrpc": {"name": ["accuracy", "f1"], "metrics": [metrics.accuracy, metrics.f1_score_with_invalid]},
    "qnli": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "qqp": {"name": ["accuracy", "f1"], "metrics": [metrics.accuracy, metrics.f1_score_with_invalid]},
    "rte": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "sst2": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
    "stsb": {"name": ["pearson", "spearmanr"], "metrics": [metrics.pearson_corrcoef, metrics.spearman_corrcoef]},
    "wnli": {"name": ["accuracy"], "metrics": [metrics.accuracy]},
}

small_datasets_without_all_splits = [
    "cola", "rte", "stsb", "wnli"
]

large_datasets_without_all_splits = [
    "mnli", "qnli", "qqp", "sst2"
]

logger = logging.getLogger(__name__)


class GlueDataset(AbstractDataset):
    def __init__(self, data_args, model_args, training_args, tokenizer):
        super().__init__(data_args, model_args, training_args, tokenizer)
        
        # labels
        self.is_regression = data_args.dataset_name == "stsb"
        if not self.is_regression:
            self.labels_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.labels_list)
        
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
        
        # Set max_target_length by labels_list
        self.max_target_length = self.set_max_target_length(training_args.generation_max_length)
        
        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
        
        # Check dataset
        logger.info(f"{colorstr('bright_yellow', 'bold', 'Check Dataset')}")
        if self.sentence2_key is None:
            print(f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : {self.id2label[self.raw_datasets['train'][0]['label']]}"
                  )
        else:
            print(f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', self.sentence2_key)} : {self.raw_datasets['train'][0][self.sentence2_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : "
                  f"{self.raw_datasets['train'][0]['label'] if self.is_regression else self.id2label[self.raw_datasets['train'][0]['label']]}"  
                  )
        
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
    
    def round_stsb_target(self, label):
        """STSB maps two sentences to a floating point number between 1 and 5
        representing their semantic similarity. Since we are treating all tasks as
        text-to-text tasks we need to convert this floating point number to a string.
        The vast majority of the similarity score labels in STSB are in the set
        [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
        entry in this set, and then we convert the result to a string (literally e.g.
        "3.4"). This converts STSB roughly into a 26-class classification dataset.
        Args:
        label: original label.
        Returns:
        A preprocessed label.
        """
        return np.round((label * 5) / 5, decimals=1)
    
    def preprocessor(self, example, add_prefix=True):
        if self.sentence2_key is None:
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key]]
        else:
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key],
                         self.sentence2_key + ":", example[self.sentence2_key]]
        
        tgt_texts = [str(example['label']) if not self.is_regression 
                     else str(self.round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
    def preprocess_k_shot_dataset(self):
        class_num_dct = {}
        
        if not self.is_regression:
            for _, value in enumerate(self.label2id.values()):
                class_num_dct[str(value)] = 0
        else:
            class_num_dct = {
                "0": 0,
                "1": 0,
            }
        
        num_example_per_class = self.data_args.k_shot_example // len(class_num_dct)
        if self.data_args.k_shot_example % len(class_num_dct) > 0: num_example_per_class += 1
        shuffled_train_dataset = self.processed_dataset["train"].shuffle(seed=self.training_args.seed)
        index_lst = []
        
        for i, data in enumerate(shuffled_train_dataset):
            if sum(class_num_dct.values()) == self.data_args.k_shot_example:
                break
            
            label = data["target"]
            if self.data_args.task_name == "stsb":
                label = "0" if float(label) <= 2.5 else "1"
            if class_num_dct[label] < num_example_per_class and sum(class_num_dct.values()) < self.data_args.k_shot_example:
                class_num_dct[label] += 1
                index_lst.append(i)
        
        if self.name in large_datasets_without_all_splits:
            if self.name == "mnli":
                split_key = {"train": "train", "validation": "validation_matched", "test": "test_matched"}
            else:
                split_key = {"train": "train", "validation": "validation", "test": "test"}
            
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
        
        if self.name == "mnli":
            dct = {"train": "train", "validation": "validation_matched", "test": "test_matched"}
        else:
            dct = {"train": "train", "validation": "validation", "test": "test"}
        
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
