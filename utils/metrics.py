import numpy as np
import scipy
import math
import sklearn
import collections
from logging import getLogger
import sklearn.metrics
import functools

from utils.qa_utils import normalize_squad, qa_metrics

logger = getLogger(__name__)


def string_to_float(string, default=-1., **unused_kwargs):
    try:
        return float(string)
    except ValueError:
        return default


def accuracy(predictions, labels) -> dict:
    """Compute the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(labels)).mean())}


def pearson_corrcoef(predictions, labels)-> dict:
    """Compute Pearson correlation coefficient."""
    labels = [string_to_float(label) for label in labels]
    predictions = [string_to_float(prediction) for prediction in predictions]
    pearson_corrcoef = 100 * scipy.stats.pearsonr(labels, predictions)[0]
    
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    
    return {"pearson": pearson_corrcoef}


def spearman_corrcoef(predictions, labels) -> dict:
    labels = [string_to_float(label) for label in labels]
    predictions = [string_to_float(prediction) for prediction in predictions]
    spearman_corrcoef = 100 * scipy.stats.spearmanr(labels, predictions)[0]
    
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    
    return {"spearmanr": spearman_corrcoef}


def preprocess_invalid(predictions, labels, num_classes: int):
    """Preprocess invalid value."""
    def binary_reverse(labels):
        return ['0' if label == '1' else '1' for label in labels]
    
    def safe_convert(value):
        try:
            return int(value)
        except ValueError:
            return -1
    
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    # Get indices of invalid predictions
    if "int" in str(predictions.dtype):
        labels = np.array([safe_convert(x) for x in labels], dtype=np.int32)
        labels = labels.astype(np.int32)
        predictions = predictions.astype(np.int32)
        
        return predictions, labels
    
    logicals = [predictions != str(num) for num in range(num_classes)]
    invalid_idx_mask = np.all(logicals, axis=0)
    # For any prediction != 0 or 1 or ..., we set the prediction to the opposite of its corresponding labels.
    predictions[invalid_idx_mask] = binary_reverse(labels[invalid_idx_mask])
    labels = labels.astype(np.int32)
    predictions = predictions.astype(np.int32)
    
    return predictions, labels



def f1_score_with_invalid(predictions, labels) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      labels: list of labels, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    predictions, labels = preprocess_invalid(predictions, labels, num_classes=2)
    return {"f1": 100 * sklearn.metrics.f1_score(labels, predictions)}


def matthews_corrcoef(predictions, labels) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"matthews_correlation": 100 * sklearn.metrics.matthews_corrcoef(labels, predictions)}


def exact_match(predictions, labels):
    """Computes whether the labels match predictions exactly."""
    return {"em": 100 * float(np.array_equal(labels, predictions))}


def sklearn_metrics_wrapper(num_classes,
                            metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
    """Wraps any sklearn.metric function and returns a t5 metric function.
    Args:
      metric_str: string, the function from `sklearn.metrics` to use.
      metric_dict_str: optional string, if not specified `metric_str` is used as
        the key in the returned dictionary.
      metric_post_process_fn: callable, if specified the final computed metric
        will be passed through this.
      **metric_fn_kwargs: kwargs, passed to the metric function we are calling.
    Returns:
      the function that calculates the metric in a dict.
    """
    if not hasattr(sklearn.metrics, metric_str):
        raise ValueError("sklearn.metrics does not have: %s" % metric_str)

    def fn(predictions, labels):
        predictions, labels = preprocess_invalid(predictions, labels, num_classes=num_classes)
        metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(labels, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return {metric_dict_str or metric_str: metric_val}
    return fn


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
    """Computes the unweighted average of the F1 per class."""
    return sklearn_metrics_wrapper(
        num_classes,
        "fbeta_score",
        metric_dict_str="f1_multiclass",
        metric_post_process_fn=lambda x: 100 * x,
        beta=1,
        labels=range(num_classes),
        average="macro",
        **metric_fn_kwargs)


def squad(predictions, labels):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
      labels: list of lists of strings
      predictions: list of strings
    Returns:
      dict with score_key: squad score across all labels and predictions
    """
    if type(labels[0]) is list:
        labels = [[normalize_squad(t) for t in u] for u in labels]
    else:
        labels = [normalize_squad(t) for t in labels]
    
    predictions = [normalize_squad(p) for p in predictions]
    return qa_metrics(labels, predictions)
