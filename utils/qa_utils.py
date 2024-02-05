import collections
import string
import regex as re
import numpy as np


def _normalize_answer(text, punc_chars, punc_repl):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_artices(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)
    
    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())
    
    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def normalize_trivia_qa(answer):
    """Normalization used in official TriviaQA evaluation script."""
    return _normalize_answer(
        answer, punc_chars=string.punctuation + "‘’´`_", punc_repl=" ").strip()


def normalize_squad(answer):
    """Normalization used in official SQuAD evaluation script."""
    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
    """Computes the maximum of the metric over all ground truths."""
    return max(
        metric_fn(ground_truth, prediction) for ground_truth in ground_truths
    )


def _exact_match_score(label, prediction):
    return label == prediction


def _f1_score(label, prediction):
    """Compute token f1 score for a single label and prediction."""
    prediction_tokens = prediction.split()
    label_tokens = label.split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(label_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(label_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_metrics(labels, predictions):
    """Computes exact match and f1 QA scores, expecting pre-normalized text."""
    if len(labels) != len(predictions):
        raise ValueError("Number of labels and predictions must match.")
    em = np.mean([
        _metric_max_over_ground_truths(_exact_match_score, t, p)
        for p, t in zip(predictions, labels)
    ])
    f1 = np.mean([
        _metric_max_over_ground_truths(_f1_score, t, p)
        for p, t in zip(predictions, labels)
    ])
    em *= 100
    f1 *= 100
    return {"em": em, "f1": f1}
