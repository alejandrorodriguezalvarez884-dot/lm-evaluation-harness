import functools
import logging
import math
import os
import random
import re
import string
from collections.abc import Iterable
from typing import Callable, List, Optional, Sequence, TypeVar

import numpy as np
import sacrebleu

from lm_eval.api.registry import register_aggregation, register_metric


T = TypeVar("T")

eval_logger = logging.getLogger(__name__)


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("nanmean")
def nanmean(arr):
    if len(arr) == 0 or all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return sorted(arr)[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF is an evaluation metric for machine translation output based on
    character n-gram precision and recall.

    Computed with sacrebleu's defaults: char_order=6, word_order=0, beta=2.
    For chrF++ (word_order=2), use the ``chrf++`` metric instead.

    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("chrf++")
def chrfpp(items):
    """chrF++ is chrF extended with word n-grams, for automatic evaluation of
    machine translation output.

    Computed with word_order=2 and sacrebleu's remaining defaults:
    char_order=6, beta=2. For plain chrF (word_order=0), use the ``chrf``
    metric instead.

    Source: https://github.com/m-popovic/chrF
    Paper: https://aclanthology.org/W17-4770.pdf

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs, word_order=2).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("brier_score")
def brier_score(items):  # This is a passthrough function
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


# Calibration metrics for multiple-choice tasks. As with `brier_score`, each
# document contributes `(gold, probs)` and the score is computed over the whole
# dataset in the aggregation, since binning and temperature fitting need every
# document at once.
CALIBRATION_METRICS = ("ece", "rms_ce", "brier_score_tt", "rms_ce_tt")
# The ones that take `n_bins` and `binning` options from `metric_list`.
BINNED_CALIBRATION_METRICS = ("ece", "rms_ce", "rms_ce_tt")
_BINNING_SCHEMES = ("equal_width", "equal_mass")


def _check_binning(n_bins: int, binning: str) -> None:
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    if binning not in _BINNING_SCHEMES:
        raise ValueError(f"binning must be one of {_BINNING_SCHEMES}, got {binning!r}")


def bind_calibration_kwargs(
    aggregation: Callable[..., float],
    n_bins: int = 10,
    binning: str = "equal_width",
) -> Callable[..., float]:
    """Bind `metric_list` options to a calibration aggregation.

    Validating here means a bad option fails when the task is built, not after
    every document has been scored.
    """
    _check_binning(n_bins, binning)
    return functools.partial(aggregation, n_bins=n_bins, binning=binning)


def _top_label(items) -> tuple[np.ndarray, np.ndarray]:
    """Confidence and correctness of the argmax choice, the one `acc` scores.

    `gold` may be a list of accepted indices. An invalid gold (-100) counts as
    wrong, as it does for `acc`.
    """
    confidences = np.empty(len(items))
    correct = np.empty(len(items))
    for i, (gold, probs) in enumerate(items):
        probs = np.asarray(probs, dtype=float)
        pred = int(np.argmax(probs))
        confidences[i] = probs[pred]
        golds = gold if isinstance(gold, (list, tuple)) else (gold,)
        correct[i] = float(pred in golds)
    return confidences, correct


def _binned_gaps(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int, binning: str
) -> tuple[np.ndarray, np.ndarray]:
    """Weight and |accuracy - confidence| of each non-empty bin."""
    _check_binning(n_bins, binning)
    if binning == "equal_width":
        # Right-closed bins (lo, hi]; the first one also takes 0.
        inner_edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        ids = np.searchsorted(inner_edges, confidences, side="left")
    else:
        # Split the sorted confidences into bins whose sizes differ by at most one.
        order = np.argsort(confidences, kind="stable")
        ids = np.empty(len(confidences), dtype=int)
        for b, chunk in enumerate(np.array_split(order, n_bins)):
            ids[chunk] = b
    counts = np.bincount(ids, minlength=n_bins)
    nonempty = counts > 0
    counts = counts[nonempty]
    conf_sums = np.bincount(ids, weights=confidences, minlength=n_bins)[nonempty]
    acc_sums = np.bincount(ids, weights=correct, minlength=n_bins)[nonempty]
    return counts / len(confidences), np.abs(acc_sums - conf_sums) / counts


def _temperature_scaled(items) -> tuple[np.ndarray, np.ndarray]:
    """Rescale all probabilities with the temperature that minimises the gold NLL.

    The temperature is fit on the evaluated documents themselves. Scaling the
    log-probabilities is equivalent to scaling the choice loglikelihoods, since
    softmax ignores the per-document normalising constant.

    Returns an (n_docs, max_choices) array, zero-padded for documents with fewer
    choices, and the gold indices.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    n_choices = max(len(probs) for _, probs in items)
    # -inf padding keeps missing choices at probability 0 for any temperature.
    log_probs = np.full((len(items), n_choices), -np.inf)
    golds = np.empty(len(items), dtype=int)
    for i, (gold, probs) in enumerate(items):
        probs = np.asarray(probs, dtype=float)
        if isinstance(gold, (list, tuple)) or not 0 <= gold < len(probs):
            raise ValueError(
                "Temperature-tuned calibration metrics need a single in-range "
                f"gold index per document; got gold={gold!r} for a document "
                f"with {len(probs)} choices."
            )
        # Keep probabilities that underflowed to 0 finite in log space.
        log_probs[i, : len(probs)] = np.log(np.clip(probs, np.finfo(float).tiny, None))
        golds[i] = gold
    gold_log_probs = log_probs[np.arange(len(golds)), golds]

    def nll(log_beta: float) -> float:
        beta = np.exp(log_beta)
        return float(
            np.mean(logsumexp(beta * log_probs, axis=1) - beta * gold_log_probs)
        )

    # The NLL is convex in beta = 1/T, hence unimodal in log(beta).
    result = minimize_scalar(
        nll,
        bounds=(np.log(1e-3), np.log(1e3)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    beta = float(np.exp(result.x))
    eval_logger.debug(f"Fitted calibration temperature T={1 / beta:.4f}")

    scaled = beta * log_probs
    scaled -= scaled.max(axis=1, keepdims=True)
    tempered = np.exp(scaled)
    return tempered / tempered.sum(axis=1, keepdims=True), golds


@register_aggregation("ece")
def ece(items, n_bins: int = 10, binning: str = "equal_width") -> float:
    """Expected calibration error of the top-label confidence.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|, where the confidence is
    the probability of the predicted choice. Guo et al., "On Calibration of
    Modern Neural Networks", ICML 2017. https://arxiv.org/abs/1706.04599

    Args:
        items: `(gold, probs)` pairs, one per document.
        n_bins: Number of bins.
        binning: "equal_width" (bins of width 1/n_bins) or "equal_mass" (bins
            holding the same number of documents).
    """
    weights, gaps = _binned_gaps(*_top_label(items), n_bins, binning)
    return float(np.sum(weights * gaps))


@register_aggregation("rms_ce")
def rms_ce(items, n_bins: int = 10, binning: str = "equal_width") -> float:
    """Root-mean-square calibration error of the top-label confidence.

    RMS-CE = sqrt(sum_b (|B_b| / N) * (acc(B_b) - conf(B_b))^2), which weighs
    large gaps more than ECE. Used for MMLU in Hendrycks et al., "Measuring
    Massive Multitask Language Understanding", ICLR 2021.
    https://arxiv.org/abs/2009.03300

    Takes the same arguments as `ece`.
    """
    weights, gaps = _binned_gaps(*_top_label(items), n_bins, binning)
    return float(np.sqrt(np.sum(weights * gaps**2)))


@register_aggregation("brier_score_tt")
def brier_score_tt(items) -> float:
    """Brier score after rescaling with one temperature fit on the evaluated documents.

    Measures the miscalibration that a single global temperature cannot fix.
    """
    probs, golds = _temperature_scaled(items)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(golds)), golds] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


@register_aggregation("rms_ce_tt")
def rms_ce_tt(items, n_bins: int = 10, binning: str = "equal_width") -> float:
    """RMS calibration error after the same temperature fit as `brier_score_tt`.

    Takes the same arguments as `ece`.
    """
    probs, golds = _temperature_scaled(items)
    # A positive temperature keeps the argmax, so correctness is unchanged.
    correct = (probs.argmax(axis=1) == golds).astype(float)
    weights, gaps = _binned_gaps(probs.max(axis=1), correct, n_bins, binning)
    return float(np.sqrt(np.sum(weights * gaps**2)))


@register_metric(
    metric="ece",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="ece",
)
def ece_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="rms_ce",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="rms_ce",
)
def rms_ce_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="brier_score_tt",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score_tt",
)
def brier_score_tt_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="rms_ce_tt",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="rms_ce_tt",
)
def rms_ce_tt_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_bytes",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_bytes_fn(items):  # This is a passthrough function
    return items


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="likelihood",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def likelihood_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr: Sequence[T]) -> float:
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return None


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf++",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf++",
)
def chrfpp_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=False,
    output_type="generate_until",
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    """
    Pool worker: `(i, xs)` → `n` bootstrap replicates
    of `f(xs)`using a RNG seeded with `i`.
    """

    def __init__(self, f: Callable[[Sequence[T]], float], n: int) -> None:
        self.f = f
        self.n = n

    def __call__(self, v: tuple[int, Sequence[T]]) -> list[float]:
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def _bootstrap_internal_no_mp(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> list[float]:
    """
    Single-process fallback: compute `iters` bootstrap replicates
    of statistic`f(xs)`, chunked (≤ 1000 draws).
    """
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print(f"bootstrapping for stddev: {f.__name__}")

    # A single loop replaces the multiprocessing pool.
    for i in tqdm(range(iters // chunk_size)):
        rnd = random.Random(i)
        for _ in range(chunk_size):
            res.append(f(rnd.choices(xs, k=len(xs))))

    return res


def bootstrap_stderr(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> float:
    """
    Bootstrap estimate of the standard error of statistic `f(xs)`
    using up to `iters` resamples, chunked (≤ 1000 draws)

    Executes in parallel unless the env-var `DISABLE_MULTIPROC` is set;
    """
    if not os.getenv("DISABLE_MULTIPROC"):
        import multiprocessing as mp

        # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
        # equivalent to stderr calculated without Bessel's correction in the stddev.
        # Unfortunately, I haven't been able to figure out what the right correction is
        # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
        # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
        # Thankfully, shouldn't matter because our samples are pretty big usually anyways
        res = []
        chunk_size = min(1000, iters)
        from tqdm import tqdm

        print("bootstrapping for stddev:", f.__name__)
        with mp.Pool(mp.cpu_count()) as pool:
            for bootstrap in tqdm(
                pool.imap(
                    _bootstrap_internal(f, chunk_size),
                    [(i, xs) for i in range(iters // chunk_size)],
                ),
                total=iters // chunk_size,
            ):
                # sample w replacement
                res.extend(bootstrap)
    else:
        res = _bootstrap_internal_no_mp(f, xs, iters)

    return sample_stddev(res)


def stderr_for_metric(
    metric: Callable[[Sequence[T]], float], bootstrap_iters: int
) -> Optional[Callable[[Sequence[T]], float]]:
    """
    Return a function that estimates the standard error of `metric(xs)`.

    * If `bootstrap_iters > 0` and the metric is in the pre-approved
      bootstrappable list, use `bootstrap_stderr` with that many draws.
    * If the metric has a closed-form SE (e.g. `mean`, `acc_all`), use it.
    * Otherwise, return `None`.
    """

    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        chrfpp,
        ter,
        nanmean,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def unweighted_mean_stderr(stderrs: List[float]) -> float:
    # Used to aggregate stderrs across subtasks in a group when we are NOT weighting
    # by subtask size (weight_by_size=False), i.e. the group score is the simple
    # unweighted mean of the k subtask means. For k independent subtask means,
    # Var(mean) = (1 / k**2) * sum(se_i**2), so the stderr is sqrt(sum(se_i**2)) / k.
    # (This coincides exactly with pooled_sample_stderr when all subtasks are the same
    # size; it diverges only for unequal sizes, which is precisely the case the pooled
    # formula mis-handles for an unweighted mean.)
    return np.sqrt(sum(stderr**2 for stderr in stderrs)) / len(stderrs)


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert metrics is not None, (
        "Need to pass a list of each subtask's metric for this stderr aggregation"
    )
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = ((curr_score * curr_size) + (score * size)) / (
            curr_size + size
        )  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (
            curr_size + size - 1
        ) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (
            curr_score - score
        ) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)
