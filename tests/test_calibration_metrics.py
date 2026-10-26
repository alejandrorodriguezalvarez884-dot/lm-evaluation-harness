"""Tests for the calibration metrics of multiple-choice tasks (#2189)."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from lm_eval.api.metrics import (
    bind_calibration_kwargs,
    brier_score_tt,
    ece,
    rms_ce,
    rms_ce_tt,
)


# Four binary documents. Top-label confidence / correct:
#   0.9 / yes, 0.8 / no, 0.7 / yes, 0.6 / no  -> accuracy 0.5, mean confidence 0.75
FOUR_DOCS = [
    (0, [0.9, 0.1]),
    (1, [0.8, 0.2]),
    (1, [0.3, 0.7]),
    (0, [0.4, 0.6]),
]


def reference_calibration_error(items, n_bins, p):
    """Straightforward per-bin loop (equal-width, right-closed bins)."""
    conf = np.array([max(probs) for _, probs in items])
    correct = np.array([float(int(np.argmax(probs)) == gold) for gold, probs in items])
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = (conf > lo) & (conf <= hi)
        if b == 0:
            in_bin |= conf == 0.0
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - conf[in_bin].mean())
            total += in_bin.mean() * gap**p
    return total if p == 1 else math.sqrt(total)


def test_one_doc_per_bin():
    # Each document sits alone in a bin: gaps 0.1, 0.8, 0.3, 0.6.
    assert ece(FOUR_DOCS) == pytest.approx((0.1 + 0.8 + 0.3 + 0.6) / 4)
    assert rms_ce(FOUR_DOCS) == pytest.approx(
        math.sqrt((0.01 + 0.64 + 0.09 + 0.36) / 4)
    )


def test_single_bin_is_accuracy_minus_mean_confidence():
    assert ece(FOUR_DOCS, n_bins=1) == pytest.approx(0.25)
    assert rms_ce(FOUR_DOCS, n_bins=1) == pytest.approx(0.25)


def test_equal_mass_bins():
    # Sorted by confidence: {0.6 no, 0.7 yes} and {0.8 no, 0.9 yes}.
    # Gaps |0.5 - 0.65| = 0.15 and |0.5 - 0.85| = 0.35, each with weight 1/2.
    assert ece(FOUR_DOCS, n_bins=2, binning="equal_mass") == pytest.approx(0.25)
    assert rms_ce(FOUR_DOCS, n_bins=2, binning="equal_mass") == pytest.approx(
        math.sqrt(0.5 * 0.15**2 + 0.5 * 0.35**2)
    )


def test_equal_mass_with_more_bins_than_docs():
    # Every document gets its own bin and the empty bins are skipped.
    assert ece(FOUR_DOCS, n_bins=10, binning="equal_mass") == pytest.approx(0.45)


def test_bins_are_right_closed():
    # Confidence exactly 0.5 belongs to [0, 0.5], not to (0.5, 1].
    items = [(0, [0.5, 0.5]), (1, [0.6, 0.4])]  # 0.5 / yes, 0.6 / no
    assert ece(items, n_bins=2) == pytest.approx(0.5 * 0.5 + 0.5 * 0.6)


def test_dataset_level_not_mean_of_per_doc_scores():
    # Two docs in one bin cancel out: a calibrated bin despite per-doc errors.
    items = [(0, [0.5, 0.5]), (1, [0.5, 0.5])]  # 0.5 / yes, 0.5 / no
    assert ece(items) == pytest.approx(0.0)
    assert np.mean([ece([item]) for item in items]) == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [ece, rms_ce])
def test_all_correct_and_certain_is_zero(metric):
    items = [(0, [1.0, 0.0, 0.0]), (2, [0.0, 0.0, 1.0])]
    assert metric(items) == pytest.approx(0.0)


@pytest.mark.parametrize("metric", [ece, rms_ce])
def test_all_wrong_and_certain_is_one(metric):
    items = [(1, [1.0, 0.0, 0.0]), (0, [0.0, 0.0, 1.0])]
    assert metric(items) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [ece, rms_ce])
def test_all_correct_underconfident(metric):
    items = [(0, [0.8, 0.2]), (1, [0.1, 0.8, 0.1])]
    assert metric(items) == pytest.approx(0.2)


def test_different_number_of_choices():
    items = [
        (0, [0.5, 0.3, 0.2]),  # 0.5 / yes
        (2, [0.1, 0.2, 0.3, 0.4]),  # 0.4 / no
        (1, [0.25, 0.75]),  # 0.75 / yes
    ]
    assert ece(items) == pytest.approx((0.5 + 0.4 + 0.25) / 3)


def test_multiple_gold_answers():
    # Predicted choice 2 is one of the accepted answers.
    assert ece([([0, 2], [0.1, 0.2, 0.7])]) == pytest.approx(0.3)


def test_out_of_range_gold_counts_as_wrong():
    # process_results maps an invalid gold to -100, which `acc` scores as wrong.
    assert ece([(-100, [0.7, 0.3])]) == pytest.approx(0.7)


@pytest.mark.parametrize("n_bins", [1, 2, 7, 10, 15])
def test_matches_reference_on_random_data(n_bins):
    rng = np.random.default_rng(n_bins)
    items = []
    for _ in range(300):
        n_choices = int(rng.integers(2, 6))
        probs = rng.dirichlet(np.ones(n_choices) * 0.5)
        items.append((int(rng.integers(n_choices)), probs.tolist()))
    assert ece(items, n_bins=n_bins) == pytest.approx(
        reference_calibration_error(items, n_bins, p=1)
    )
    assert rms_ce(items, n_bins=n_bins) == pytest.approx(
        reference_calibration_error(items, n_bins, p=2)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bins": 0},
        {"n_bins": 2.5},
        {"binning": "quantile"},
    ],
)
def test_invalid_binning_options(kwargs):
    with pytest.raises(ValueError):
        ece(FOUR_DOCS, **kwargs)
    with pytest.raises(ValueError):
        bind_calibration_kwargs(ece, **kwargs)


def test_bind_calibration_kwargs():
    bound = bind_calibration_kwargs(rms_ce, n_bins=2, binning="equal_mass")
    assert bound(FOUR_DOCS) == rms_ce(FOUR_DOCS, n_bins=2, binning="equal_mass")
    with pytest.raises(TypeError):
        bind_calibration_kwargs(ece, nbins=5)


def test_temperature_tuned_hand_computed():
    # Every doc says P(choice 0) = 0.9 but choice 0 is right 3 times out of 4.
    # The NLL is minimised when the rescaled P(choice 0) equals 0.75, i.e.
    # T = log(9) / log(3) = 2. Brier per doc: 2 * 0.25^2 when right and
    # 2 * 0.75^2 when wrong.
    items = [(0, [0.9, 0.1])] * 3 + [(1, [0.9, 0.1])]
    assert brier_score_tt(items) == pytest.approx((3 * 0.125 + 1.125) / 4)
    # A single global temperature fixes this miscalibration completely.
    assert rms_ce(items) == pytest.approx(0.15)
    assert rms_ce_tt(items) == pytest.approx(0.0, abs=1e-6)


def test_temperature_tuning_recovers_true_probabilities():
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 2.0, size=(20000, 4))
    true_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    golds = [int(rng.choice(4, p=p)) for p in true_probs]
    # The model is overconfident: its probabilities use temperature 1/3.
    sharp = np.exp(3 * logits) / np.exp(3 * logits).sum(axis=1, keepdims=True)
    items = list(zip(golds, sharp, strict=True))

    true_brier = np.mean(np.sum((true_probs - np.eye(4)[golds]) ** 2, axis=1))
    assert brier_score_tt(items) == pytest.approx(true_brier, abs=1e-3)
    assert rms_ce(items) > 0.1
    assert rms_ce_tt(items) < 0.02


def test_temperature_tuned_different_number_of_choices():
    rng = np.random.default_rng(1)
    items = []
    for _ in range(200):
        n_choices = int(rng.integers(2, 6))
        probs = rng.dirichlet(np.ones(n_choices))
        items.append((int(rng.integers(n_choices)), probs))

    # Reference: per-doc loop with a brute-force grid search on the NLL.
    def tempered(probs, beta):
        p = np.asarray(probs) ** beta
        return p / p.sum()

    def nll(beta):
        return -np.mean([np.log(tempered(p, beta)[g]) for g, p in items])

    betas = np.exp(np.linspace(np.log(1e-2), np.log(1e2), 4001))
    best = betas[np.argmin([nll(b) for b in betas])]
    expected = np.mean(
        [np.sum((tempered(p, best) - np.eye(len(p))[g]) ** 2) for g, p in items]
    )
    assert brier_score_tt(items) == pytest.approx(expected, abs=1e-4)


def test_temperature_tuned_handles_zero_probabilities():
    items = [(0, [1.0, 0.0]), (1, [0.0, 1.0]), (0, [0.6, 0.4])]
    assert np.isfinite(brier_score_tt(items))
    assert np.isfinite(rms_ce_tt(items))


@pytest.mark.parametrize("gold", [[0, 1], -100, 2])
@pytest.mark.parametrize("metric", [brier_score_tt, rms_ce_tt])
def test_temperature_tuned_needs_a_single_valid_gold(metric, gold):
    with pytest.raises(ValueError, match="single in-range gold"):
        metric([(gold, [0.7, 0.3]), (0, [0.6, 0.4])])


TEST_DATA = Path(__file__).parent / "test_configs" / "test_data.json"


def make_task(metric_list):
    from lm_eval.api.task import ConfigurableTask

    return ConfigurableTask(
        config={
            "task": "calibration_test",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(TEST_DATA)}},
            "output_type": "multiple_choice",
            "doc_to_text": "{{question}}",
            "doc_to_choice": "{{choices}}",
            "doc_to_target": "{{answer}}",
            "test_split": "test",
            "metric_list": metric_list,
        }
    )


def test_task_scores_calibration_metrics():
    from lm_eval.evaluator_utils import _compute_task_aggregations

    task = make_task(
        [
            {"metric": "acc"},
            {"metric": "ece", "n_bins": 1},
            {"metric": "rms_ce", "n_bins": 3, "binning": "equal_mass"},
            {"metric": "brier_score_tt"},
            {"metric": "rms_ce_tt"},
        ]
    )
    raw_metrics = defaultdict(list)
    items = []
    for i, doc in enumerate(task.test_docs()):
        n_choices, target = len(task.doc_to_choice(doc)), task.doc_to_target(doc)
        # Alternate confident-right, confident-wrong and hesitant-right documents,
        # so that the score depends on the binning.
        lls = np.full(n_choices, -5.0 if i % 3 < 2 else -0.5)
        lls[target if i % 3 != 1 else (target + 1) % n_choices] = 0.0
        result = task.process_results(doc, [(ll, False) for ll in lls])
        for key, value in result.items():
            raw_metrics[(key, "none")].append(value)
        gold, probs = result["ece"]
        assert gold == task.doc_to_target(doc)
        assert sum(probs) == pytest.approx(1.0)
        assert int(np.argmax(probs)) == int(np.argmax(lls))
        assert result["acc"] == float(np.argmax(lls) == gold)
        items.append(result["ece"])

    agg, n_docs = _compute_task_aggregations(task, raw_metrics, bootstrap_iters=100)
    assert n_docs == len(items)
    # The options from metric_list reach the aggregation.
    assert ece(items, n_bins=1) != pytest.approx(ece(items))
    assert agg["ece,none"] == pytest.approx(ece(items, n_bins=1))
    assert rms_ce(items, n_bins=3, binning="equal_mass") != pytest.approx(rms_ce(items))
    assert agg["rms_ce,none"] == pytest.approx(
        rms_ce(items, n_bins=3, binning="equal_mass")
    )
    assert agg["brier_score_tt,none"] == pytest.approx(brier_score_tt(items))
    assert agg["rms_ce_tt,none"] == pytest.approx(rms_ce_tt(items))
    assert agg["ece_stderr,none"] == "N/A"
    assert task.higher_is_better()["ece"] is False


def test_task_rejects_bad_binning_options_at_construction():
    with pytest.raises(ValueError, match="n_bins"):
        make_task([{"metric": "ece", "n_bins": 0}])
    with pytest.raises(TypeError):
        make_task([{"metric": "rms_ce", "nbins": 15}])
