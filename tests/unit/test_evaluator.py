"""Unit tests for mindlog.pipeline.evaluator — per-field metrics, exact
match, bootstrap CIs, and the full report assembly."""

import pytest
from mindlog.pipeline.evaluator import (
    bootstrap_accuracy_ci,
    compute_exact_match,
    compute_field_metrics,
    generate_evaluation_report,
)


def test_compute_field_metrics_perfect_agreement():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]

    metrics = compute_field_metrics(
        y_true, y_pred, "affect_valence", ["positive", "negative", "neutral"]
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["cohens_kappa"] == 1.0
    assert metrics["n_errors"] == 0


def test_compute_field_metrics_partial_agreement():
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "neutral", "negative"]

    metrics = compute_field_metrics(
        y_true, y_pred, "affect_valence", ["positive", "negative", "neutral"]
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["n_samples"] == 4


def test_compute_field_metrics_excludes_error_entries_from_scoring():
    y_true = ["positive", "negative", "ERROR"]
    y_pred = ["positive", "ERROR", "neutral"]

    metrics = compute_field_metrics(
        y_true, y_pred, "affect_valence", ["positive", "negative", "neutral"]
    )

    # Only the first pair has no ERROR on either side.
    assert metrics["n_samples"] == 1
    assert metrics["n_errors"] == 2
    assert metrics["accuracy"] == 1.0


def test_compute_field_metrics_all_errors_reports_error_key():
    metrics = compute_field_metrics(["ERROR"], ["ERROR"], "affect_valence", ["positive"])

    assert "error" in metrics


def test_compute_field_metrics_empty_input():
    metrics = compute_field_metrics([], [], "affect_valence", ["positive"])

    assert metrics == {"error": "No samples to evaluate"}


def test_compute_exact_match_counts_all_fields_correct():
    extractions = [
        {"id": "S1", "extraction": {"a": "x", "b": "y"}},
        {"id": "S2", "extraction": {"a": "x", "b": "z"}},  # b wrong
    ]
    ground_truths = [
        {"id": "S1", "labels": {"a": "x", "b": "y"}},
        {"id": "S2", "labels": {"a": "x", "b": "y"}},
    ]

    result = compute_exact_match(extractions, ground_truths, fields=["a", "b"])

    assert result == {"exact_match_rate": 0.5, "n_exact_match": 1, "n_total": 2}


def test_compute_exact_match_skips_samples_with_error_fields():
    extractions = [{"id": "S1", "extraction": {"a": "ERROR", "b": "y"}}]
    ground_truths = [{"id": "S1", "labels": {"a": "x", "b": "y"}}]

    result = compute_exact_match(extractions, ground_truths, fields=["a", "b"])

    assert result == {"exact_match_rate": 0.0, "n_exact_match": 0, "n_total": 0}


def test_bootstrap_accuracy_ci_point_estimate_within_bounds():
    y_true = ["a", "b", "a", "b", "a", "b", "a", "b"]
    y_pred = ["a", "b", "a", "b", "a", "b", "b", "a"]  # 6/8 correct = 0.75

    ci = bootstrap_accuracy_ci(y_true, y_pred, n_iterations=200, seed=1)

    assert ci["accuracy"] == pytest.approx(0.75, abs=0.05)
    assert ci["ci_lower"] <= ci["accuracy"] <= ci["ci_upper"]


def test_bootstrap_accuracy_ci_empty_input():
    ci = bootstrap_accuracy_ci([], [])

    assert ci == {"accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}


def test_generate_evaluation_report_smoke(tiny_eval_config):
    extractions = [
        {"id": "S1", "extraction": {"affect_valence": "positive"}},
        {"id": "S2", "extraction": {"affect_valence": "negative"}},
    ]
    ground_truths = [
        {"id": "S1", "labels": {"affect_valence": "positive"}},
        {"id": "S2", "labels": {"affect_valence": "positive"}},
    ]

    report = generate_evaluation_report(extractions, ground_truths, tiny_eval_config)

    assert report["overall"]["accuracy_macro"] == 0.5
    assert report["exact_match"]["n_total"] == 2
    assert "affect_valence" in report["field_reports"]
