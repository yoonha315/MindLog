"""
================================================================================
01_src/03_evaluation/evaluator.py — Extraction Pipeline Evaluation
================================================================================

[Purpose]
Compute classification metrics comparing LLM extraction results against
ground-truth annotations. Produces the evaluation table reported in
MindLog Proposal §7.4 (Table 1).

[Metrics]
Per field:
  - Accuracy
  - Precision (macro-averaged across label categories)
  - Recall    (macro-averaged across label categories)
  - F1        (macro-averaged across label categories)
  - Cohen's Kappa (inter-rater agreement)

Aggregate:
  - Overall accuracy (macro-averaged across fields)
  - Exact match rate (all 5 fields correct simultaneously)
  - 95% Bootstrap confidence intervals for accuracy

[Reference]
MindLog Proposal §7.4 — Preliminary Validation: Extraction Pipeline Accuracy

================================================================================
"""

import json
import os
import random
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Optional


# ============================================================================
# [1] Per-Field Classification Metrics
# ============================================================================
def compute_field_metrics(
    y_true: list[str],
    y_pred: list[str],
    field_name: str,
    valid_labels: list[str],
) -> dict:
    """
    [Function Description]
    Compute classification metrics for a single extraction field.

    [Processing Flow]
    1. Build confusion matrix from true/pred pairs
    2. Compute per-class precision, recall, F1
    3. Macro-average across classes
    4. Compute Cohen's Kappa for inter-rater agreement

    [Parameters]
    - y_true:       ground truth labels
    - y_pred:       predicted labels from extraction
    - field_name:   name of the field (for reporting)
    - valid_labels: list of all valid label categories

    [Returns]
    Dict with: accuracy, precision_macro, recall_macro, f1_macro,
               cohens_kappa, confusion_matrix, per_class_metrics
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No samples to evaluate"}

    # Filter out ERROR entries
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if p != "ERROR" and t != "ERROR"
    ]
    if not valid_pairs:
        return {"error": "All predictions are ERROR", "n_errors": n}

    y_t = [p[0] for p in valid_pairs]
    y_p = [p[1] for p in valid_pairs]
    n_valid = len(valid_pairs)
    n_errors = n - n_valid

    # ── Accuracy ──────────────────────────────────────────────
    correct = sum(1 for t, p in zip(y_t, y_p) if t == p)
    accuracy = correct / n_valid

    # ── Confusion Matrix ──────────────────────────────────────
    label_set = sorted(set(valid_labels) | set(y_t) | set(y_p))
    label_to_idx = {label: i for i, label in enumerate(label_set)}
    matrix = [[0] * len(label_set) for _ in range(len(label_set))]

    for t, p in zip(y_t, y_p):
        matrix[label_to_idx[t]][label_to_idx[p]] += 1

    # ── Per-Class Precision / Recall / F1 ─────────────────────
    per_class = {}
    precisions, recalls, f1s = [], [], []

    for label in label_set:
        idx = label_to_idx[label]
        tp = matrix[idx][idx]
        fp = sum(matrix[r][idx] for r in range(len(label_set))) - tp
        fn = sum(matrix[idx][c] for c in range(len(label_set))) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   sum(matrix[idx]),
        }

        # Only include in macro average if the class has support
        if sum(matrix[idx]) > 0 or sum(matrix[r][idx] for r in range(len(label_set))) > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    precision_macro = np.mean(precisions) if precisions else 0.0
    recall_macro    = np.mean(recalls)    if recalls    else 0.0
    f1_macro        = np.mean(f1s)        if f1s        else 0.0

    # ── Cohen's Kappa ─────────────────────────────────────────
    kappa = _cohens_kappa(y_t, y_p, label_set)

    return {
        "field":            field_name,
        "n_samples":        n_valid,
        "n_errors":         n_errors,
        "accuracy":         round(accuracy, 4),
        "precision_macro":  round(precision_macro, 4),
        "recall_macro":     round(recall_macro, 4),
        "f1_macro":         round(f1_macro, 4),
        "cohens_kappa":     round(kappa, 4),
        "per_class":        per_class,
        "confusion_matrix": {
            "labels": label_set,
            "matrix": matrix,
        },
    }


def _cohens_kappa(y_true: list, y_pred: list, labels: list) -> float:
    """
    [Function Description]
    Compute Cohen's Kappa — a measure of inter-rater agreement that accounts
    for chance agreement. Used here to evaluate extraction vs. annotation
    agreement beyond what accuracy alone captures.

    [Formula]
    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance

    [Returns]
    Kappa value in [-1, 1]. 1 = perfect agreement, 0 = chance agreement.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Count per-class frequencies for each rater
    freq_true = [0] * k
    freq_pred = [0] * k
    agree = 0

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t, -1)
        pi = label_to_idx.get(p, -1)
        if ti >= 0:
            freq_true[ti] += 1
        if pi >= 0:
            freq_pred[pi] += 1
        if t == p:
            agree += 1

    p_o = agree / n
    p_e = sum(ft * fp for ft, fp in zip(freq_true, freq_pred)) / (n * n)

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


# ============================================================================
# [2] Exact Match Rate
# ============================================================================
def compute_exact_match(
    extractions: list[dict],
    ground_truths: list[dict],
    fields: list[str],
) -> dict:
    """
    [Function Description]
    Compute the exact-match rate: proportion of samples where ALL fields
    are simultaneously correct.

    [Parameters]
    - extractions:    list of {id, extraction: {field: label}}
    - ground_truths:  list of {id, labels: {field: label}}
    - fields:         list of field names to compare

    [Returns]
    Dict with: exact_match_rate, n_exact_match, n_total
    """
    gt_map = {g["id"]: g["labels"] for g in ground_truths}
    n_match = 0
    n_total = 0

    for ext in extractions:
        sid = ext["id"]
        if sid not in gt_map:
            continue

        gt_labels  = gt_map[sid]
        ext_labels = ext["extraction"]

        # Skip if any field has ERROR
        if any(ext_labels.get(f) == "ERROR" for f in fields):
            continue

        n_total += 1
        if all(ext_labels.get(f) == gt_labels.get(f) for f in fields):
            n_match += 1

    return {
        "exact_match_rate": round(n_match / n_total, 4) if n_total > 0 else 0.0,
        "n_exact_match":    n_match,
        "n_total":          n_total,
    }


# ============================================================================
# [3] Bootstrap Confidence Intervals
# ============================================================================
def bootstrap_accuracy_ci(
    y_true: list[str],
    y_pred: list[str],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    [Function Description]
    Compute bootstrap confidence intervals for accuracy.
    Non-parametric approach — resamples with replacement.

    [Parameters]
    - y_true:        ground truth labels
    - y_pred:        predicted labels
    - n_iterations:  number of bootstrap resamples
    - confidence:    confidence level (e.g. 0.95)
    - seed:          random seed

    [Returns]
    Dict with: accuracy, ci_lower, ci_upper, n_iterations
    """
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    accuracies = []
    for _ in range(n_iterations):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        correct = sum(1 for i in indices if y_true[i] == y_pred[i])
        accuracies.append(correct / n)

    accuracies.sort()
    alpha = 1 - confidence
    lo = int((alpha / 2) * n_iterations)
    hi = int((1 - alpha / 2) * n_iterations) - 1

    return {
        "accuracy":     round(np.mean(accuracies), 4),
        "ci_lower":     round(accuracies[lo], 4),
        "ci_upper":     round(accuracies[hi], 4),
        "n_iterations": n_iterations,
    }


# ============================================================================
# [4] Full Evaluation Report
# ============================================================================
def generate_evaluation_report(
    extractions: list[dict],
    ground_truths: list[dict],
    config: dict,
) -> dict:
    """
    [Function Description]
    Generate the complete evaluation report matching Proposal Table 1 format.

    [Processing Flow]
    1. Align extraction results with ground truth by sample ID
    2. Compute per-field metrics (accuracy, precision, recall, F1, kappa)
    3. Compute overall macro-averaged metrics
    4. Compute exact match rate
    5. Compute bootstrap CIs for per-field accuracy
    6. Compile error analysis

    [Parameters]
    - extractions:   list of {id, extraction: {field: label}}
    - ground_truths: list of {id, labels: {field: label}, reasoning: {...}}
    - config:        loaded config.yaml

    [Returns]
    Complete evaluation report dict
    """
    extraction_cfg = config["extraction"]
    eval_cfg       = config["evaluation"]
    fields         = list(extraction_cfg["fields"].keys())

    # ── Align by sample ID ────────────────────────────────────
    gt_map = {g["id"]: g["labels"] for g in ground_truths}

    # ── Per-Field Metrics ─────────────────────────────────────
    field_reports = {}
    overall_accuracies = []

    for field in fields:
        valid_labels = extraction_cfg["fields"][field]["labels"]

        y_true = []
        y_pred = []

        for ext in extractions:
            sid = ext["id"]
            if sid not in gt_map:
                continue
            y_true.append(gt_map[sid].get(field, "MISSING"))
            y_pred.append(ext["extraction"].get(field, "ERROR"))

        metrics = compute_field_metrics(y_true, y_pred, field, valid_labels)

        # Bootstrap CI for this field's accuracy
        valid_yt = [t for t, p in zip(y_true, y_pred) if p != "ERROR" and t != "ERROR"]
        valid_yp = [p for t, p in zip(y_true, y_pred) if p != "ERROR" and t != "ERROR"]

        if valid_yt:
            ci = bootstrap_accuracy_ci(
                valid_yt, valid_yp,
                n_iterations=eval_cfg.get("bootstrap_iterations", 1000),
                confidence=eval_cfg.get("confidence_interval", 0.95),
            )
            metrics["bootstrap_ci"] = ci

        field_reports[field] = metrics
        if "accuracy" in metrics:
            overall_accuracies.append(metrics["accuracy"])

    # ── Overall Metrics ───────────────────────────────────────
    overall = {
        "accuracy_macro": round(np.mean(overall_accuracies), 4) if overall_accuracies else 0.0,
        "precision_macro": round(
            np.mean([r["precision_macro"] for r in field_reports.values() if "precision_macro" in r]),
            4,
        ),
        "recall_macro": round(
            np.mean([r["recall_macro"] for r in field_reports.values() if "recall_macro" in r]),
            4,
        ),
        "f1_macro": round(
            np.mean([r["f1_macro"] for r in field_reports.values() if "f1_macro" in r]),
            4,
        ),
    }

    # ── Exact Match ───────────────────────────────────────────
    exact_match = compute_exact_match(extractions, ground_truths, fields)

    # ── Error Analysis ────────────────────────────────────────
    error_analysis = _compile_error_analysis(
        extractions, ground_truths, fields
    )

    return {
        "field_reports":  field_reports,
        "overall":        overall,
        "exact_match":    exact_match,
        "error_analysis": error_analysis,
        "metadata": {
            "n_samples":       len(extractions),
            "n_fields":        len(fields),
            "model":           config["llm"]["model"],
            "annotation_model": config["annotation"]["model"],
        },
    }


def _compile_error_analysis(
    extractions: list[dict],
    ground_truths: list[dict],
    fields: list[str],
) -> dict:
    """
    [Function Description]
    Compile a structured error analysis identifying:
    - Most common misclassification patterns per field
    - Samples with the most field-level errors
    - API error count

    [Returns]
    Dict with: misclassification_patterns, high_error_samples, api_errors
    """
    gt_map  = {g["id"]: g["labels"] for g in ground_truths}
    gt_reas = {g["id"]: g.get("reasoning", {}) for g in ground_truths}

    patterns = defaultdict(lambda: defaultdict(int))
    sample_errors = {}
    api_errors = 0

    for ext in extractions:
        sid = ext["id"]
        if sid not in gt_map:
            continue

        n_errors = 0
        for field in fields:
            pred = ext["extraction"].get(field, "ERROR")
            true = gt_map[sid].get(field, "MISSING")

            if pred == "ERROR":
                api_errors += 1
                n_errors += 1
                continue

            if pred != true:
                patterns[field][f"{true} → {pred}"] += 1
                n_errors += 1

        if n_errors > 0:
            sample_errors[sid] = n_errors

    # Top 3 misclassification patterns per field
    top_patterns = {}
    for field, pats in patterns.items():
        sorted_pats = sorted(pats.items(), key=lambda x: -x[1])[:3]
        top_patterns[field] = [
            {"pattern": p, "count": c} for p, c in sorted_pats
        ]

    # Top 5 highest-error samples
    high_error = sorted(sample_errors.items(), key=lambda x: -x[1])[:5]

    return {
        "misclassification_patterns": top_patterns,
        "high_error_samples":         [
            {"id": sid, "n_errors": n} for sid, n in high_error
        ],
        "api_errors":                 api_errors,
    }


# ============================================================================
# [5] Results Table (Proposal Table 1 Format)
# ============================================================================
def format_results_table(report: dict) -> pd.DataFrame:
    """
    [Function Description]
    Format evaluation results as a DataFrame matching Proposal Table 1.

    [Table Columns]
    Field | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Kappa

    [Returns]
    pd.DataFrame ready for display or export
    """
    rows = []

    for field, metrics in report["field_reports"].items():
        field_display = field.replace("_", " ").title()
        rows.append({
            "Field":              field_display,
            "Accuracy":           metrics.get("accuracy", 0),
            "Precision (macro)":  metrics.get("precision_macro", 0),
            "Recall (macro)":     metrics.get("recall_macro", 0),
            "F1 (macro)":         metrics.get("f1_macro", 0),
            "Cohen's Kappa":      metrics.get("cohens_kappa", 0),
        })

    # Overall row
    rows.append({
        "Field":              "Overall (macro avg)",
        "Accuracy":           report["overall"]["accuracy_macro"],
        "Precision (macro)":  report["overall"]["precision_macro"],
        "Recall (macro)":     report["overall"]["recall_macro"],
        "F1 (macro)":         report["overall"]["f1_macro"],
        "Cohen's Kappa":      round(
            np.mean([
                m.get("cohens_kappa", 0)
                for m in report["field_reports"].values()
            ]),
            4,
        ),
    })

    # Exact Match row
    rows.append({
        "Field":              "Exact Match (all fields)",
        "Accuracy":           report["exact_match"]["exact_match_rate"],
        "Precision (macro)":  "—",
        "Recall (macro)":     "—",
        "F1 (macro)":         "—",
        "Cohen's Kappa":      "—",
    })

    return pd.DataFrame(rows)
