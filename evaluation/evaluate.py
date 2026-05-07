# ──────────────────────────────────────────────────────────────────
# MindLog — Extraction Accuracy Evaluation
# ──────────────────────────────────────────────────────────────────
# Compares LLM extraction results against ground truth labels.
# Outputs Table 1 for the research proposal.
#
# Metrics per field:
#   - Accuracy  (correct / total)
#   - Precision (macro-averaged across labels)
#   - Recall    (macro-averaged across labels)
#
# Usage: python evaluation/evaluate.py
# ──────────────────────────────────────────────────────────────────

import json
import os
from collections import defaultdict
from tabulate import tabulate

# ── File paths ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERSATIONS_PATH = os.path.join(SCRIPT_DIR, "synthetic_conversations.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "extraction_results.json")

# ── The 5 fields we are evaluating ──────────────────────────────
FIELDS = ["affect", "energy", "sleep_quality", "medication_taken", "dominant_theme"]


# ── Metric calculation ──────────────────────────────────────────
def compute_metrics(ground_truths, predictions):
    """
    Compute accuracy, macro precision, and macro recall
    for a single field across all samples.

    Args:
        ground_truths: list of correct labels   (e.g. ["positive", "negative", ...])
        predictions:   list of predicted labels  (e.g. ["positive", "neutral", ...])

    Returns:
        dict with accuracy, precision, recall
    """
    n = len(ground_truths)

    # ── Accuracy: simple % correct ──
    correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == pred)
    accuracy = correct / n

    # ── Collect all unique labels from both lists ──
    all_labels = sorted(set(ground_truths) | set(predictions))

    # ── Per-label precision and recall ──
    precisions = []
    recalls = []

    for label in all_labels:
        # True positives: predicted this label AND it was correct
        tp = sum(1 for gt, pred in zip(ground_truths, predictions)
                 if pred == label and gt == label)

        # False positives: predicted this label BUT it was wrong
        fp = sum(1 for gt, pred in zip(ground_truths, predictions)
                 if pred == label and gt != label)

        # False negatives: should have been this label BUT predicted something else
        fn = sum(1 for gt, pred in zip(ground_truths, predictions)
                 if gt == label and pred != label)

        # Precision = tp / (tp + fp), avoid division by zero
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)

        # Recall = tp / (tp + fn), avoid division by zero
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)

    # ── Macro average: mean across all labels ──
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall
    }


# ── Error analysis ──────────────────────────────────────────────
def print_errors(conversations, results):
    """
    Print every mismatch so we can see where the model struggles.
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS — Mismatched extractions")
    print("=" * 60)

    error_count = 0

    for conv, res in zip(conversations, results):
        if res["extracted"] is None:
            print(f"\n  [ID {conv['id']}] SKIPPED — extraction failed")
            error_count += 1
            continue

        gt = conv["ground_truth"]
        pred = res["extracted"]
        mismatches = []

        for field in FIELDS:
            gt_val = gt.get(field, "N/A")
            pred_val = pred.get(field, "N/A")
            if gt_val != pred_val:
                mismatches.append(f"    {field}: expected '{gt_val}' → got '{pred_val}'")

        if mismatches:
            error_count += 1
            # Show first 40 chars of conversation for context
            snippet = conv["conversation"][:40] + "..."
            print(f"\n  [ID {conv['id']}] \"{snippet}\"")
            for m in mismatches:
                print(m)

    if error_count == 0:
        print("\n  No errors — perfect extraction!")

    print()


# ── Main ────────────────────────────────────────────────────────
def main():
    # ── Load data ──
    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    # ── Filter out failed extractions ──
    valid_pairs = [
        (conv, res)
        for conv, res in zip(conversations, results)
        if res.get("extracted") is not None
    ]

    n_total = len(conversations)
    n_valid = len(valid_pairs)

    print()
    print("=" * 60)
    print("  MindLog — LLM Extraction Accuracy Benchmark")
    print("=" * 60)
    print(f"  Total conversations:  {n_total}")
    print(f"  Successfully parsed:  {n_valid}")
    print(f"  Failed:               {n_total - n_valid}")
    print()

    # ── Compute per-field metrics ──
    table_rows = []
    all_accuracies = []

    for field in FIELDS:
        # Collect ground truth and predictions for this field
        gts = [conv["ground_truth"][field] for conv, _ in valid_pairs]
        preds = [res["extracted"].get(field, "N/A") for _, res in valid_pairs]

        metrics = compute_metrics(gts, preds)
        all_accuracies.append(metrics["accuracy"])

        # Format field name for display
        display_name = field.replace("_", " ").title()

        table_rows.append([
            display_name,
            f"{metrics['accuracy']:.2f}",
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}"
        ])

    # ── Exact match: all 5 fields correct ──
    exact_matches = 0
    for conv, res in valid_pairs:
        if all(conv["ground_truth"][f] == res["extracted"].get(f) for f in FIELDS):
            exact_matches += 1

    exact_match_rate = exact_matches / n_valid if n_valid > 0 else 0.0

    # ── Overall macro averages ──
    overall_acc = sum(float(r[1]) for r in table_rows) / len(table_rows)
    overall_prec = sum(float(r[2]) for r in table_rows) / len(table_rows)
    overall_rec = sum(float(r[3]) for r in table_rows) / len(table_rows)

    # ── Add separator and summary rows ──
    table_rows.append(["─" * 18, "─" * 6, "─" * 6, "─" * 6])
    table_rows.append(["Overall (macro)", f"{overall_acc:.2f}", f"{overall_prec:.2f}", f"{overall_rec:.2f}"])
    table_rows.append(["Exact match (5/5)", f"{exact_match_rate:.2f}", "—", "—"])

    # ── Print the final table ──
    print(tabulate(
        table_rows,
        headers=["Field", "Accuracy", "Precision", "Recall"],
        tablefmt="simple_outline",
        colalign=("left", "center", "center", "center")
    ))
    print(f"\n  Exact matches: {exact_matches} / {n_valid}")
    print()

    # ── Print error details ──
    print_errors(conversations, results)


if __name__ == "__main__":
    main()