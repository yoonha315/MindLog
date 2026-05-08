"""
================================================================================
02_scripts/01_run_validation.py — MindLog Extraction Pipeline Validation
================================================================================

[Purpose]
End-to-end orchestration of the preliminary validation experiment described
in MindLog Proposal §7.4. Executes the following pipeline:

  Step 1: Load & sample conversations from the Kaggle dataset
  Step 2: Generate ground-truth annotations (LLM-as-judge, CoT pass)
  Step 3: Run extraction pipeline on the same samples
  Step 4: Evaluate extraction vs. ground truth
  Step 5: Export results (JSON report + CSV table + error analysis)

[Usage]
    # Full pipeline (requires OPENAI_API_KEY in .env)
    python 02_scripts/01_run_validation.py

    # Skip annotation if ground_truth_labels.json already exists
    python 02_scripts/01_run_validation.py --skip-annotation

    # Skip extraction if extraction_results.json already exists
    python 02_scripts/01_run_validation.py --skip-extraction

    # Evaluate only (both JSON files must exist)
    python 02_scripts/01_run_validation.py --evaluate-only

    # Custom sample size
    python 02_scripts/01_run_validation.py --sample-size 100

[Output Files]
    00_data/01_processed/sampled_conversations.json
    00_data/01_processed/ground_truth_labels.json
    00_data/01_processed/extraction_results.json
    00_data/02_results/evaluation_report.json
    00_data/02_results/evaluation_metrics.csv

================================================================================
"""

import argparse
import json
import os
import sys
import time

# ── Path Setup (mirrors FLOW convention) ─────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "01_src", "00_common"))
sys.path.insert(0, os.path.join(ROOT, "01_src", "01_data"))
sys.path.insert(0, os.path.join(ROOT, "01_src", "02_extraction"))
sys.path.insert(0, os.path.join(ROOT, "01_src", "03_evaluation"))

from config_loader import load_config, resolve_path, resolve_output, resolve_result
from logger import get_logger
from loader import (
    download_dataset,
    load_raw_csv,
    filter_by_length,
    sample_conversations,
    save_samples,
    load_samples,
)
from extractor import build_client, extract_batch, annotate_batch
from evaluator import generate_evaluation_report, format_results_table

logger = get_logger("mindlog_validation")


# ============================================================================
# [Step 1] Load & Sample Dataset
# ============================================================================
def step_1_load_and_sample(cfg: dict, sample_size: int = None) -> list[dict]:
    """
    [Function Description]
    Load raw CSV, validate schema, filter by length, draw random sample.

    [Processing Flow]
    1. Check if sampled_conversations.json already exists → load it
    2. Otherwise: download CSV via kagglehub → filter → sample → save

    [Parameters]
    - cfg:          loaded config dict
    - sample_size:  override sample size (default: from config)

    [Returns]
    List of conversation sample dicts
    """
    logger.info("=" * 60)
    logger.info("[Step 1] Load & Sample Dataset")
    logger.info("=" * 60)

    output_path = resolve_output(cfg, "sampled_conversations")

    # Check for existing samples
    if os.path.exists(output_path):
        logger.info(f"  Found existing samples: {output_path}")
        samples = load_samples(output_path)
        logger.info(f"  Loaded {len(samples)} samples from cache")
        return samples

    # Load raw CSV (auto-download via kagglehub if not present)
    ds_cfg   = cfg["dataset"]
    raw_dir  = resolve_path(cfg, "raw_dir")
    csv_path = download_dataset(raw_dir, cfg["paths"]["raw_files"]["conversations"])

    logger.info(f"  Loading: {csv_path}")
    df = load_raw_csv(csv_path, ds_cfg["required_cols"])
    logger.info(f"  Raw rows: {len(df):,}")

    # Filter by length
    df = filter_by_length(
        df,
        text_col="Context",
        min_len=ds_cfg["min_context_length"],
        max_len=ds_cfg["max_context_length"],
    )
    logger.info(f"  After length filter: {len(df):,}")

    # Sample
    n = sample_size or ds_cfg["sample_size"]
    seed = ds_cfg["random_seed"]

    samples = sample_conversations(df, n=n, seed=seed)
    logger.info(f"  Sampled {len(samples)} conversations (seed={seed})")

    # Save
    save_samples(samples, output_path)
    logger.info(f"  Saved to: {output_path}")

    return samples


# ============================================================================
# [Step 2] Generate Ground Truth Annotations
# ============================================================================
def step_2_annotate(
    cfg: dict,
    samples: list[dict],
    skip: bool = False,
) -> list[dict]:
    """
    [Function Description]
    Generate ground-truth labels using the LLM-as-judge approach with
    Chain-of-Thought reasoning. This is a separate, higher-fidelity pass
    from the extraction pipeline.

    [Design Decision]
    The annotation prompt explicitly requires per-field reasoning before
    labeling, which reduces annotation noise compared to the extraction
    prompt (which optimizes for speed/cost). Inter-pass agreement between
    annotation and extraction serves as an additional quality signal.

    [Parameters]
    - cfg:     loaded config dict
    - samples: conversation samples from Step 1
    - skip:    if True, load existing annotations instead

    [Returns]
    List of ground truth dicts with reasoning and labels
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 2] Generate Ground Truth Annotations (CoT Pass)")
    logger.info("=" * 60)

    output_path = resolve_output(cfg, "ground_truth")

    if skip or os.path.exists(output_path):
        if os.path.exists(output_path):
            logger.info(f"  Found existing annotations: {output_path}")
            with open(output_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            logger.info(f"  Loaded {len(annotations)} annotations from cache")
            return annotations
        else:
            raise FileNotFoundError(
                f"--skip-annotation set but no file at {output_path}"
            )

    client   = build_client()
    ann_cfg  = cfg["annotation"]
    llm_cfg  = cfg["llm"]

    logger.info(f"  Model: {ann_cfg['model']} | Temp: {ann_cfg['temperature']}")
    logger.info(f"  CoT reasoning: {ann_cfg['require_reasoning']}")
    logger.info(f"  Samples: {len(samples)}")

    t0 = time.time()
    annotations = annotate_batch(
        client=client,
        samples=samples,
        annotation_config=ann_cfg,
        batch_delay=llm_cfg.get("batch_delay_seconds", 0.5),
        logger=logger,
    )
    elapsed = time.time() - t0

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Check for errors
    n_errors = sum(
        1 for a in annotations
        if any(v == "ERROR" for v in a["labels"].values())
    )

    logger.info(f"  Annotation complete: {elapsed:.1f}s")
    logger.info(f"  Errors: {n_errors}/{len(annotations)}")
    logger.info(f"  Saved to: {output_path}")

    return annotations


# ============================================================================
# [Step 3] Run Extraction Pipeline
# ============================================================================
def step_3_extract(
    cfg: dict,
    samples: list[dict],
    skip: bool = False,
) -> list[dict]:
    """
    [Function Description]
    Run the extraction pipeline (single-pass, no CoT) on all samples.
    This simulates the MindLog production extraction module.

    [Parameters]
    - cfg:     loaded config dict
    - samples: conversation samples from Step 1
    - skip:    if True, load existing extraction results

    [Returns]
    List of extraction result dicts
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 3] Run Extraction Pipeline")
    logger.info("=" * 60)

    output_path = resolve_output(cfg, "extraction_results")

    if skip or os.path.exists(output_path):
        if os.path.exists(output_path):
            logger.info(f"  Found existing results: {output_path}")
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"  Loaded {len(results)} results from cache")
            return results
        else:
            raise FileNotFoundError(
                f"--skip-extraction set but no file at {output_path}"
            )

    client  = build_client()
    llm_cfg = cfg["llm"]

    logger.info(f"  Model: {llm_cfg['model']} | Temp: {llm_cfg['temperature']}")
    logger.info(f"  Samples: {len(samples)}")

    t0 = time.time()
    results = extract_batch(
        client=client,
        samples=samples,
        llm_config=llm_cfg,
        batch_delay=llm_cfg.get("batch_delay_seconds", 0.5),
        logger=logger,
    )
    elapsed = time.time() - t0

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_errors = sum(
        1 for r in results
        if any(v == "ERROR" for v in r["extraction"].values())
    )

    logger.info(f"  Extraction complete: {elapsed:.1f}s")
    logger.info(f"  Errors: {n_errors}/{len(results)}")
    logger.info(f"  Saved to: {output_path}")

    return results


# ============================================================================
# [Step 4] Evaluate Extraction vs. Ground Truth
# ============================================================================
def step_4_evaluate(
    cfg: dict,
    extractions: list[dict],
    ground_truths: list[dict],
) -> dict:
    """
    [Function Description]
    Compare extraction results against ground-truth annotations.
    Computes per-field and aggregate metrics matching Proposal Table 1.

    [Processing Flow]
    1. Align results by sample ID
    2. Compute per-field: accuracy, precision, recall, F1, kappa
    3. Compute overall macro averages
    4. Compute exact match rate
    5. Compile error analysis
    6. Bootstrap confidence intervals

    [Parameters]
    - cfg:            loaded config dict
    - extractions:    results from Step 3
    - ground_truths:  annotations from Step 2

    [Returns]
    Complete evaluation report dict
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 4] Evaluate Extraction vs. Ground Truth")
    logger.info("=" * 60)

    report = generate_evaluation_report(extractions, ground_truths, cfg)

    # ── Display Results Table ─────────────────────────────────
    df = format_results_table(report)

    logger.info("")
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│  Table 1. LLM Extraction Accuracy — Kaggle MH Conversations Benchmark" + " " * 5 + "│")
    logger.info("├" + "─" * 78 + "┤")

    header = (
        f"│ {'Field':<25} {'Accuracy':>8} {'Prec':>8} {'Recall':>8} "
        f"{'F1':>8} {'Kappa':>8}    │"
    )
    logger.info(header)
    logger.info("├" + "─" * 78 + "┤")

    for _, row in df.iterrows():
        prec = row['Precision (macro)']
        rec  = row['Recall (macro)']
        f1   = row['F1 (macro)']
        kap  = row["Cohen's Kappa"]

        prec_s = f"{prec:.4f}" if isinstance(prec, float) else str(prec)
        rec_s  = f"{rec:.4f}"  if isinstance(rec, float)  else str(rec)
        f1_s   = f"{f1:.4f}"   if isinstance(f1, float)   else str(f1)
        kap_s  = f"{kap:.4f}"  if isinstance(kap, float)  else str(kap)

        acc = row['Accuracy']
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)

        line = (
            f"│ {row['Field']:<25} {acc_s:>8} {prec_s:>8} {rec_s:>8} "
            f"{f1_s:>8} {kap_s:>8}    │"
        )
        logger.info(line)

    logger.info("└" + "─" * 78 + "┘")

    # ── Error Analysis Summary ────────────────────────────────
    ea = report["error_analysis"]
    if ea["misclassification_patterns"]:
        logger.info("")
        logger.info("Error Analysis — Top Misclassification Patterns:")
        for field, patterns in ea["misclassification_patterns"].items():
            for p in patterns:
                logger.info(f"  {field}: {p['pattern']} (×{p['count']})")

    if ea["api_errors"] > 0:
        logger.info(f"  API Errors: {ea['api_errors']}")

    return report


# ============================================================================
# [Step 5] Export Results
# ============================================================================
def step_5_export(cfg: dict, report: dict, df_table) -> None:
    """
    [Function Description]
    Save evaluation report (JSON) and metrics table (CSV) to results directory.

    [Output Files]
    - evaluation_report.json:  complete report with per-class metrics,
                                confusion matrices, error analysis
    - evaluation_metrics.csv:   Table 1 in CSV format for paper inclusion
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 5] Export Results")
    logger.info("=" * 60)

    results_dir = resolve_path(cfg, "results_dir")
    os.makedirs(results_dir, exist_ok=True)

    # JSON report
    report_path = resolve_result(cfg, "evaluation_report")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Report: {report_path}")

    # CSV table
    csv_path = resolve_result(cfg, "evaluation_csv")
    df_table.to_csv(csv_path, index=False)
    logger.info(f"  Table:  {csv_path}")


# ============================================================================
# [Main] Pipeline Orchestrator
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MindLog Extraction Pipeline — Preliminary Validation"
    )
    parser.add_argument(
        "--skip-annotation", action="store_true",
        help="Skip annotation; load existing ground_truth_labels.json",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip extraction; load existing extraction_results.json",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Run evaluation only (both JSON files must exist)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Override sample size from config (default: 50)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MindLog Extraction Pipeline — Preliminary Validation")
    logger.info("=" * 60)

    cfg = load_config()
    t_start = time.time()

    # ── Step 1: Load & Sample ─────────────────────────────────
    samples = step_1_load_and_sample(cfg, sample_size=args.sample_size)

    # ── Step 2: Annotate ──────────────────────────────────────
    skip_ann = args.skip_annotation or args.evaluate_only
    annotations = step_2_annotate(cfg, samples, skip=skip_ann)

    # ── Step 3: Extract ───────────────────────────────────────
    skip_ext = args.skip_extraction or args.evaluate_only
    extractions = step_3_extract(cfg, samples, skip=skip_ext)

    # ── Step 4: Evaluate ──────────────────────────────────────
    report = step_4_evaluate(cfg, extractions, annotations)

    # ── Step 5: Export ────────────────────────────────────────
    df_table = format_results_table(report)
    step_5_export(cfg, report, df_table)

    elapsed = time.time() - t_start
    logger.info("")
    logger.info(f"Pipeline complete. Total time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()