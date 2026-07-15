"""
End-to-end orchestration of the extraction-pipeline validation experiment.

  Step 1: Load & sample conversations from the Kaggle dataset
  Step 2: Generate ground-truth annotations (LLM-as-judge, CoT pass)
  Step 3: Run extraction pipeline on the same samples
  Step 4: Evaluate extraction vs. ground truth
  Step 5: Export results (JSON report + CSV table + error analysis)

The CLI entry point lives in scripts/run_eval.py; this module holds the
reusable pipeline logic so it can also be called programmatically or from
notebooks.
"""

import json
import os
import time

from mindlog.agent.annotator import annotate_batch
from mindlog.agent.client import build_client
from mindlog.agent.extractor import extract_batch
from mindlog.data.loaders import (
    download_dataset,
    filter_by_length,
    load_raw_csv,
    load_samples,
    sample_conversations,
    save_samples,
)
from mindlog.pipeline.evaluator import format_results_table, generate_evaluation_report
from mindlog.utils.config_loader import load_config, resolve_output, resolve_path, resolve_result
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_validation")


def step_1_load_and_sample(cfg: dict, sample_size: int = None) -> list[dict]:
    """
    Load raw CSV, validate schema, filter by length, draw random sample.
    Reuses sampled_conversations.json from a prior run if present.
    """
    logger.info("=" * 60)
    logger.info("[Step 1] Load & Sample Dataset")
    logger.info("=" * 60)

    output_path = resolve_output(cfg, "sampled_conversations")

    if os.path.exists(output_path):
        logger.info(f"  Found existing samples: {output_path}")
        samples = load_samples(output_path)
        logger.info(f"  Loaded {len(samples)} samples from cache")
        return samples

    ds_cfg = cfg["dataset"]
    raw_dir = resolve_path(cfg, "raw_dir")
    csv_path = download_dataset(raw_dir, cfg["paths"]["raw_files"]["conversations"])

    logger.info(f"  Loading: {csv_path}")
    df = load_raw_csv(csv_path, ds_cfg["required_cols"])
    logger.info(f"  Raw rows: {len(df):,}")

    df = filter_by_length(
        df,
        text_col="Context",
        min_len=ds_cfg["min_context_length"],
        max_len=ds_cfg["max_context_length"],
    )
    logger.info(f"  After length filter: {len(df):,}")

    n = sample_size or ds_cfg["sample_size"]
    seed = ds_cfg["random_seed"]

    samples = sample_conversations(df, n=n, seed=seed)
    logger.info(f"  Sampled {len(samples)} conversations (seed={seed})")

    save_samples(samples, output_path)
    logger.info(f"  Saved to: {output_path}")

    return samples


def step_2_annotate(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    """
    Generate ground-truth labels using the LLM-as-judge approach with
    Chain-of-Thought reasoning — a separate, higher-fidelity pass from the
    extraction pipeline.
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
        raise FileNotFoundError(f"--skip-annotation set but no file at {output_path}")

    client = build_client()
    ann_cfg = cfg["annotation"]
    llm_cfg = cfg["llm"]

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    n_errors = sum(1 for a in annotations if any(v == "ERROR" for v in a["labels"].values()))

    logger.info(f"  Annotation complete: {elapsed:.1f}s")
    logger.info(f"  Errors: {n_errors}/{len(annotations)}")
    logger.info(f"  Saved to: {output_path}")

    return annotations


def step_3_extract(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    """Run the single-pass extraction pipeline on all samples (simulates production)."""
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
        raise FileNotFoundError(f"--skip-extraction set but no file at {output_path}")

    client = build_client()
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_errors = sum(1 for r in results if any(v == "ERROR" for v in r["extraction"].values()))

    logger.info(f"  Extraction complete: {elapsed:.1f}s")
    logger.info(f"  Errors: {n_errors}/{len(results)}")
    logger.info(f"  Saved to: {output_path}")

    return results


def step_4_evaluate(cfg: dict, extractions: list[dict], ground_truths: list[dict]) -> dict:
    """Compare extraction results against ground-truth annotations and print Table 1."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 4] Evaluate Extraction vs. Ground Truth")
    logger.info("=" * 60)

    report = generate_evaluation_report(extractions, ground_truths, cfg)
    df = format_results_table(report)

    logger.info("")
    logger.info("┌" + "─" * 78 + "┐")
    title = "Table 1. LLM Extraction Accuracy — Kaggle MH Conversations Benchmark"
    logger.info("│  " + title + " " * 5 + "│")
    logger.info("├" + "─" * 78 + "┤")

    header = (
        f"│ {'Field':<25} {'Accuracy':>8} {'Prec':>8} {'Recall':>8} " f"{'F1':>8} {'Kappa':>8}    │"
    )
    logger.info(header)
    logger.info("├" + "─" * 78 + "┤")

    for _, row in df.iterrows():
        prec = row["Precision (macro)"]
        rec = row["Recall (macro)"]
        f1 = row["F1 (macro)"]
        kap = row["Cohen's Kappa"]

        prec_s = f"{prec:.4f}" if isinstance(prec, float) else str(prec)
        rec_s = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
        f1_s = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        kap_s = f"{kap:.4f}" if isinstance(kap, float) else str(kap)

        acc = row["Accuracy"]
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)

        line = (
            f"│ {row['Field']:<25} {acc_s:>8} {prec_s:>8} {rec_s:>8} " f"{f1_s:>8} {kap_s:>8}    │"
        )
        logger.info(line)

    logger.info("└" + "─" * 78 + "┘")

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


def step_5_export(cfg: dict, report: dict, df_table) -> None:
    """Save the evaluation report (JSON) and metrics table (CSV) to the results directory."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 5] Export Results")
    logger.info("=" * 60)

    results_dir = resolve_path(cfg, "results_dir")
    os.makedirs(results_dir, exist_ok=True)

    report_path = resolve_result(cfg, "evaluation_report")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Report: {report_path}")

    csv_path = resolve_result(cfg, "evaluation_csv")
    df_table.to_csv(csv_path, index=False)
    logger.info(f"  Table:  {csv_path}")


def run(
    skip_annotation: bool = False,
    skip_extraction: bool = False,
    evaluate_only: bool = False,
    sample_size: int = None,
) -> dict:
    """Run the full validation pipeline end-to-end and return the evaluation report."""
    logger.info("=" * 60)
    logger.info("MindLog Extraction Pipeline — Preliminary Validation")
    logger.info("=" * 60)

    cfg = load_config()
    t_start = time.time()

    samples = step_1_load_and_sample(cfg, sample_size=sample_size)

    annotations = step_2_annotate(cfg, samples, skip=skip_annotation or evaluate_only)
    extractions = step_3_extract(cfg, samples, skip=skip_extraction or evaluate_only)

    report = step_4_evaluate(cfg, extractions, annotations)

    df_table = format_results_table(report)
    step_5_export(cfg, report, df_table)

    elapsed = time.time() - t_start
    logger.info("")
    logger.info(f"Pipeline complete. Total time: {elapsed:.1f}s")
    logger.info("=" * 60)

    return report
