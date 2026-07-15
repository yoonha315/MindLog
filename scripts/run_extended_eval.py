"""
CLI entry point — MindLog Extended-Field Validation Pipeline
(medication_adherence, somatic_symptoms, interpersonal_status).

Independent from scripts/run_eval.py: validates the 3 extended fields via a
separate sample set, ground-truth pass, extractor, and evaluator. Requires
scripts/sample_extended_validation.py to have been run first.

NOTE: unlike run_eval.py's --evaluate-only, evaluation here is NOT free —
somatic_symptoms/interpersonal_status are scored by an LLM judge
(judge_text_agreement), so every run of this script's evaluate step makes
real API calls even with --evaluate-only. Only annotation/extraction are
skippable from cache.

Usage:
    python scripts/run_extended_eval.py
    python scripts/run_extended_eval.py --skip-annotation
    python scripts/run_extended_eval.py --skip-extraction
    python scripts/run_extended_eval.py --evaluate-only

Output files:
    data/processed/extended_sampled_conversations.json  (from sample_extended_validation.py)
    data/processed/extended_ground_truth_labels.json
    data/processed/extended_extraction_results.json
    artifacts/evaluation_report_extended.json
"""

import argparse
import json
import os
import time

from mindlog.agent.client import build_client
from mindlog.agent.extended_annotator import annotate_extended_batch
from mindlog.agent.extended_extractor import extract_extended_batch
from mindlog.data.loaders import load_samples
from mindlog.pipeline.extended_evaluator import generate_extended_evaluation_report
from mindlog.utils.config_loader import load_config, resolve_path
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_extended_validation")


def _samples_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_sampled_conversations.json")


def _ground_truth_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_ground_truth_labels.json")


def _extraction_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_extraction_results.json")


def _report_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "results_dir"), "evaluation_report_extended.json")


def step_1_load_samples(cfg: dict) -> list[dict]:
    logger.info("=" * 60)
    logger.info("[Step 1] Load Extended Sample Set")
    logger.info("=" * 60)

    path = _samples_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No extended sample set at {path}. Run scripts/sample_extended_validation.py first."
        )

    samples = load_samples(path)
    logger.info(f"  Loaded {len(samples)} extended samples from {path}")
    return samples


def step_2_annotate(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 2] Generate Extended Ground Truth Annotations (CoT Pass)")
    logger.info("=" * 60)

    path = _ground_truth_path(cfg)

    if skip or os.path.exists(path):
        if os.path.exists(path):
            logger.info(f"  Found existing annotations: {path}")
            with open(path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            logger.info(f"  Loaded {len(annotations)} annotations from cache")
            return annotations
        raise FileNotFoundError(f"--skip-annotation set but no file at {path}")

    client = build_client()
    ann_cfg = cfg.get("annotation", {})
    llm_cfg = cfg.get("llm", {})

    t0 = time.time()
    annotations = annotate_extended_batch(
        client=client,
        samples=samples,
        annotation_config=ann_cfg,
        batch_delay=llm_cfg.get("batch_delay_seconds", 0.5),
        logger=logger,
    )
    logger.info(f"  Annotation complete: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved to: {path}")

    return annotations


def step_3_extract(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 3] Run Extended Extraction Pipeline")
    logger.info("=" * 60)

    path = _extraction_path(cfg)

    if skip or os.path.exists(path):
        if os.path.exists(path):
            logger.info(f"  Found existing results: {path}")
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"  Loaded {len(results)} results from cache")
            return results
        raise FileNotFoundError(f"--skip-extraction set but no file at {path}")

    client = build_client()
    llm_cfg = cfg.get("llm", {})

    t0 = time.time()
    results = extract_extended_batch(
        client=client,
        samples=samples,
        llm_config=llm_cfg,
        batch_delay=llm_cfg.get("batch_delay_seconds", 0.5),
        logger=logger,
    )
    logger.info(f"  Extraction complete: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved to: {path}")

    return results


def step_4_evaluate(cfg: dict, extractions: list[dict], ground_truths: list[dict]) -> dict:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 4] Evaluate Extended Fields (medication metrics + LLM judge)")
    logger.info("=" * 60)

    client = build_client()
    llm_cfg = cfg.get("llm", {})
    judge_config = {"model": llm_cfg.get("model", "gpt-4o"), "temperature": 0}

    report = generate_extended_evaluation_report(
        client, extractions, ground_truths, judge_config=judge_config, logger=logger
    )

    med = report["medication_adherence"]
    logger.info(
        f"  medication_adherence — accuracy: {med.get('accuracy')} kappa: {med.get('cohens_kappa')}"
    )
    for field in ("somatic_symptoms", "interpersonal_status"):
        pct = report[field]["percentages"]
        logger.info(
            f"  {field} — correct: {pct['correct']}% partial: {pct['partial']}% "
            f"incorrect: {pct['incorrect']}%"
        )

    return report


def step_5_export(cfg: dict, report: dict) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 5] Export Results")
    logger.info("=" * 60)

    results_dir = resolve_path(cfg, "results_dir")
    os.makedirs(results_dir, exist_ok=True)

    path = _report_path(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Report: {path}")


def run(
    skip_annotation: bool = False,
    skip_extraction: bool = False,
    evaluate_only: bool = False,
) -> dict:
    """Run the extended-field validation pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("MindLog Extended-Field Validation Pipeline")
    logger.info("=" * 60)

    cfg = load_config()
    t_start = time.time()

    samples = step_1_load_samples(cfg)
    annotations = step_2_annotate(cfg, samples, skip=skip_annotation or evaluate_only)
    extractions = step_3_extract(cfg, samples, skip=skip_extraction or evaluate_only)
    report = step_4_evaluate(cfg, extractions, annotations)
    step_5_export(cfg, report)

    elapsed = time.time() - t_start
    logger.info("")
    logger.info(f"Pipeline complete. Total time: {elapsed:.1f}s")
    logger.info("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="MindLog Extended-Field Validation Pipeline")
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation; load existing extended_ground_truth_labels.json",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction; load existing extended_extraction_results.json",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip annotation and extraction (both JSON files must exist); still calls the "
        "judge API for somatic_symptoms/interpersonal_status",
    )
    args = parser.parse_args()

    run(
        skip_annotation=args.skip_annotation,
        skip_extraction=args.skip_extraction,
        evaluate_only=args.evaluate_only,
    )


if __name__ == "__main__":
    main()
