"""
Re-evaluate interpersonal_status extraction against the enhanced prompt in
extended_extractor.py's EXTENDED_EXTRACTION_SYSTEM_PROMPT (professional-
interaction scope + background/context/emotional-nuance instructions).

Independent from scripts/run_extended_eval.py — this only re-runs extraction
and judging for interpersonal_status, on the subset of the original 66
stratified samples whose ground truth for that field is NOT "not_mentioned".
The cached ground truth (data/processed/extended_ground_truth_labels.json)
is reused as-is; annotation is not regenerated.

Usage:
    python scripts/reeval_interpersonal.py
    python scripts/reeval_interpersonal.py --skip-extraction

Output files:
    data/processed/reeval_interpersonal_extraction_results.json
    artifacts/reeval_interpersonal_report.json
"""

import argparse
import json
import os
import time

from mindlog.agent.client import build_client
from mindlog.agent.extended_annotator import annotate_extended_single
from mindlog.agent.extended_extractor import extract_extended_single
from mindlog.data.loaders import load_samples
from mindlog.pipeline.extended_evaluator import judge_text_agreement
from mindlog.utils.config_loader import load_config, resolve_path
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_reeval_interpersonal")

FIELD = "interpersonal_status"


def _samples_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_sampled_conversations.json")


def _ground_truth_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_ground_truth_labels.json")


def _extraction_path(cfg: dict) -> str:
    return os.path.join(
        resolve_path(cfg, "processed_dir"), "reeval_interpersonal_extraction_results.json"
    )


def _report_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "results_dir"), "reeval_interpersonal_report.json")


def _new_ground_truth_path(cfg: dict) -> str:
    """Re-annotated ground truth for the 46-sample subset — kept separate
    from data/processed/extended_ground_truth_labels.json, which is never
    overwritten by this script."""
    return os.path.join(
        resolve_path(cfg, "processed_dir"), "reeval_interpersonal_ground_truth.json"
    )


def _v2_report_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "results_dir"), "reeval_interpersonal_report_v2.json")


def load_mentioned_subset(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load the original 66 samples + cached ground truth, filtered down to
    samples whose interpersonal_status ground truth isn't not_mentioned."""
    samples_path = _samples_path(cfg)
    ground_truth_path = _ground_truth_path(cfg)

    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"No extended sample set at {samples_path}. "
            f"Run scripts/sample_extended_validation.py first."
        )
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(
            f"No cached ground truth at {ground_truth_path}. "
            f"Run scripts/run_extended_eval.py first to generate it."
        )

    samples = load_samples(samples_path)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truths = json.load(f)

    gt_map = {g["id"]: g["labels"] for g in ground_truths}
    mentioned_ids = {sid for sid, labels in gt_map.items() if labels.get(FIELD) != "not_mentioned"}

    filtered_samples = [s for s in samples if s["id"] in mentioned_ids]
    filtered_ground_truths = [g for g in ground_truths if g["id"] in mentioned_ids]

    return filtered_samples, filtered_ground_truths


def step_extract(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    logger.info("=" * 60)
    logger.info("[Step 1] Re-run Extraction (enhanced prompt)")
    logger.info("=" * 60)

    path = _extraction_path(cfg)

    if skip or os.path.exists(path):
        if os.path.exists(path):
            logger.info(f"  Found existing re-extraction: {path}")
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"  Loaded {len(results)} results from cache")
            return results
        raise FileNotFoundError(f"--skip-extraction set but no file at {path}")

    client = build_client()
    llm_cfg = cfg.get("llm", {})

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        extraction = extract_extended_single(
            client,
            context=sample["context"],
            model=llm_cfg.get("model", "gpt-4o"),
            temperature=llm_cfg.get("temperature", 0),
            max_tokens=llm_cfg.get("max_tokens", 512),
            retry_attempts=llm_cfg.get("retry_attempts", 3),
            retry_delay=llm_cfg.get("retry_delay_seconds", 2.0),
        )
        results.append({"id": sample["id"], "extraction": extraction})

        if (i + 1) % 10 == 0:
            logger.info(f"  Re-extraction progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(llm_cfg.get("batch_delay_seconds", 0.5))

    logger.info(f"  Re-extraction complete: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved to: {path}")

    return results


def step_reannotate(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    """Regenerate ground truth for the 46-sample subset with the enhanced
    interpersonal_status annotation prompt. Saved to a separate file —
    data/processed/extended_ground_truth_labels.json is never touched."""
    logger.info("=" * 60)
    logger.info("[Step 0] Re-annotate Ground Truth (enhanced prompt)")
    logger.info("=" * 60)

    path = _new_ground_truth_path(cfg)

    if skip or os.path.exists(path):
        if os.path.exists(path):
            logger.info(f"  Found existing re-annotation: {path}")
            with open(path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            logger.info(f"  Loaded {len(annotations)} annotations from cache")
            return annotations
        raise FileNotFoundError(f"--skip-reannotate set but no file at {path}")

    client = build_client()
    ann_cfg = cfg.get("annotation", {})
    llm_cfg = cfg.get("llm", {})

    annotations = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        annotation = annotate_extended_single(
            client,
            context=sample["context"],
            model=ann_cfg.get("model", "gpt-4o"),
            temperature=ann_cfg.get("temperature", 0),
            max_tokens=ann_cfg.get("max_tokens", 1024),
            retry_attempts=ann_cfg.get("retry_attempts", 3),
            retry_delay=ann_cfg.get("retry_delay_seconds", 2.0),
        )
        annotations.append(
            {
                "id": sample["id"],
                "reasoning": annotation["reasoning"],
                "labels": annotation["labels"],
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"  Re-annotation progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(llm_cfg.get("batch_delay_seconds", 0.5))

    logger.info(f"  Re-annotation complete: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved to: {path}")

    return annotations


def step_evaluate(cfg: dict, extractions: list[dict], ground_truths: list[dict]) -> dict:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 2] Judge New Extraction vs Cached Ground Truth")
    logger.info("=" * 60)

    client = build_client()
    llm_cfg = cfg.get("llm", {})
    judge_config = {"model": llm_cfg.get("model", "gpt-4o"), "temperature": 0}

    gt_map = {g["id"]: g["labels"] for g in ground_truths}
    cases = []

    for i, ext in enumerate(extractions):
        sid = ext["id"]
        if sid not in gt_map:
            continue

        extracted_text = ext["extraction"].get(FIELD, "ERROR")
        ground_truth_text = gt_map[sid].get(FIELD, "MISSING")

        if extracted_text == "ERROR":
            verdict = "incorrect"
        else:
            verdict = judge_text_agreement(
                client, extracted_text, ground_truth_text, FIELD, **judge_config
            )

        cases.append(
            {
                "id": sid,
                "extracted": extracted_text,
                "ground_truth": ground_truth_text,
                "verdict": verdict,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"  Judged: {i + 1}/{len(extractions)}")

    n = len(cases)
    counts = {"correct": 0, "partial": 0, "incorrect": 0}
    for case in cases:
        counts[case["verdict"]] = counts.get(case["verdict"], 0) + 1

    return {
        "field": FIELD,
        "n_samples": n,
        "counts": counts,
        "percentages": {k: round(100 * c / n, 1) if n > 0 else 0.0 for k, c in counts.items()},
        "cases": cases,
    }


def run(skip_extraction: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info("Re-evaluate interpersonal_status (enhanced prompt)")
    logger.info("=" * 60)

    cfg = load_config()

    samples, ground_truths = load_mentioned_subset(cfg)
    logger.info(f"  {len(samples)} samples with {FIELD} != not_mentioned (out of the original 66)")

    extractions = step_extract(cfg, samples, skip=skip_extraction)
    report = step_evaluate(cfg, extractions, ground_truths)

    logger.info("")
    pct = report["percentages"]
    logger.info(
        f"  {FIELD} (new prompt) — correct: {pct['correct']}% "
        f"partial: {pct['partial']}% incorrect: {pct['incorrect']}%"
    )

    report_path = _report_path(cfg)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Report: {report_path}")

    return report


def run_v2(skip_reannotate: bool = False) -> dict:
    """Re-annotate the 46-sample subset with the enhanced interpersonal_status
    annotation prompt, then re-judge the ALREADY-generated re-extraction
    (from run()/scripts/reeval_interpersonal.py) against this new ground
    truth. Extraction is NOT re-run — it's loaded from
    reeval_interpersonal_extraction_results.json, which must already exist.
    """
    logger.info("=" * 60)
    logger.info("Re-evaluate interpersonal_status vs NEW ground truth")
    logger.info("=" * 60)

    cfg = load_config()

    # Same 46 samples as run() — selection is based on the ORIGINAL ground
    # truth's interpersonal_status label, not the one we're about to generate.
    samples, _original_ground_truths = load_mentioned_subset(cfg)
    logger.info(f"  {len(samples)} samples with {FIELD} != not_mentioned (out of the original 66)")

    new_ground_truths = step_reannotate(cfg, samples, skip=skip_reannotate)

    extraction_path = _extraction_path(cfg)
    if not os.path.exists(extraction_path):
        raise FileNotFoundError(
            f"No existing re-extraction at {extraction_path}. "
            f"Run scripts/reeval_interpersonal.py (without --reannotate) first."
        )
    with open(extraction_path, "r", encoding="utf-8") as f:
        extractions = json.load(f)
    logger.info(
        f"  Reusing {len(extractions)} existing re-extraction results from {extraction_path}"
    )

    report = step_evaluate(cfg, extractions, new_ground_truths)

    logger.info("")
    pct = report["percentages"]
    logger.info(
        f"  {FIELD} (new prompt vs NEW ground truth) — correct: {pct['correct']}% "
        f"partial: {pct['partial']}% incorrect: {pct['incorrect']}%"
    )

    report_path = _v2_report_path(cfg)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate interpersonal_status with the enhanced extraction prompt"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip re-extraction; load existing reeval_interpersonal_extraction_results.json",
    )
    parser.add_argument(
        "--reannotate",
        action="store_true",
        help="Regenerate ground truth with the enhanced annotation prompt (saved separately to "
        "reeval_interpersonal_ground_truth.json), then re-judge the existing re-extraction "
        "against it instead of the original ground truth",
    )
    parser.add_argument(
        "--skip-reannotate",
        action="store_true",
        help="With --reannotate: skip regeneration, load existing "
        "reeval_interpersonal_ground_truth.json",
    )
    args = parser.parse_args()

    if args.reannotate:
        run_v2(skip_reannotate=args.skip_reannotate)
    else:
        run(skip_extraction=args.skip_extraction)


if __name__ == "__main__":
    main()
