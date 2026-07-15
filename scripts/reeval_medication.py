"""
Re-evaluate medication_adherence with a small resample targeted at the
"missed" class (see scripts/resample_medication_missed.py) — the original
66-sample stratified set had zero true "missed" ground-truth instances, so
that label's per-class metrics were never actually exercised.

Independent from scripts/run_extended_eval.py and scripts/reeval_interpersonal.py:
only medication_adherence is evaluated here (extraction still returns
somatic_symptoms/interpersonal_status too, since extract_extended_single
always extracts all 3 fields in one call, but this script ignores them).
extended_extractor.py, extended_annotator.py, and extended_evaluator.py are
not modified — this script only imports and reuses their existing functions,
plus mindlog.pipeline.evaluator.compute_field_metrics directly (the same
reuse pattern extended_evaluator.py itself uses for medication_adherence).

Usage:
    python scripts/reeval_medication.py
    python scripts/reeval_medication.py --skip-annotation
    python scripts/reeval_medication.py --skip-extraction

Output files:
    data/processed/medication_missed_ground_truth.json
    data/processed/medication_missed_extraction_results.json
    artifacts/evaluation_report_medication_final.json
"""

import argparse
import json
import os
import time

from mindlog.agent.client import build_client
from mindlog.agent.extended_annotator import annotate_extended_single
from mindlog.agent.extended_extractor import extract_extended_single
from mindlog.agent.extended_schema import MEDICATION_ADHERENCE_LABELS
from mindlog.data.loaders import load_samples
from mindlog.pipeline.evaluator import compute_field_metrics
from mindlog.utils.config_loader import load_config, resolve_path
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_reeval_medication")

FIELD = "medication_adherence"


def _samples_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "medication_missed_resample.json")


def _ground_truth_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "medication_missed_ground_truth.json")


def _extraction_path(cfg: dict) -> str:
    return os.path.join(
        resolve_path(cfg, "processed_dir"), "medication_missed_extraction_results.json"
    )


def _original_extraction_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_extraction_results.json")


def _original_ground_truth_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "processed_dir"), "extended_ground_truth_labels.json")


def _report_path(cfg: dict) -> str:
    return os.path.join(resolve_path(cfg, "results_dir"), "evaluation_report_medication_final.json")


def load_new_samples(cfg: dict) -> list[dict]:
    path = _samples_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No medication-missed resample at {path}. "
            f"Run scripts/resample_medication_missed.py first."
        )
    return load_samples(path)


def step_annotate(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    logger.info("=" * 60)
    logger.info("[Step 1] Annotate Ground Truth (medication-missed resample)")
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

        if i < len(samples) - 1:
            time.sleep(llm_cfg.get("batch_delay_seconds", 0.5))

    logger.info(f"  Annotation complete: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved to: {path}")

    return annotations


def step_extract(cfg: dict, samples: list[dict], skip: bool = False) -> list[dict]:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 2] Extract (medication-missed resample)")
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

        if i < len(samples) - 1:
            time.sleep(llm_cfg.get("batch_delay_seconds", 0.5))

    logger.info(f"  Extraction complete: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved to: {path}")

    return results


def _field_arrays(
    extractions: list[dict], ground_truths: list[dict]
) -> tuple[list[str], list[str]]:
    gt_map = {g["id"]: g["labels"] for g in ground_truths}
    y_true, y_pred = [], []
    for ext in extractions:
        sid = ext["id"]
        if sid not in gt_map:
            continue
        y_true.append(gt_map[sid].get(FIELD, "MISSING"))
        y_pred.append(ext["extraction"].get(FIELD, "ERROR"))
    return y_true, y_pred


def step_evaluate(
    cfg: dict,
    new_extractions: list[dict],
    new_ground_truths: list[dict],
) -> dict:
    logger.info("")
    logger.info("=" * 60)
    logger.info("[Step 3] Combine with Original 66-Sample Results")
    logger.info("=" * 60)

    original_extraction_path = _original_extraction_path(cfg)
    original_ground_truth_path = _original_ground_truth_path(cfg)

    if not os.path.exists(original_extraction_path) or not os.path.exists(
        original_ground_truth_path
    ):
        raise FileNotFoundError(
            f"Original extended results not found at {original_extraction_path} / "
            f"{original_ground_truth_path}. Run scripts/run_extended_eval.py first."
        )

    with open(original_extraction_path, "r", encoding="utf-8") as f:
        original_extractions = json.load(f)
    with open(original_ground_truth_path, "r", encoding="utf-8") as f:
        original_ground_truths = json.load(f)

    original_y_true, original_y_pred = _field_arrays(original_extractions, original_ground_truths)
    new_y_true, new_y_pred = _field_arrays(new_extractions, new_ground_truths)

    logger.info(f"  Original set: {len(original_y_true)} samples")
    logger.info(f"  New (missed-targeted) set: {len(new_y_true)} samples")

    combined_y_true = original_y_true + new_y_true
    combined_y_pred = original_y_pred + new_y_pred

    report = {
        "field": FIELD,
        "combined": compute_field_metrics(
            combined_y_true, combined_y_pred, FIELD, MEDICATION_ADHERENCE_LABELS
        ),
        "original_only": compute_field_metrics(
            original_y_true, original_y_pred, FIELD, MEDICATION_ADHERENCE_LABELS
        ),
        "new_only": compute_field_metrics(
            new_y_true, new_y_pred, FIELD, MEDICATION_ADHERENCE_LABELS
        ),
        "metadata": {
            "n_original": len(original_y_true),
            "n_new": len(new_y_true),
            "n_combined": len(combined_y_true),
        },
    }

    return report


def run(skip_annotation: bool = False, skip_extraction: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info("Re-evaluate medication_adherence (missed-targeted resample)")
    logger.info("=" * 60)

    cfg = load_config()

    samples = load_new_samples(cfg)
    logger.info(f"  {len(samples)} medication-missed resample samples")

    ground_truths = step_annotate(cfg, samples, skip=skip_annotation)
    extractions = step_extract(cfg, samples, skip=skip_extraction)
    report = step_evaluate(cfg, extractions, ground_truths)

    logger.info("")
    combined = report["combined"]
    logger.info(
        f"  {FIELD} (combined, n={report['metadata']['n_combined']}) — "
        f"accuracy: {combined.get('accuracy')} kappa: {combined.get('cohens_kappa')}"
    )
    missed_support = combined.get("per_class", {}).get("missed", {}).get("support", 0)
    logger.info(f"  'missed' class support in combined set: {missed_support}")

    report_path = _report_path(cfg)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate medication_adherence with a missed-targeted resample"
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation; load existing medication_missed_ground_truth.json",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction; load existing medication_missed_extraction_results.json",
    )
    args = parser.parse_args()

    run(skip_annotation=args.skip_annotation, skip_extraction=args.skip_extraction)


if __name__ == "__main__":
    main()
