"""
CLI entry point — MindLog Extraction Pipeline Validation.

Usage:
    # Full pipeline (requires OPENAI_API_KEY in .env)
    python scripts/run_eval.py

    # Skip annotation if ground_truth_labels.json already exists
    python scripts/run_eval.py --skip-annotation

    # Skip extraction if extraction_results.json already exists
    python scripts/run_eval.py --skip-extraction

    # Evaluate only (both JSON files must exist)
    python scripts/run_eval.py --evaluate-only

    # Custom sample size
    python scripts/run_eval.py --sample-size 100

Output files:
    data/processed/sampled_conversations.json
    data/processed/ground_truth_labels.json
    data/processed/extraction_results.json
    artifacts/evaluation_report.json
    artifacts/evaluation_metrics.csv
"""

import argparse

from mindlog.pipeline.validation import run


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

    run(
        skip_annotation=args.skip_annotation,
        skip_extraction=args.skip_extraction,
        evaluate_only=args.evaluate_only,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
