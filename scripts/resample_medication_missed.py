"""
Targeted re-sampling for the medication_adherence "missed" class.

The original 66-sample stratified set (scripts/sample_extended_validation.py)
turned out to contain zero true "missed" ground-truth instances, so
medication_adherence's per-class metrics for that label were never actually
exercised. This pulls a fresh batch specifically keyed to missed-dose
phrasing, excluding anything already in the original 66-sample set.

No API calls — pure CSV filtering/sampling.

Usage:
    python scripts/resample_medication_missed.py
    python scripts/resample_medication_missed.py --count 18

Output:
    data/processed/medication_missed_resample.json
"""

import argparse
import os
import random

from mindlog.data.loaders import load_raw_csv, load_samples, save_samples
from mindlog.utils.config_loader import load_config, resolve_path
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_resample_medication_missed")

# A row must mention BOTH a medication term AND a missed/stopped-taking
# action to match — bare action words like "forgot" or "went off" are far
# too generic on their own (they matched "forgot my phone", "forgot about
# me", "went off the straight and narrow" in earlier keyword-only attempts).
MEDICATION_TERMS = [
    "medication",
    "medicine",
    "pills",
    "pill",
    "meds",
    "dose",
    "dosage",
    "prescription",
]

ACTION_TERMS = [
    "forgot",
    "forget",
    "skip",
    "skipped",
    "missed",
    "stopped taking",
    "stop taking",
    "ran out",
    "quit",
    "off my",
    "without my",
    "not taking",
    "haven't taken",
    "haven't been taking",
    "didn't take",
    "no longer taking",
    "not been taking",
]

COUNT_RANGE = (15, 20)


def _matches_missed_dose(text: str) -> bool:
    """A medication term AND an action term must both be present — requiring
    co-occurrence instead of matching either list alone."""
    lowered = text.lower()
    has_medication_term = any(term in lowered for term in MEDICATION_TERMS)
    has_action_term = any(term in lowered for term in ACTION_TERMS)
    return has_medication_term and has_action_term


def _normalize_for_dedup(text: str) -> str:
    """Collapse all whitespace runs (including stray \\r from inconsistent
    line-ending scraping) to single spaces. This dataset has rows that are
    the same underlying conversation but differ only in whitespace, which a
    plain .strip() comparison does not catch."""
    return " ".join(text.split())


def build_candidate_pool(df, text_col: str = "Context") -> list:
    """Row indices whose Context matches the missed-dose co-occurrence rule,
    deduplicated to one row per unique (whitespace-normalized) Context text.
    This dataset has many rows sharing essentially identical Context —
    either paired with different Response text (multiple counselors
    answering the same question) or differing only by stray whitespace —
    which would otherwise inflate the pool with near-duplicate "candidates"
    that are really the same conversation.
    """
    seen_contexts = set()
    pool = []
    for idx, text in df[text_col].astype(str).items():
        if not _matches_missed_dose(text):
            continue
        normalized = _normalize_for_dedup(text)
        if normalized in seen_contexts:
            continue
        seen_contexts.add(normalized)
        pool.append(idx)
    return pool


def load_existing_contexts(cfg: dict) -> set:
    """Context text already used in the original 66-sample extended set —
    excluded so this resample only contains genuinely new conversations."""
    path = os.path.join(resolve_path(cfg, "processed_dir"), "extended_sampled_conversations.json")
    if not os.path.exists(path):
        return set()
    existing = load_samples(path)
    return {_normalize_for_dedup(s["context"]) for s in existing}


def resample_missed_candidates(
    df,
    pool: list,
    existing_contexts: set,
    seed: int,
    count_range: tuple = COUNT_RANGE,
    text_col: str = "Context",
    response_col: str = "Response",
) -> list[dict]:
    """Draw a reproducible random sample of missed-dose candidates,
    excluding any row whose context already appears in the original
    66-sample set."""
    rng = random.Random(seed)

    fresh_pool = [
        idx
        for idx in pool
        if _normalize_for_dedup(str(df.loc[idx, text_col])) not in existing_contexts
    ]

    if len(fresh_pool) < len(pool):
        logger.info(
            f"  Excluded {len(pool) - len(fresh_pool)} rows already present in the "
            f"original 66-sample set"
        )

    target = rng.randint(*count_range)
    n = min(target, len(fresh_pool))
    if n < target:
        logger.info(
            f"  Fresh candidate pool has only {len(fresh_pool)} rows; sampling all of "
            f"them (wanted {target})"
        )

    chosen = sorted(rng.sample(fresh_pool, n)) if fresh_pool else []

    samples = []
    for rank, idx in enumerate(chosen):
        row = df.loc[idx]
        samples.append(
            {
                "id": f"M{rank + 1:03d}",
                "context": str(row[text_col]).strip(),
                "response": str(row[response_col]).strip(),
            }
        )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Targeted re-sampling for medication_adherence 'missed' class"
    )
    parser.add_argument(
        "--count", type=int, default=None, help="Override sample count (default: random 15-20)"
    )
    args = parser.parse_args()

    cfg = load_config()
    ds_cfg = cfg["dataset"]

    raw_dir = resolve_path(cfg, "raw_dir")
    csv_path = os.path.join(raw_dir, cfg["paths"]["raw_files"]["conversations"])

    logger.info(f"Loading: {csv_path}")
    df = load_raw_csv(csv_path, ds_cfg["required_cols"])
    logger.info(f"Raw rows: {len(df):,}")

    pool = build_candidate_pool(df)
    logger.info(f"  Candidate pool (missed-dose keywords): {len(pool)} rows")

    existing_contexts = load_existing_contexts(cfg)
    logger.info(f"  Existing extended-sample contexts to exclude: {len(existing_contexts)}")

    count_range = (args.count, args.count) if args.count else COUNT_RANGE
    samples = resample_missed_candidates(
        df, pool, existing_contexts, seed=ds_cfg["random_seed"], count_range=count_range
    )
    logger.info(f"Sampled {len(samples)} new missed-dose candidates")

    output_path = os.path.join(
        resolve_path(cfg, "processed_dir"), "medication_missed_resample.json"
    )
    save_samples(samples, output_path)
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
