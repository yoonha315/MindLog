"""
Stratified keyword-based sampling for the extended-field validation track
(medication_adherence, somatic_symptoms, interpersonal_status).

Independent from the random sampling in mindlog.pipeline.validation: pulls
candidate rows by keyword category from the same raw CSV, then adds a
"no keywords at all" control group to validate the not_mentioned path.
No API calls — pure CSV filtering/sampling.

Usage:
    python scripts/sample_extended_validation.py
    python scripts/sample_extended_validation.py --none-count 10

Output:
    data/processed/extended_sampled_conversations.json
"""

import argparse
import os
import random

from mindlog.data.loaders import load_raw_csv, save_samples
from mindlog.utils.config_loader import load_config, resolve_path
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_extended_sampling")

KEYWORD_CATEGORIES = {
    "medication_adherence": ["medication", "pills", "dose", "prescription", "therapy"],
    "somatic_symptoms": ["chest", "heart", "breath", "dizzy", "shaking", "sweat"],
    "interpersonal_status": ["friend", "family", "partner", "relationship", "alone", "isolat"],
}

PER_CATEGORY_RANGE = (15, 20)


def _matches_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def build_candidate_pools(df, text_col: str = "Context") -> dict:
    """Row indices matching each keyword category, plus 'none' (matches nothing)."""
    pools = {category: [] for category in KEYWORD_CATEGORIES}
    none_pool = []

    for idx, text in df[text_col].astype(str).items():
        matched_any = False
        for category, keywords in KEYWORD_CATEGORIES.items():
            if _matches_any(text, keywords):
                pools[category].append(idx)
                matched_any = True
        if not matched_any:
            none_pool.append(idx)

    pools["none"] = none_pool
    return pools


def sample_extended_conversations(
    df,
    pools: dict,
    seed: int,
    per_category_range: tuple = PER_CATEGORY_RANGE,
    none_count: int = 10,
    text_col: str = "Context",
    response_col: str = "Response",
) -> list[dict]:
    """
    Draw a reproducible stratified sample: 15-20 rows per keyword category
    plus `none_count` rows with no keyword matches at all. Rows that match
    multiple categories are deduplicated (kept once, with all matched
    categories recorded).
    """
    rng = random.Random(seed)
    selected: dict = {}  # df index -> set of matched categories

    for category in KEYWORD_CATEGORIES:
        pool = pools[category]
        target = rng.randint(*per_category_range)
        n = min(target, len(pool))
        if n < target:
            logger.info(
                f"  '{category}' pool has only {len(pool)} rows; sampling all of them "
                f"(wanted {target})"
            )
        chosen = rng.sample(pool, n) if pool else []
        for idx in chosen:
            selected.setdefault(idx, set()).add(category)

    none_pool = pools["none"]
    n_none = min(none_count, len(none_pool))
    for idx in rng.sample(none_pool, n_none) if none_pool else []:
        selected.setdefault(idx, set()).add("none")

    samples = []
    for rank, idx in enumerate(sorted(selected.keys())):
        row = df.loc[idx]
        samples.append(
            {
                "id": f"E{rank + 1:03d}",
                "context": str(row[text_col]).strip(),
                "response": str(row[response_col]).strip(),
                "match_categories": sorted(selected[idx]),
            }
        )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Stratified keyword sampling for the extended-field validation track"
    )
    parser.add_argument(
        "--none-count", type=int, default=10, help="Number of no-keyword-match rows to include"
    )
    args = parser.parse_args()

    cfg = load_config()
    ds_cfg = cfg["dataset"]

    raw_dir = resolve_path(cfg, "raw_dir")
    csv_path = os.path.join(raw_dir, cfg["paths"]["raw_files"]["conversations"])

    logger.info(f"Loading: {csv_path}")
    df = load_raw_csv(csv_path, ds_cfg["required_cols"])
    logger.info(f"Raw rows: {len(df):,}")

    pools = build_candidate_pools(df)
    for category, pool in pools.items():
        logger.info(f"  Candidate pool '{category}': {len(pool)} rows")

    samples = sample_extended_conversations(
        df,
        pools,
        seed=ds_cfg["random_seed"],
        none_count=args.none_count,
    )
    logger.info(f"Sampled {len(samples)} unique conversations total")

    output_path = os.path.join(
        resolve_path(cfg, "processed_dir"), "extended_sampled_conversations.json"
    )
    save_samples(samples, output_path)
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
