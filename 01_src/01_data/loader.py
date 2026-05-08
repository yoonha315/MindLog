"""
================================================================================
01_src/01_data/loader.py — Dataset Ingestion & Sampling
================================================================================

[Purpose]
Load the Kaggle NLP Mental Health Conversations dataset, validate schema,
filter by length constraints, and produce a stratified random sample of N
conversations for the extraction benchmark.

[Dataset]
Source: kaggle.com/datasets/thedevastator/nlp-mental-health-conversations
Format: CSV with columns [Context, Response]
- Context: user's conversational self-report (simulates MindLog daily log)
- Response: counselor/bot reply (not used for extraction, retained for context)

[Design Decision]
The "Context" column maps to MindLog's conversational daily log input.
Although this dataset contains counseling-style exchanges (not self-initiated
daily journaling), the user utterances exhibit the same naturalistic,
unstructured self-disclosure patterns that MindLog's extraction pipeline
must handle: affect expression, theme mention, energy cues, sleep references.

This makes it the best available proxy among the three candidate datasets:
- elvis23/mental-health-conversational-data → FAQ intent/pattern JSON;
  structured Q&A pairs, not naturalistic self-report
- abhishekjaiswal4896/mental-wellness-tracker → tabular survey scores;
  no conversational text to extract from
- thedevastator/nlp-mental-health-conversations → ✅ open-ended user text

================================================================================
"""

import os
import json
import glob
import random
import shutil
import pandas as pd

from typing import Optional


# ============================================================================
# [0] Kaggle Dataset Auto-Download
# ============================================================================
def download_dataset(raw_dir: str, filename: str = "combined_data.csv") -> str:
    """
    [Function Description]
    Download the Kaggle dataset via kagglehub if the CSV is not already
    present in raw_dir. Copies the file into the project's 00_data/00_raw/
    directory so subsequent runs skip the download.

    [Parameters]
    - raw_dir:   absolute path to 00_data/00_raw/
    - filename:  expected CSV filename (default: combined_data.csv)

    [Returns]
    Absolute path to the CSV file in raw_dir
    """
    csv_path = os.path.join(raw_dir, filename)

    if os.path.exists(csv_path):
        return csv_path

    import kagglehub

    downloaded_path = kagglehub.dataset_download(
        "thedevastator/nlp-mental-health-conversations"
    )

    # kagglehub returns a directory — find the CSV inside it
    candidates = glob.glob(
        os.path.join(downloaded_path, "**", filename), recursive=True
    )
    if not candidates:
        # fallback: grab any .csv in the downloaded dir
        candidates = glob.glob(
            os.path.join(downloaded_path, "**", "*.csv"), recursive=True
        )

    if not candidates:
        raise FileNotFoundError(
            f"kagglehub downloaded to {downloaded_path} but no CSV found inside."
        )

    os.makedirs(raw_dir, exist_ok=True)
    shutil.copy2(candidates[0], csv_path)

    return csv_path


# ============================================================================
# [1] CSV Loading & Validation
# ============================================================================
def load_raw_csv(csv_path: str, required_cols: list[str]) -> pd.DataFrame:
    """
    [Function Description]
    Load raw CSV and validate that required columns exist.

    [Parameters]
    - csv_path:      absolute path to the CSV file
    - required_cols: list of column names that must be present

    [Returns]
    pd.DataFrame with validated columns

    [Raises]
    FileNotFoundError if csv_path does not exist
    ValueError if required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def filter_by_length(
    df: pd.DataFrame,
    text_col: str,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    """
    [Function Description]
    Filter rows by character length of the text column.
    Removes trivially short entries (too little signal for extraction)
    and excessively long entries (unlikely in a 5-10 min daily log).

    [Parameters]
    - df:       input DataFrame
    - text_col: column to measure length on
    - min_len:  minimum character count (inclusive)
    - max_len:  maximum character count (inclusive)

    [Returns]
    Filtered DataFrame
    """
    lengths = df[text_col].astype(str).str.len()
    mask = (lengths >= min_len) & (lengths <= max_len)
    return df[mask].reset_index(drop=True)


def sample_conversations(
    df: pd.DataFrame,
    n: int,
    seed: int,
    text_col: str = "Context",
    response_col: str = "Response",
) -> list[dict]:
    """
    [Function Description]
    Draw a reproducible random sample of N conversations.
    Each conversation is stored as a dict with:
      - id:       zero-padded sample index (e.g. "S001")
      - context:  the user's self-report text
      - response: the counselor's response (retained for reference)

    [Parameters]
    - df:           filtered DataFrame
    - n:            number of samples to draw
    - seed:         random seed for reproducibility
    - text_col:     column containing user text
    - response_col: column containing response text

    [Returns]
    List of conversation dicts, sorted by id
    """
    if len(df) < n:
        raise ValueError(
            f"Requested {n} samples but only {len(df)} rows remain "
            f"after filtering. Lower sample_size or relax length filters."
        )

    random.seed(seed)
    indices = sorted(random.sample(range(len(df)), n))

    samples = []
    for rank, idx in enumerate(indices):
        row = df.iloc[idx]
        samples.append({
            "id":       f"S{rank + 1:03d}",
            "context":  str(row[text_col]).strip(),
            "response": str(row[response_col]).strip(),
        })

    return samples


def save_samples(samples: list[dict], output_path: str) -> None:
    """
    [Function Description]
    Persist sampled conversations as pretty-printed JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def load_samples(input_path: str) -> list[dict]:
    """
    [Function Description]
    Load previously saved conversation samples from JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)