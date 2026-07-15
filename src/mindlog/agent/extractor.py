"""
Single-pass structured indicator extraction (production pipeline simulation).

Design decisions:
- Single-pass extraction: all 5 fields extracted in one API call to minimize
  cost and latency.
- Structured output via JSON mode: ensures machine-parseable results.
- Temperature = 0: deterministic extraction for benchmark reproducibility.
"""

import json
import time

from openai import OpenAI

from mindlog.agent.prompts.extraction_prompt import EXTRACTION_SYSTEM_PROMPT
from mindlog.agent.schema import VALID_LABELS


def extract_single(
    client: OpenAI,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 512,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Run the extraction pipeline on a single conversation context.
    Parses the LLM's JSON response into a validated dict, retrying on
    rate limits or malformed JSON.
    """
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            result = {}
            for field, allowed in VALID_LABELS.items():
                value = parsed.get(field, "").strip().lower()
                if value not in allowed:
                    # Attempt fuzzy match (e.g. "Not Mentioned" → "not_mentioned")
                    value_clean = value.replace(" ", "_")
                    if value_clean in allowed:
                        value = value_clean
                    else:
                        raise ValueError(
                            f"Invalid label for {field}: '{value}' "
                            f"(allowed: {allowed})"
                        )
                result[field] = value

            return result

        except Exception as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "affect_valence": "ERROR",
                    "energy_level": "ERROR",
                    "sleep_quality": "ERROR",
                    "dominant_theme": "ERROR",
                    "risk_indicators": "ERROR",
                    "_error": str(e),
                }


def extract_batch(
    client: OpenAI,
    samples: list[dict],
    llm_config: dict,
    batch_delay: float = 0.5,
    logger=None,
) -> list[dict]:
    """
    Run extraction on a batch of conversation samples with rate-limit delays.
    Returns a list of {id, extraction} dicts.
    """
    results = []

    for i, sample in enumerate(samples):
        extraction = extract_single(
            client=client,
            context=sample["context"],
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            retry_attempts=llm_config.get("retry_attempts", 3),
            retry_delay=llm_config.get("retry_delay_seconds", 2.0),
        )

        results.append({
            "id": sample["id"],
            "extraction": extraction,
        })

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Extraction progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(batch_delay)

    return results
