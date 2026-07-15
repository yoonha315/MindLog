"""
Extraction for the 3 extended monitoring fields — medication_adherence,
somatic_symptoms, interpersonal_status.

Independent from mindlog.agent.extractor: the original 5-field extraction
pipeline is already-validated production logic and is not touched by this
module. This is a second, separate validation track for 3 additional
fields under evaluation.
"""

import json
import time

from openai import OpenAI

from mindlog.agent.extended_schema import (
    FREE_TEXT_FIELDS,
    MEDICATION_ADHERENCE_LABELS,
    NOT_MENTIONED,
    looks_like_a_sentence,
)

EXTENDED_EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical NLP extraction module for a mental health self-monitoring app.

Given a user's conversational self-report text, extract the following 3
structured fields. Respond ONLY with a valid JSON object — no markdown, no
preamble.

1. "medication_adherence": Whether the user mentioned taking, missing, or not
   needing medication.
   - "taken": user mentions taking medication as prescribed
   - "missed": user mentions skipping, forgetting, or stopping medication
   - "not_applicable": user explicitly states they are not on medication
   - "not_mentioned": no medication information anywhere in the text

2. "somatic_symptoms": A 1-2 sentence summary of any physical/bodily symptoms
   mentioned — e.g. heart racing, hyperventilating, sweating, trembling,
   dizziness. If no physical symptoms are mentioned, respond with EXACTLY the
   string "not_mentioned" (no other text).

3. "interpersonal_status": A 1-2 sentence summary of anything the user says
   about their relationships — changes, conflict, support, or withdrawal. If
   nothing interpersonal is mentioned, respond with EXACTLY the string
   "not_mentioned" (no other text).

IMPORTANT:
- Extract based ONLY on what the user says, not on what the assistant/counselor says.
- Return EXACTLY these 3 keys.
"""


def extract_extended_single(
    client: OpenAI,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 512,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Extract medication_adherence, somatic_symptoms, and interpersonal_status
    from a single conversation context in one API call.
    """
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTENDED_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            result = {"medication_adherence": _validate_medication(parsed)}
            for field in FREE_TEXT_FIELDS:
                result[field] = _validate_free_text(parsed, field)

            return result

        except Exception as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "medication_adherence": "ERROR",
                    "somatic_symptoms": "ERROR",
                    "interpersonal_status": "ERROR",
                    "_error": str(e),
                }


def _validate_medication(parsed: dict) -> str:
    value = str(parsed.get("medication_adherence", "")).strip().lower()
    if value not in MEDICATION_ADHERENCE_LABELS:
        # Attempt fuzzy match (e.g. "Not Mentioned" → "not_mentioned")
        value_clean = value.replace(" ", "_")
        if value_clean in MEDICATION_ADHERENCE_LABELS:
            return value_clean
        raise ValueError(
            f"Invalid label for medication_adherence: '{value}' "
            f"(allowed: {MEDICATION_ADHERENCE_LABELS})"
        )
    return value


def _validate_free_text(parsed: dict, field: str) -> str:
    value = str(parsed.get(field, "")).strip()
    if not value:
        raise ValueError(f"Empty value for {field}")
    if value.lower() == NOT_MENTIONED:
        return NOT_MENTIONED
    if not looks_like_a_sentence(value):
        raise ValueError(f"{field} value too short to be a real summary: '{value}'")
    return value


def extract_extended_batch(
    client: OpenAI,
    samples: list[dict],
    llm_config: dict,
    batch_delay: float = 0.5,
    logger=None,
) -> list[dict]:
    """
    Run extended extraction on a batch of conversation samples with
    rate-limit delays. Returns a list of {id, extraction} dicts.
    """
    results = []

    for i, sample in enumerate(samples):
        extraction = extract_extended_single(
            client=client,
            context=sample["context"],
            model=llm_config.get("model", "gpt-4o"),
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 512),
            retry_attempts=llm_config.get("retry_attempts", 3),
            retry_delay=llm_config.get("retry_delay_seconds", 2.0),
        )

        results.append(
            {
                "id": sample["id"],
                "extraction": extraction,
            }
        )

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Extended extraction progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(batch_delay)

    return results
