"""
Ground-truth annotation for the 3 extended monitoring fields — a
Chain-of-Thought (CoT) pass producing gold-standard labels/summaries for
the extended validation track.

Independent from mindlog.agent.annotator: the original 5-field annotation
pipeline is already-validated production logic and is not touched by this
module.
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

EXTENDED_ANNOTATION_SYSTEM_PROMPT = """\
You are an expert clinical psychologist annotating mental health conversation
data for a research benchmark. Your task is to produce gold-standard labels
for 3 extended structured fields.

Given a user's conversational self-report text, provide:
1. Chain-of-thought reasoning for EACH field (1-3 sentences explaining your
   judgment, citing specific phrases from the text).
2. The final label/summary for each field.

Respond with a valid JSON object containing:
{
  "reasoning": {
    "medication_adherence": "...",
    "somatic_symptoms": "...",
    "interpersonal_status": "..."
  },
  "labels": {
    "medication_adherence": "taken | missed | not_applicable | not_mentioned",
    "somatic_symptoms": "1-2 sentence summary, or exactly 'not_mentioned'",
    "interpersonal_status": "summary (see definitions below), or exactly 'not_mentioned'"
  }
}

Label definitions:

medication_adherence:
  - "taken": user mentions taking medication as prescribed
  - "missed": user mentions skipping, forgetting, or stopping medication
  - "not_applicable": user explicitly states they are not on medication
  - "not_mentioned": no medication information anywhere in the text

somatic_symptoms:
  - Summarize physical/bodily symptoms mentioned (heart racing,
    hyperventilating, sweating, trembling, dizziness, etc.) in 1-2 sentences.
  - If none are mentioned, the label must be EXACTLY "not_mentioned".

interpersonal_status:
  - Summarize anything about relationships (changes, conflict, support,
    withdrawal). This includes interactions with professionals such as
    counselors or doctors (e.g. reluctance to discuss something with them,
    trust issues) — but only the relational aspect of that interaction, not
    the medication content itself (whether taken, dosage, etc.).
  - Summarize not just what happened but, where possible, why it happened,
    the relationship's background/context, and the speaker's emotional
    nuance. Normally 1-2 sentences, but extend to 2-3 sentences if needed
    to capture this.
  - If nothing interpersonal is mentioned, the label must be EXACTLY
    "not_mentioned".

IMPORTANT:
- Annotate based ONLY on what the user says, not counselor/assistant responses.
- Your reasoning must cite specific phrases or patterns from the text.
- This ground-truth summary will later be compared against a separately
  extracted summary by an LLM judge — write it precisely and factually.
"""


def annotate_extended_single(
    client: OpenAI,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 1024,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Generate ground-truth labels/summaries with CoT reasoning for the 3
    extended fields. Returns {"reasoning": {...}, "labels": {...}}.
    """
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTENDED_ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            labels = parsed.get("labels", {})
            reasoning = parsed.get("reasoning", {})

            validated = {"medication_adherence": _validate_medication(labels)}
            for field in FREE_TEXT_FIELDS:
                validated[field] = _validate_free_text(labels, field)

            return {
                "reasoning": reasoning,
                "labels": validated,
            }

        except Exception as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "reasoning": {"_error": str(e)},
                    "labels": {
                        "medication_adherence": "ERROR",
                        "somatic_symptoms": "ERROR",
                        "interpersonal_status": "ERROR",
                    },
                }


def _validate_medication(labels: dict) -> str:
    value = str(labels.get("medication_adherence", "")).strip().lower().replace(" ", "_")
    if value not in MEDICATION_ADHERENCE_LABELS:
        raise ValueError(f"Invalid annotation label for medication_adherence: '{value}'")
    return value


def _validate_free_text(labels: dict, field: str) -> str:
    value = str(labels.get(field, "")).strip()
    if not value:
        raise ValueError(f"Empty annotation value for {field}")
    if value.lower() == NOT_MENTIONED:
        return NOT_MENTIONED
    if not looks_like_a_sentence(value):
        raise ValueError(f"Annotation for {field} too short: '{value}'")
    return value


def annotate_extended_batch(
    client: OpenAI,
    samples: list[dict],
    annotation_config: dict,
    batch_delay: float = 0.5,
    logger=None,
) -> list[dict]:
    """
    Generate extended ground-truth annotations with CoT reasoning for all
    samples. Returns a list of {id, reasoning, labels} dicts.
    """
    results = []

    for i, sample in enumerate(samples):
        annotation = annotate_extended_single(
            client=client,
            context=sample["context"],
            model=annotation_config.get("model", "gpt-4o"),
            temperature=annotation_config.get("temperature", 0),
            max_tokens=annotation_config.get("max_tokens", 1024),
            retry_attempts=annotation_config.get("retry_attempts", 3),
            retry_delay=annotation_config.get("retry_delay_seconds", 2.0),
        )

        results.append(
            {
                "id": sample["id"],
                "reasoning": annotation["reasoning"],
                "labels": annotation["labels"],
            }
        )

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Extended annotation progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(batch_delay)

    return results
