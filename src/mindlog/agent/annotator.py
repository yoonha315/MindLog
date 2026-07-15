"""
Ground-truth annotation — higher-fidelity Chain-of-Thought (CoT) labeling pass.

Design decision: using LLM-as-judge for ground truth is a deliberate
methodological choice. The annotation prompt requires explicit reasoning per
field, reducing the probability of arbitrary label assignment. Inter-pass
agreement (extraction vs. annotation) serves as a proxy for annotation quality.
"""

import json
import time

from openai import OpenAI

from mindlog.agent.prompts.annotation_prompt import ANNOTATION_SYSTEM_PROMPT
from mindlog.agent.schema import VALID_LABELS


def annotate_single(
    client: OpenAI,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 1024,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Generate ground-truth labels with Chain-of-Thought reasoning.

    Returns a dict with:
      - reasoning: {field: str} — CoT justification per field
      - labels:    {field: str} — ground truth labels
    """
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            labels = parsed.get("labels", {})
            reasoning = parsed.get("reasoning", {})

            validated_labels = {}
            for field, allowed in VALID_LABELS.items():
                value = labels.get(field, "").strip().lower().replace(" ", "_")
                if value not in allowed:
                    raise ValueError(
                        f"Invalid annotation label for {field}: '{value}'"
                    )
                validated_labels[field] = value

            return {
                "reasoning": reasoning,
                "labels": validated_labels,
            }

        except Exception as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "reasoning": {"_error": str(e)},
                    "labels": {
                        "affect_valence": "ERROR",
                        "energy_level": "ERROR",
                        "sleep_quality": "ERROR",
                        "dominant_theme": "ERROR",
                        "risk_indicators": "ERROR",
                    },
                }


def annotate_batch(
    client: OpenAI,
    samples: list[dict],
    annotation_config: dict,
    batch_delay: float = 0.5,
    logger=None,
) -> list[dict]:
    """
    Generate ground-truth annotations with CoT reasoning for all samples.
    Returns a list of {id, reasoning, labels} dicts.
    """
    results = []

    for i, sample in enumerate(samples):
        annotation = annotate_single(
            client=client,
            context=sample["context"],
            model=annotation_config["model"],
            temperature=annotation_config["temperature"],
            max_tokens=annotation_config["max_tokens"],
            retry_attempts=annotation_config.get("retry_attempts", 3),
            retry_delay=annotation_config.get("retry_delay_seconds", 2.0),
        )

        results.append({
            "id": sample["id"],
            "reasoning": annotation["reasoning"],
            "labels": annotation["labels"],
        })

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Annotation progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(batch_delay)

    return results
