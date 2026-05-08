"""
================================================================================
01_src/02_extraction/extractor.py — LLM Structured Indicator Extraction
================================================================================

[Purpose]
Extract structured psychological indicators from naturalistic conversational
text using GPT-4o. This module implements the core extraction pipeline
described in MindLog Proposal §4.2.

[Extraction Fields]
1. Affect Valence   — positive / neutral / negative
2. Energy Level     — low / medium / high
3. Sleep Quality    — poor / fair / good / not_mentioned
4. Dominant Theme   — emotional / relationships / work_academic /
                      physical_health / existential / daily_routine / other
5. Risk Indicators  — none / low / moderate / high

[Design Decisions]
- Single-pass extraction: all 5 fields extracted in one API call to minimize
  cost and latency (vs. the proposal's two-pass architecture, which adds a
  secondary extraction pass — that is reserved for the production system).
- Structured Output via JSON mode: ensures machine-parseable results.
- Temperature = 0: deterministic extraction for benchmark reproducibility.

[Reference]
MindLog Proposal §4.2 — Structured Log Extraction
MindLog Proposal §7.4 — Preliminary Validation: Extraction Pipeline Accuracy

================================================================================
"""

import os
import json
import time
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================================
# [1] Structured Output Schema
# ============================================================================
class ConversationExtraction(BaseModel):
    """
    [Purpose]
    Pydantic schema for LLM-extracted structured indicators.
    Maps 1:1 to the extraction fields in config.yaml.

    [Fields]
    - affect_valence:  overall emotional tone
    - energy_level:    inferred activity/energy state
    - sleep_quality:   sleep information if mentioned
    - dominant_theme:  primary topic of the conversation
    - risk_indicators: presence of crisis/self-harm language
    """
    affect_valence:  str = Field(description="positive | neutral | negative")
    energy_level:    str = Field(description="low | medium | high")
    sleep_quality:   str = Field(description="poor | fair | good | not_mentioned")
    dominant_theme:  str = Field(
        description=(
            "emotional | relationships | work_academic | "
            "physical_health | existential | daily_routine | other"
        )
    )
    risk_indicators: str = Field(description="none | low | moderate | high")


# ============================================================================
# [2] Extraction System Prompt
# ============================================================================
EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical NLP extraction module for a mental health self-monitoring app.

Given a user's conversational self-report text, extract the following structured
indicators. Respond ONLY with a valid JSON object — no markdown, no preamble.

Fields to extract:
1. "affect_valence": The overall emotional tone of the text.
   - "positive": predominantly hopeful, happy, calm, grateful, content
   - "neutral": matter-of-fact, mixed, or emotionally flat
   - "negative": predominantly sad, anxious, angry, hopeless, distressed

2. "energy_level": The user's apparent energy or activity level.
   - "low": fatigued, lethargic, unmotivated, withdrawn, exhausted
   - "medium": functional but not notably energetic or depleted
   - "high": active, engaged, productive, restless, agitated

3. "sleep_quality": Sleep information if mentioned anywhere in the text.
   - "poor": trouble sleeping, insomnia, nightmares, oversleeping
   - "fair": adequate but not great sleep
   - "good": restful, sufficient sleep
   - "not_mentioned": no sleep information in the text

4. "dominant_theme": The single most prominent topic or concern.
   - "emotional": mood, anxiety, depression, emotional regulation
   - "relationships": family, friends, romantic partner, social isolation
   - "work_academic": job, school, career, financial stress from work
   - "physical_health": body, exercise, illness, medication, appetite
   - "existential": purpose, meaning, identity, self-worth, life direction
   - "daily_routine": chores, habits, schedules, logistics
   - "other": does not fit any of the above categories

5. "risk_indicators": Presence of self-harm, suicidal ideation, or crisis language.
   - "none": no risk language detected
   - "low": vague references to hopelessness without specific ideation
   - "moderate": expressions of wanting to disappear, passive ideation
   - "high": explicit self-harm or suicidal statements

IMPORTANT:
- Extract based ONLY on what the user says, not on what the counselor says.
- If a field is ambiguous, choose the most likely label based on context.
- Return EXACTLY these 5 keys with values from the allowed labels above.
"""


# ============================================================================
# [3] Ground Truth Annotation Prompt (Higher-Fidelity CoT Pass)
# ============================================================================
ANNOTATION_SYSTEM_PROMPT = """\
You are an expert clinical psychologist annotating mental health conversation
data for a research benchmark. Your task is to produce gold-standard labels
for structured indicator extraction.

Given a user's conversational self-report text, provide:
1. Chain-of-thought reasoning for EACH field (2-3 sentences explaining your
   judgment, citing specific phrases from the text).
2. The final label for each field.

Respond with a valid JSON object containing:
{
  "reasoning": {
    "affect_valence": "...",
    "energy_level": "...",
    "sleep_quality": "...",
    "dominant_theme": "...",
    "risk_indicators": "..."
  },
  "labels": {
    "affect_valence": "positive | neutral | negative",
    "energy_level": "low | medium | high",
    "sleep_quality": "poor | fair | good | not_mentioned",
    "dominant_theme": "emotional | relationships | work_academic | physical_health | existential | daily_routine | other",
    "risk_indicators": "none | low | moderate | high"
  }
}

Label definitions:

affect_valence:
  - "positive": predominantly hopeful, happy, calm, grateful, content
  - "neutral": matter-of-fact, mixed, or emotionally flat
  - "negative": predominantly sad, anxious, angry, hopeless, distressed

energy_level:
  - "low": fatigued, lethargic, unmotivated, withdrawn, exhausted
  - "medium": functional but not notably energetic or depleted
  - "high": active, engaged, productive, restless, agitated

sleep_quality:
  - "poor": trouble sleeping, insomnia, nightmares, oversleeping
  - "fair": adequate but not great sleep
  - "good": restful, sufficient sleep
  - "not_mentioned": no sleep information in the text

dominant_theme:
  - "emotional": mood, anxiety, depression, emotional regulation
  - "relationships": family, friends, romantic partner, social isolation
  - "work_academic": job, school, career, financial stress from work
  - "physical_health": body, exercise, illness, medication, appetite
  - "existential": purpose, meaning, identity, self-worth, life direction
  - "daily_routine": chores, habits, schedules, logistics
  - "other": does not fit any category above

risk_indicators:
  - "none": no risk language detected
  - "low": vague references to hopelessness without specific ideation
  - "moderate": expressions of wanting to disappear, passive ideation
  - "high": explicit self-harm or suicidal statements

IMPORTANT:
- Annotate based ONLY on what the user says, not on counselor responses.
- Your reasoning must cite specific phrases or patterns from the text.
- If genuinely ambiguous, state the ambiguity in reasoning and choose the
  most defensible label.
"""


# ============================================================================
# [4] OpenAI Client Initialization
# ============================================================================
def build_client() -> OpenAI:
    """
    [Function Description]
    Initialize the OpenAI client using the OPENAI_API_KEY environment variable.

    [Returns]
    Configured OpenAI client instance.

    [Raises]
    ValueError if OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Create a .env file in the project root with: OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


# ============================================================================
# [5] Single Conversation Extraction
# ============================================================================
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
    [Function Description]
    Run the extraction pipeline on a single conversation context.
    Parses the LLM's JSON response into a validated dict.

    [Processing Flow]
    1. Send context to GPT-4o with extraction system prompt
    2. Parse JSON response
    3. Validate all 5 fields are present with valid labels
    4. Retry on failure (rate limit, malformed JSON)

    [Parameters]
    - client:         OpenAI client instance
    - context:        user's conversational self-report text
    - model:          LLM model identifier
    - temperature:    sampling temperature (0 for deterministic)
    - max_tokens:     max response tokens
    - retry_attempts: number of retries on failure
    - retry_delay:    seconds between retries

    [Returns]
    Dict with keys: affect_valence, energy_level, sleep_quality,
                    dominant_theme, risk_indicators
    """
    valid_labels = {
        "affect_valence":  {"positive", "neutral", "negative"},
        "energy_level":    {"low", "medium", "high"},
        "sleep_quality":   {"poor", "fair", "good", "not_mentioned"},
        "dominant_theme":  {
            "emotional", "relationships", "work_academic",
            "physical_health", "existential", "daily_routine", "other",
        },
        "risk_indicators": {"none", "low", "moderate", "high"},
    }

    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": context},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # Validate all fields present and labels valid
            result = {}
            for field, allowed in valid_labels.items():
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
                    "affect_valence":  "ERROR",
                    "energy_level":    "ERROR",
                    "sleep_quality":   "ERROR",
                    "dominant_theme":  "ERROR",
                    "risk_indicators": "ERROR",
                    "_error": str(e),
                }


# ============================================================================
# [6] Ground Truth Annotation (CoT Pass)
# ============================================================================
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
    [Function Description]
    Generate ground-truth labels with Chain-of-Thought reasoning.
    This is a separate, higher-fidelity annotation pass that produces
    justified labels for benchmark evaluation.

    [Design Decision]
    Using LLM-as-judge for ground truth is a deliberate methodological
    choice documented in the proposal. The annotation prompt requires
    explicit reasoning per field, reducing the probability of arbitrary
    label assignment. This approach is validated by comparing inter-pass
    agreement (extraction vs. annotation) as a proxy for annotation quality.

    [Parameters]
    Same as extract_single, but uses the annotation prompt.

    [Returns]
    Dict with keys:
      - reasoning: {field: str} — CoT justification per field
      - labels:    {field: str} — ground truth labels
    """
    valid_labels = {
        "affect_valence":  {"positive", "neutral", "negative"},
        "energy_level":    {"low", "medium", "high"},
        "sleep_quality":   {"poor", "fair", "good", "not_mentioned"},
        "dominant_theme":  {
            "emotional", "relationships", "work_academic",
            "physical_health", "existential", "daily_routine", "other",
        },
        "risk_indicators": {"none", "low", "moderate", "high"},
    }

    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": context},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # Validate labels
            labels = parsed.get("labels", {})
            reasoning = parsed.get("reasoning", {})

            validated_labels = {}
            for field, allowed in valid_labels.items():
                value = labels.get(field, "").strip().lower().replace(" ", "_")
                if value not in allowed:
                    raise ValueError(
                        f"Invalid annotation label for {field}: '{value}'"
                    )
                validated_labels[field] = value

            return {
                "reasoning": reasoning,
                "labels":    validated_labels,
            }

        except Exception as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "reasoning": {"_error": str(e)},
                    "labels": {
                        "affect_valence":  "ERROR",
                        "energy_level":    "ERROR",
                        "sleep_quality":   "ERROR",
                        "dominant_theme":  "ERROR",
                        "risk_indicators": "ERROR",
                    },
                }


# ============================================================================
# [7] Batch Extraction
# ============================================================================
def extract_batch(
    client: OpenAI,
    samples: list[dict],
    llm_config: dict,
    batch_delay: float = 0.5,
    logger=None,
) -> list[dict]:
    """
    [Function Description]
    Run extraction on a batch of conversation samples with rate-limit delays.

    [Processing Flow]
    1. Iterate over samples
    2. Call extract_single per sample
    3. Log progress every 10 samples
    4. Delay between calls to respect API rate limits

    [Parameters]
    - client:      OpenAI client
    - samples:     list of conversation dicts (from loader.sample_conversations)
    - llm_config:  dict with model, temperature, max_tokens, retry_* keys
    - batch_delay: seconds between API calls
    - logger:      optional logger instance

    [Returns]
    List of dicts, each containing:
      - id:         sample ID
      - extraction: {field: label} dict from extract_single
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
            "id":         sample["id"],
            "extraction": extraction,
        })

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Extraction progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(batch_delay)

    return results


# ============================================================================
# [8] Batch Annotation (Ground Truth Generation)
# ============================================================================
def annotate_batch(
    client: OpenAI,
    samples: list[dict],
    annotation_config: dict,
    batch_delay: float = 0.5,
    logger=None,
) -> list[dict]:
    """
    [Function Description]
    Generate ground-truth annotations with CoT reasoning for all samples.

    [Parameters]
    - client:            OpenAI client
    - samples:           list of conversation dicts
    - annotation_config: dict with model, temperature, max_tokens keys
    - batch_delay:       seconds between API calls
    - logger:            optional logger instance

    [Returns]
    List of dicts, each containing:
      - id:        sample ID
      - reasoning: {field: str} CoT justification
      - labels:    {field: str} ground truth labels
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
            "id":        sample["id"],
            "reasoning": annotation["reasoning"],
            "labels":    annotation["labels"],
        })

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Annotation progress: {i + 1}/{len(samples)}")

        if i < len(samples) - 1:
            time.sleep(batch_delay)

    return results
