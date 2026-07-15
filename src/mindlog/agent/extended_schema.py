"""Schema for the 3 extended monitoring fields — medication_adherence,
somatic_symptoms, interpersonal_status.

Independent from mindlog.agent.schema: the original 5-field extraction
pipeline (extractor.py, annotator.py, evaluator.py, prompts/) is
already-validated production logic and is not touched by this module.
This defines a second, separate schema track for 3 additional fields
under evaluation.
"""

NOT_MENTIONED = "not_mentioned"

MEDICATION_ADHERENCE_LABELS = ["taken", "missed", "not_applicable", NOT_MENTIONED]

FREE_TEXT_FIELDS = ("somatic_symptoms", "interpersonal_status")

EXTENDED_FIELDS = {
    "medication_adherence": {
        "type": "categorical",
        "labels": MEDICATION_ADHERENCE_LABELS,
        "description": "Whether the user mentioned taking, missing, or not needing medication",
    },
    "somatic_symptoms": {
        "type": "free_text",
        "description": (
            "1-2 sentence summary of physical symptoms mentioned (e.g. heart "
            "racing, hyperventilating, sweating, trembling, dizziness). "
            f"Exactly '{NOT_MENTIONED}' if none are mentioned."
        ),
    },
    "interpersonal_status": {
        "type": "free_text",
        "description": (
            "1-2 sentence summary of interpersonal topics mentioned (relationship "
            "changes, conflict, support, withdrawal). This includes interactions "
            "with professionals such as counselors or doctors (e.g. reluctance to "
            "discuss something with them, trust issues) — but only the relational "
            "aspect of that interaction, not the medication content itself (whether "
            "taken, dosage, etc.). Summarize not just what happened but, where "
            "possible, why it happened, the relationship's background/context, and "
            "the speaker's emotional nuance — extend to 2-3 sentences if needed. "
            f"Exactly '{NOT_MENTIONED}' if none are mentioned."
        ),
    },
}


def looks_like_a_sentence(text: str, min_words: int = 3) -> bool:
    """Cheap proxy for 'at least one real sentence' — avoids fragile NLP
    sentence-splitting by just checking for a minimum word count."""
    return len(text.split()) >= min_words
