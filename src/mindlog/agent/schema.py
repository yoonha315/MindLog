"""Pydantic schema for LLM-extracted structured indicators."""

from pydantic import BaseModel, Field

VALID_LABELS = {
    "affect_valence": {"positive", "neutral", "negative"},
    "energy_level": {"low", "medium", "high"},
    "sleep_quality": {"poor", "fair", "good", "not_mentioned"},
    "dominant_theme": {
        "emotional", "relationships", "work_academic",
        "physical_health", "existential", "daily_routine", "other",
    },
    "risk_indicators": {"none", "low", "moderate", "high"},
}


class ConversationExtraction(BaseModel):
    """Maps 1:1 to the extraction fields in configs/config.yaml."""

    affect_valence: str = Field(description="positive | neutral | negative")
    energy_level: str = Field(description="low | medium | high")
    sleep_quality: str = Field(description="poor | fair | good | not_mentioned")
    dominant_theme: str = Field(
        description=(
            "emotional | relationships | work_academic | "
            "physical_health | existential | daily_routine | other"
        )
    )
    risk_indicators: str = Field(description="none | low | moderate | high")
