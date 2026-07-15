"""Shared pytest fixtures: fake OpenAI client + canned LLM responses."""

import json

import pytest


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Returns queued canned responses instead of calling the real API.

    Queue items that are Exception instances are raised instead of returned,
    so tests can simulate rate limits / API errors mid-sequence.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self._responses:
            raise AssertionError("mock LLM client ran out of queued responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class FakeOpenAIClient:
    """Stand-in for openai.OpenAI — same `.chat.completions.create(...)` shape."""

    def __init__(self, responses):
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _FakeCompletions(responses)


@pytest.fixture
def make_fake_client():
    """Factory: make_fake_client([resp1, resp2, ...]) -> FakeOpenAIClient."""

    def _make(responses):
        return FakeOpenAIClient(responses)

    return _make


@pytest.fixture
def valid_extraction_json():
    return json.dumps(
        {
            "affect_valence": "negative",
            "energy_level": "low",
            "sleep_quality": "poor",
            "dominant_theme": "emotional",
            "risk_indicators": "low",
        }
    )


@pytest.fixture
def valid_annotation_json():
    return json.dumps(
        {
            "reasoning": {
                "affect_valence": "User says 'I feel awful lately'.",
                "energy_level": "User says 'I can't get out of bed'.",
                "sleep_quality": "User mentions waking up at 3am.",
                "dominant_theme": "Primary concern is mood, not external events.",
                "risk_indicators": "No explicit self-harm or crisis language.",
            },
            "labels": {
                "affect_valence": "negative",
                "energy_level": "low",
                "sleep_quality": "poor",
                "dominant_theme": "emotional",
                "risk_indicators": "low",
            },
        }
    )


@pytest.fixture
def tiny_eval_config():
    """Minimal config.yaml-shaped dict for evaluator report tests."""
    return {
        "extraction": {
            "fields": {
                "affect_valence": {"labels": ["positive", "neutral", "negative"]},
            },
        },
        "llm": {"model": "gpt-4o"},
        "annotation": {"model": "gpt-4o"},
        "evaluation": {"bootstrap_iterations": 50, "confidence_interval": 0.95},
    }
