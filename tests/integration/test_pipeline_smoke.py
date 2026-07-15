"""
Integration smoke test — exercises the real OpenAI API.

Skipped unless OPENAI_API_KEY is set, since it makes a real network call
and incurs API cost. Unit tests in tests/unit/ cover the extraction and
evaluation logic against mocked LLM responses without this dependency.
"""

import os

import pytest

from mindlog.agent.client import build_client
from mindlog.agent.extractor import extract_single

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live API integration test",
)


@requires_openai_key
def test_extract_single_against_real_api():
    client = build_client()

    result = extract_single(
        client,
        context="I've been feeling really down and haven't slept well in days.",
    )

    assert result["affect_valence"] != "ERROR"
    assert result["sleep_quality"] in {"poor", "fair", "good", "not_mentioned"}
