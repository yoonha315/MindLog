"""Unit tests for mindlog.agent.extractor.extract_single — parsing, label
validation/fuzzy-matching, and retry/error-fallback behavior."""

import json

from mindlog.agent.extractor import extract_single


def test_extract_single_parses_valid_json(make_fake_client, valid_extraction_json):
    client = make_fake_client([valid_extraction_json])

    result = extract_single(client, context="I feel awful and can't sleep.", retry_attempts=1)

    assert result == {
        "affect_valence": "negative",
        "energy_level": "low",
        "sleep_quality": "poor",
        "dominant_theme": "emotional",
        "risk_indicators": "low",
    }
    assert client.chat.completions.calls == 1


def test_extract_single_fuzzy_matches_casing_and_spaces(make_fake_client):
    raw = json.dumps(
        {
            "affect_valence": "Negative",
            "energy_level": "Low",
            "sleep_quality": "Not Mentioned",
            "dominant_theme": "Emotional",
            "risk_indicators": "None",
        }
    )
    client = make_fake_client([raw])

    result = extract_single(client, context="...", retry_attempts=1)

    assert result["sleep_quality"] == "not_mentioned"
    assert result["risk_indicators"] == "none"


def test_extract_single_retries_on_malformed_json_then_succeeds(
    make_fake_client, valid_extraction_json
):
    client = make_fake_client(["not valid json", valid_extraction_json])

    result = extract_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["affect_valence"] == "negative"
    assert client.chat.completions.calls == 2


def test_extract_single_returns_error_dict_after_exhausting_retries(make_fake_client):
    client = make_fake_client(["not json", "still not json"])

    result = extract_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["affect_valence"] == "ERROR"
    assert result["risk_indicators"] == "ERROR"
    assert "_error" in result


def test_extract_single_rejects_invalid_label_value(make_fake_client):
    raw = json.dumps(
        {
            "affect_valence": "somewhat_ok",  # not an allowed label
            "energy_level": "low",
            "sleep_quality": "poor",
            "dominant_theme": "emotional",
            "risk_indicators": "low",
        }
    )
    client = make_fake_client([raw, raw])

    result = extract_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["affect_valence"] == "ERROR"


def test_extract_single_recovers_from_api_exception(make_fake_client, valid_extraction_json):
    client = make_fake_client([RuntimeError("rate limited"), valid_extraction_json])

    result = extract_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["affect_valence"] == "negative"
    assert client.chat.completions.calls == 2
