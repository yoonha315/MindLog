"""Unit tests for mindlog.agent.extended_extractor.extract_extended_single —
parsing, validation, and retry/error-fallback behavior for the 3 extended
monitoring fields."""

import json

from mindlog.agent.extended_extractor import extract_extended_single


def _valid_json(
    medication="taken",
    somatic="Heart racing and shortness of breath.",
    interpersonal="Conflict with a close friend.",
):
    return json.dumps(
        {
            "medication_adherence": medication,
            "somatic_symptoms": somatic,
            "interpersonal_status": interpersonal,
        }
    )


def test_extract_extended_single_parses_valid_json(make_fake_client):
    client = make_fake_client([_valid_json()])

    result = extract_extended_single(client, context="...", retry_attempts=1)

    assert result == {
        "medication_adherence": "taken",
        "somatic_symptoms": "Heart racing and shortness of breath.",
        "interpersonal_status": "Conflict with a close friend.",
    }


def test_extract_extended_single_accepts_not_mentioned_free_text(make_fake_client):
    raw = _valid_json(somatic="not_mentioned", interpersonal="not_mentioned")
    client = make_fake_client([raw])

    result = extract_extended_single(client, context="...", retry_attempts=1)

    assert result["somatic_symptoms"] == "not_mentioned"
    assert result["interpersonal_status"] == "not_mentioned"


def test_extract_extended_single_fuzzy_matches_medication_casing(make_fake_client):
    raw = _valid_json(medication="Not Mentioned")
    client = make_fake_client([raw])

    result = extract_extended_single(client, context="...", retry_attempts=1)

    assert result["medication_adherence"] == "not_mentioned"


def test_extract_extended_single_rejects_invalid_medication_label(make_fake_client):
    raw = _valid_json(medication="sometimes")
    client = make_fake_client([raw, raw])

    result = extract_extended_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["medication_adherence"] == "ERROR"
    assert "_error" in result


def test_extract_extended_single_rejects_empty_free_text(make_fake_client):
    raw = _valid_json(somatic="")
    client = make_fake_client([raw, raw])

    result = extract_extended_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["somatic_symptoms"] == "ERROR"


def test_extract_extended_single_rejects_too_short_free_text(make_fake_client):
    raw = _valid_json(interpersonal="fine")  # single word, not the sentinel
    client = make_fake_client([raw, raw])

    result = extract_extended_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["interpersonal_status"] == "ERROR"


def test_extract_extended_single_retries_on_malformed_json_then_succeeds(make_fake_client):
    client = make_fake_client(["not json", _valid_json()])

    result = extract_extended_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["medication_adherence"] == "taken"
