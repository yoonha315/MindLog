"""Unit tests for mindlog.agent.checklist.check_mentioned — mention
detection parsing and retry/fallback behavior."""

import json

from mindlog.agent.checklist import CHECKLIST_FIELDS, check_mentioned


def _checklist_json(value: bool) -> str:
    return json.dumps({field: value for field in CHECKLIST_FIELDS})


def test_check_mentioned_parses_all_true(make_fake_client):
    client = make_fake_client([_checklist_json(True)])

    result = check_mentioned(client, "I feel awful and haven't slept.", retry_attempts=1)

    assert result == {field: True for field in CHECKLIST_FIELDS}


def test_check_mentioned_handles_mixed_results(make_fake_client):
    raw = json.dumps(
        {
            "affect_valence": True,
            "energy_level": False,
            "sleep_quality": True,
            "dominant_theme": False,
            "risk_indicators": False,
            "medication_adherence": False,
            "somatic_symptoms": False,
            "interpersonal_status": False,
        }
    )
    client = make_fake_client([raw])

    result = check_mentioned(client, "...", retry_attempts=1)

    assert result["affect_valence"] is True
    assert result["sleep_quality"] is True
    assert result["energy_level"] is False


def test_check_mentioned_coerces_string_booleans(make_fake_client):
    raw = json.dumps({field: "true" for field in CHECKLIST_FIELDS})
    client = make_fake_client([raw])

    result = check_mentioned(client, "...", retry_attempts=1)

    assert all(value is True for value in result.values())


def test_check_mentioned_retries_on_malformed_json_then_succeeds(make_fake_client):
    client = make_fake_client(["not json", _checklist_json(False)])

    result = check_mentioned(client, "...", retry_attempts=2, retry_delay=0)

    assert result == {field: False for field in CHECKLIST_FIELDS}
    assert client.chat.completions.calls == 2


def test_check_mentioned_returns_all_false_after_exhausting_retries(make_fake_client):
    client = make_fake_client(["broken", "still broken"])

    result = check_mentioned(client, "...", retry_attempts=2, retry_delay=0)

    assert result == {field: False for field in CHECKLIST_FIELDS}


def test_check_mentioned_returns_all_false_when_a_field_is_missing(make_fake_client):
    raw = json.dumps({"affect_valence": True})  # missing the other 7 fields
    client = make_fake_client([raw, raw])

    result = check_mentioned(client, "...", retry_attempts=2, retry_delay=0)

    assert result == {field: False for field in CHECKLIST_FIELDS}


def test_check_mentioned_passes_model_config_through_to_the_api_call(make_fake_client):
    client = make_fake_client([_checklist_json(False)])

    check_mentioned(
        client,
        "...",
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=256,
        retry_attempts=1,
    )

    sent = client.chat.completions.received_kwargs[-1]
    assert sent["model"] == "gpt-4o-mini"
    assert sent["max_tokens"] == 256
