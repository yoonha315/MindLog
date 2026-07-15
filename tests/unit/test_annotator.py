"""Unit tests for mindlog.agent.annotator.annotate_single — the CoT
ground-truth pass shares extract_single's validation contract."""

from mindlog.agent.annotator import annotate_single


def test_annotate_single_parses_valid_json(make_fake_client, valid_annotation_json):
    client = make_fake_client([valid_annotation_json])

    result = annotate_single(client, context="...", retry_attempts=1)

    assert result["labels"]["affect_valence"] == "negative"
    assert "awful" in result["reasoning"]["affect_valence"]


def test_annotate_single_returns_error_labels_after_exhausting_retries(make_fake_client):
    client = make_fake_client(["broken json", "still broken"])

    result = annotate_single(client, context="...", retry_attempts=2, retry_delay=0)

    assert result["labels"]["affect_valence"] == "ERROR"
    assert "_error" in result["reasoning"]
