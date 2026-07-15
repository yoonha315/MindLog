"""Unit tests for mindlog.pipeline.extended_evaluator — the LLM judge for
free-text fields, and the combined extended evaluation report."""

import json

from mindlog.pipeline.extended_evaluator import (
    generate_extended_evaluation_report,
    judge_text_agreement,
)


def _verdict_json(verdict: str) -> str:
    return json.dumps({"verdict": verdict})


def test_judge_text_agreement_parses_correct(make_fake_client):
    client = make_fake_client([_verdict_json("correct")])

    verdict = judge_text_agreement(
        client, "heart racing", "heart racing episodes", "somatic_symptoms", retry_attempts=1
    )

    assert verdict == "correct"


def test_judge_text_agreement_parses_partial(make_fake_client):
    client = make_fake_client([_verdict_json("partial")])

    verdict = judge_text_agreement(
        client,
        "conflict with partner",
        "conflict with family",
        "interpersonal_status",
        retry_attempts=1,
    )

    assert verdict == "partial"


def test_judge_text_agreement_rejects_invalid_verdict_then_retries(make_fake_client):
    client = make_fake_client([_verdict_json("maybe"), _verdict_json("incorrect")])

    verdict = judge_text_agreement(
        client, "a", "b", "somatic_symptoms", retry_attempts=2, retry_delay=0
    )

    assert verdict == "incorrect"
    assert client.chat.completions.calls == 2


def test_judge_text_agreement_falls_back_to_incorrect_after_exhausting_retries(make_fake_client):
    client = make_fake_client(["not json", "still not json"])

    verdict = judge_text_agreement(
        client, "a", "b", "somatic_symptoms", retry_attempts=2, retry_delay=0
    )

    assert verdict == "incorrect"


def test_generate_extended_evaluation_report_combines_medication_and_judge_results(
    make_fake_client,
):
    extractions = [
        {
            "id": "S1",
            "extraction": {
                "medication_adherence": "taken",
                "somatic_symptoms": "heart racing",
                "interpersonal_status": "not_mentioned",
            },
        },
        {
            "id": "S2",
            "extraction": {
                "medication_adherence": "missed",
                "somatic_symptoms": "not_mentioned",
                "interpersonal_status": "conflict with a friend",
            },
        },
    ]
    ground_truths = [
        {
            "id": "S1",
            "labels": {
                "medication_adherence": "taken",
                "somatic_symptoms": "heart racing episodes",
                "interpersonal_status": "not_mentioned",
            },
        },
        {
            "id": "S2",
            "labels": {
                "medication_adherence": "taken",
                "somatic_symptoms": "not_mentioned",
                "interpersonal_status": "argument with a coworker",
            },
        },
    ]

    # Judge call order: somatic_symptoms(S1, S2), then interpersonal_status(S1, S2)
    client = make_fake_client(
        [
            _verdict_json("correct"),  # somatic S1
            _verdict_json("correct"),  # somatic S2 (both not_mentioned)
            _verdict_json("correct"),  # interpersonal S1 (both not_mentioned)
            _verdict_json("partial"),  # interpersonal S2
        ]
    )

    report = generate_extended_evaluation_report(
        client, extractions, ground_truths, judge_config={"retry_attempts": 1}
    )

    assert report["medication_adherence"]["accuracy"] == 0.5

    assert report["somatic_symptoms"]["counts"] == {"correct": 2, "partial": 0, "incorrect": 0}
    assert report["somatic_symptoms"]["percentages"]["correct"] == 100.0

    assert report["interpersonal_status"]["counts"] == {"correct": 1, "partial": 1, "incorrect": 0}
    assert report["interpersonal_status"]["percentages"]["correct"] == 50.0

    assert report["metadata"]["n_samples"] == 2

    # Per-sample cases let a caller pull out exactly which ids were partial/incorrect
    interpersonal_cases = report["interpersonal_status"]["cases"]
    assert [c["id"] for c in interpersonal_cases] == ["S1", "S2"]
    assert interpersonal_cases[0]["verdict"] == "correct"
    assert interpersonal_cases[1]["verdict"] == "partial"
    assert interpersonal_cases[1]["extracted"] == "conflict with a friend"
    assert interpersonal_cases[1]["ground_truth"] == "argument with a coworker"


def test_generate_extended_evaluation_report_treats_extraction_error_as_incorrect(
    make_fake_client,
):
    extractions = [
        {
            "id": "S1",
            "extraction": {
                "medication_adherence": "ERROR",
                "somatic_symptoms": "ERROR",
                "interpersonal_status": "ERROR",
            },
        }
    ]
    ground_truths = [
        {
            "id": "S1",
            "labels": {
                "medication_adherence": "taken",
                "somatic_symptoms": "heart racing",
                "interpersonal_status": "conflict",
            },
        }
    ]

    client = make_fake_client([])  # judge should never be called for ERROR extractions

    report = generate_extended_evaluation_report(client, extractions, ground_truths)

    assert report["somatic_symptoms"]["counts"]["incorrect"] == 1
    assert report["interpersonal_status"]["counts"]["incorrect"] == 1

    assert report["somatic_symptoms"]["cases"][0]["verdict"] == "incorrect"
    assert report["somatic_symptoms"]["cases"][0]["extracted"] == "ERROR"
