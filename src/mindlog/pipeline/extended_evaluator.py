"""
Evaluation for the 3 extended monitoring fields — medication_adherence,
somatic_symptoms, interpersonal_status.

Independent from mindlog.pipeline.evaluator: medication_adherence is
categorical, so it reuses compute_field_metrics from the original
(already-validated) evaluator as-is. somatic_symptoms and
interpersonal_status are free text, so they're scored with an LLM judge
(judge_text_agreement) that classifies agreement as correct/partial/incorrect
rather than exact-match accuracy.
"""

import json
import time

from openai import OpenAI

from mindlog.agent.extended_schema import FREE_TEXT_FIELDS, MEDICATION_ADHERENCE_LABELS
from mindlog.pipeline.evaluator import compute_field_metrics

VALID_VERDICTS = ("correct", "partial", "incorrect")

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether an automatically extracted summary
matches a ground-truth summary for the same conversational field.

You will be given the field name, the "extracted summary" (produced by an
extraction pipeline under test), and the "ground truth summary" (produced by
a separate, higher-fidelity annotation pass). Classify their agreement as:

- "correct": the two summaries describe the same underlying content (same
  symptoms / same relationship information), even if worded differently. If
  BOTH summaries are exactly "not_mentioned", that also counts as "correct".
- "partial": the two summaries overlap but one contains information the
  other misses, or one is notably vaguer/more specific than the other.
- "incorrect": the two summaries describe different content, OR exactly one
  of them is "not_mentioned" while the other describes real content.

Respond ONLY with a valid JSON object: {"verdict": "correct" | "partial" | "incorrect"}
"""


def judge_text_agreement(
    client: OpenAI,
    extracted_text: str,
    ground_truth_text: str,
    field_name: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 256,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    LLM-as-judge: compare an extracted free-text summary against the
    ground-truth summary for the same field and classify their agreement.

    On repeated failure, falls back to "incorrect" — the conservative
    choice, since a silent pass would inflate the reported agreement rate.
    """
    user_content = (
        f"Field: {field_name}\n\n"
        f"Extracted summary: {extracted_text}\n"
        f"Ground truth summary: {ground_truth_text}\n"
    )

    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            verdict = str(parsed.get("verdict", "")).strip().lower()

            if verdict not in VALID_VERDICTS:
                raise ValueError(f"Invalid verdict: '{verdict}' (allowed: {VALID_VERDICTS})")

            return verdict

        except Exception:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return "incorrect"


def _judge_batch(
    client: OpenAI,
    extractions: list[dict],
    ground_truths: list[dict],
    field: str,
    judge_config: dict = None,
    logger=None,
) -> list[dict]:
    """Run judge_text_agreement over every aligned (extraction, ground_truth)
    pair for a single free-text field. Returns per-sample case dicts (id,
    extracted, ground_truth, verdict) so callers can inspect exactly which
    samples were partial/incorrect, not just the aggregate counts."""
    judge_config = judge_config or {}
    gt_map = {g["id"]: g["labels"] for g in ground_truths}

    cases = []
    for i, ext in enumerate(extractions):
        sid = ext["id"]
        if sid not in gt_map:
            continue

        extracted_text = ext["extraction"].get(field, "ERROR")
        ground_truth_text = gt_map[sid].get(field, "MISSING")

        if extracted_text == "ERROR":
            verdict = "incorrect"
        else:
            verdict = judge_text_agreement(
                client, extracted_text, ground_truth_text, field, **judge_config
            )

        cases.append(
            {
                "id": sid,
                "extracted": extracted_text,
                "ground_truth": ground_truth_text,
                "verdict": verdict,
            }
        )

        if logger and (i + 1) % 10 == 0:
            logger.info(f"  Judged {field}: {i + 1}/{len(extractions)}")

    return cases


def _summarize_verdicts(cases: list[dict]) -> dict:
    n = len(cases)
    counts = {"correct": 0, "partial": 0, "incorrect": 0}
    for case in cases:
        counts[case["verdict"]] = counts.get(case["verdict"], 0) + 1

    return {
        "n_samples": n,
        "counts": counts,
        "percentages": {k: round(100 * c / n, 1) if n > 0 else 0.0 for k, c in counts.items()},
        "cases": cases,
    }


def generate_extended_evaluation_report(
    client: OpenAI,
    extractions: list[dict],
    ground_truths: list[dict],
    judge_config: dict = None,
    logger=None,
) -> dict:
    """
    Build the combined extended-field evaluation report:
      - medication_adherence: accuracy/precision/recall/f1/kappa, via the
        original evaluator's compute_field_metrics (categorical, reused as-is)
      - somatic_symptoms, interpersonal_status: correct/partial/incorrect
        counts and percentages, via judge_text_agreement (free text) — plus
        a per-sample "cases" list (id/extracted/ground_truth/verdict) so
        the specific partial/incorrect samples can be pulled out later
        without re-running the judge
    """
    gt_map = {g["id"]: g["labels"] for g in ground_truths}

    y_true = []
    y_pred = []
    for ext in extractions:
        sid = ext["id"]
        if sid not in gt_map:
            continue
        y_true.append(gt_map[sid].get("medication_adherence", "MISSING"))
        y_pred.append(ext["extraction"].get("medication_adherence", "ERROR"))

    medication_report = compute_field_metrics(
        y_true, y_pred, "medication_adherence", MEDICATION_ADHERENCE_LABELS
    )

    text_field_reports = {}
    for field in FREE_TEXT_FIELDS:
        cases = _judge_batch(
            client, extractions, ground_truths, field, judge_config=judge_config, logger=logger
        )
        text_field_reports[field] = _summarize_verdicts(cases)

    return {
        "medication_adherence": medication_report,
        **text_field_reports,
        "metadata": {
            "n_samples": len(extractions),
        },
    }
