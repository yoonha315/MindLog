"""Checklist scanner — tracks whether each monitoring topic has been
mentioned yet in the conversation, purely to help decide when a session
has collected enough signal to end.

This is a separate module from mindlog.agent.extractor and must stay that
way: extractor.py produces the final validated 5-field extraction and is
already-verified production logic. check_mentioned() only answers a much
cheaper yes/no "was X mentioned" question over a superset of topics (the
original 5 fields plus 3 additional ones) — it never judges label values
and its output is not persisted as an extraction result.
"""

import json
import time

CHECKLIST_FIELDS = [
    "affect_valence",
    "energy_level",
    "sleep_quality",
    "dominant_theme",
    "risk_indicators",
    "medication_adherence",
    "somatic_symptoms",  # 신체 증상 언급 여부 — 심장 두근거림, 과호흡, 발한,
    # 떨림, 어지러움 등 공황/불안 관련 신체 증상 전반
    # (특정 장애에 국한하지 않음)
    "interpersonal_status",  # 대인관계 관련 언급 여부 — 위축/고립뿐 아니라
    # 관계 변화, 갈등, 지지 등 대인관계 전반. dominant_theme이 세션당 하나의
    # 주제만 고르는 강제선택형이라 놓칠 수 있는 대인관계 신호를 별도로
    # 잡기 위한 보완 필드
]

CHECKLIST_SYSTEM_PROMPT = """\
You are a clinical NLP checklist scanner for a mental health self-monitoring app.

Given the full conversation transcript so far, determine whether each of the
following topics has been mentioned ANYWHERE in the conversation — you are
judging only whether the topic came up, not what was said about it or what
label it would receive.

Respond ONLY with a valid JSON object mapping each field name below to a
boolean (true or false) — no markdown, no preamble.

Fields to check:
1. "affect_valence": Has the user expressed anything about their emotional
   state or mood (positive, negative, or neutral)?
2. "energy_level": Has the user said anything about their energy, fatigue,
   or activity level?
3. "sleep_quality": Has the user mentioned anything about their sleep?
4. "dominant_theme": Has the user described a clear main topic or concern
   (e.g. work, relationships, health)?
5. "risk_indicators": Has the user said anything that could indicate
   self-harm, suicidal ideation, or crisis?
6. "medication_adherence": Has the user mentioned anything about taking,
   skipping, or adjusting medication?
7. "somatic_symptoms": Has the user mentioned any physical/bodily symptoms
   often associated with panic or anxiety in general — e.g. heart racing,
   hyperventilating, sweating, trembling, dizziness — regardless of
   diagnosis? Judge the presence of physical symptoms broadly, not just
   panic-disorder-specific language.
8. "interpersonal_status": Has the user said anything about their
   relationships in general — not just withdrawal or isolation, but also
   relationship changes, conflict, or support from others?

IMPORTANT:
- Judge only whether the topic was mentioned, not what was said about it.
- Base your judgment ONLY on what the user says, not the assistant's
  questions or prompts.
- Return EXACTLY these 8 keys with boolean values (true or false).
"""


def check_mentioned(
    client,
    conversation_text: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 512,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Determine, for each CHECKLIST_FIELDS entry, whether it has been
    mentioned anywhere in conversation_text so far. Returns {field: bool}.

    On repeated failure, returns all-False — the safe default, since the
    caller (ConversationManager) treats "not yet confirmed mentioned" as
    "keep the session going" rather than ending prematurely.
    """
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CHECKLIST_SYSTEM_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            result = {}
            for field in CHECKLIST_FIELDS:
                if field not in parsed:
                    raise ValueError(f"Missing field in checklist response: '{field}'")
                result[field] = _coerce_bool(parsed[field])

            return result

        except Exception:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {field: False for field in CHECKLIST_FIELDS}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes"}
    return bool(value)
