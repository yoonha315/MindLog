"""Conversation session lifecycle: turn tracking, end-of-session detection,
and end-of-session structured extraction."""

import time
from datetime import datetime, timezone

from mindlog.agent.checklist import CHECKLIST_FIELDS, check_mentioned
from mindlog.agent.client import build_client
from mindlog.agent.extended_extractor import extract_extended_single
from mindlog.agent.extractor import extract_single
from mindlog.data.models import Extraction, Message, Session
from mindlog.utils.logger import get_logger

logger = get_logger("mindlog_conversation")


class ConversationManager:
    """Drives a single conversation session against the DB and the extractor.

    A new instance (or a fresh start_session() call) is expected per active
    session; get_history() is the only method that looks across sessions.

    Don't construct this directly outside of tests — use
    create_conversation_manager(session_factory, cfg=load_config()) instead,
    so the checklist scanner is wired to its own (cheaper) model from
    configs/config.yaml rather than silently falling back to extraction's
    gpt-4o default.
    """

    def __init__(
        self,
        session_factory,
        client=None,
        timeout_seconds: float = 600,
        min_turns: int = 6,
        checklist_check_interval: int = 3,
        extraction_kwargs: dict = None,
        checklist_kwargs: dict = None,
    ):
        self._session_factory = session_factory
        self._client = client
        self.timeout_seconds = timeout_seconds
        self.min_turns = min_turns
        self.checklist_check_interval = checklist_check_interval
        self.extraction_kwargs = extraction_kwargs or {}
        self.checklist_kwargs = checklist_kwargs or {}

        self.session_id = None
        self.user_id = None
        self.messages = []
        self._last_message_time = None
        self._checklist = {field: False for field in CHECKLIST_FIELDS}

    def start_session(self, user_id: str) -> int:
        with self._session_factory() as db:
            session_row = Session(user_id=user_id)
            db.add(session_row)
            db.commit()
            self.session_id = session_row.id

        self.user_id = user_id
        self.messages = []
        self._last_message_time = time.time()
        self._checklist = {field: False for field in CHECKLIST_FIELDS}
        return self.session_id

    def add_turn(self, role: str, content: str) -> None:
        if self.session_id is None:
            raise RuntimeError("start_session() must be called before add_turn()")

        turn_index = len(self.messages)
        self.messages.append({"role": role, "content": content, "turn_index": turn_index})
        self._last_message_time = time.time()

        with self._session_factory() as db:
            db.add(
                Message(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    role=role,
                    content=content,
                )
            )
            session_row = db.get(Session, self.session_id)
            session_row.turn_count = len(self.messages)
            db.commit()

        if len(self.messages) % self.checklist_check_interval == 0:
            self._update_checklist()

    def _update_checklist(self) -> None:
        """Scan the transcript so far and OR-merge newly mentioned topics into
        the running checklist — a field already marked True never reverts."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)
        mentioned = check_mentioned(self._get_client(), transcript, **self.checklist_kwargs)
        for field, is_mentioned in mentioned.items():
            if is_mentioned:
                self._checklist[field] = True

    def _get_client(self):
        if self._client is None:
            self._client = build_client()
        return self._client

    def check_session_end(self) -> bool:
        """True if the session should end: timeout since the last message, or
        the checklist is fully covered and the min-turn safety floor is met."""
        if self._last_message_time is not None:
            elapsed = time.time() - self._last_message_time
            if elapsed > self.timeout_seconds:
                return True

        if all(self._checklist.values()) and len(self.messages) >= self.min_turns:
            return True

        return False

    def end_session(self, reason: str) -> dict:
        """Mark the session ended, run extraction over the full transcript,
        and persist the result. Returns the extraction dict."""
        with self._session_factory() as db:
            session_row = db.get(Session, self.session_id)
            session_row.ended_at = datetime.now(timezone.utc)
            session_row.end_reason = reason
            db.commit()

        context = "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

        client = self._client or build_client()
        extraction = extract_single(client, context=context, **self.extraction_kwargs)
        model_name = self.extraction_kwargs.get("model", "gpt-4o")

        somatic_symptoms, interpersonal_status, medication_adherence = (
            self._extract_extended_fields(client, context)
        )

        with self._session_factory() as db:
            db.add(
                Extraction(
                    session_id=self.session_id,
                    affect_valence=extraction["affect_valence"],
                    energy_level=extraction["energy_level"],
                    sleep_quality=extraction["sleep_quality"],
                    dominant_theme=extraction["dominant_theme"],
                    risk_indicators=extraction["risk_indicators"],
                    somatic_symptoms=somatic_symptoms,
                    interpersonal_status=interpersonal_status,
                    medication_adherence=medication_adherence,
                    model=model_name,
                )
            )
            db.commit()

        return extraction

    def _extract_extended_fields(
        self, client, context: str
    ) -> tuple[str | None, str | None, str | None]:
        """Extract somatic_symptoms, interpersonal_status, and
        medication_adherence via a single extended-field extractor call.
        Each field is checked for the "ERROR" sentinel independently and
        stored as null on failure — one field failing must never block the
        others, or the 5-field extraction above, from being saved.
        """
        try:
            extended = extract_extended_single(client, context=context, **self.extraction_kwargs)
        except Exception:
            logger.exception("extended-field extraction raised; storing null for all 3 fields")
            return None, None, None

        somatic_symptoms = extended.get("somatic_symptoms")
        if somatic_symptoms == "ERROR":
            logger.warning("somatic_symptoms extraction returned ERROR; storing null")
            somatic_symptoms = None

        interpersonal_status = extended.get("interpersonal_status")
        if interpersonal_status == "ERROR":
            logger.warning("interpersonal_status extraction returned ERROR; storing null")
            interpersonal_status = None

        medication_adherence = extended.get("medication_adherence")
        if medication_adherence == "ERROR":
            logger.warning("medication_adherence extraction returned ERROR; storing null")
            medication_adherence = None

        return somatic_symptoms, interpersonal_status, medication_adherence

    def get_history(self, user_id: str, limit: int = 5) -> list[dict]:
        """Prior sessions' extractions for user_id, most recent first."""
        with self._session_factory() as db:
            rows = (
                db.query(Extraction)
                .join(Session, Extraction.session_id == Session.id)
                .filter(Session.user_id == user_id)
                .order_by(Extraction.extracted_at.desc(), Extraction.id.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "session_id": row.session_id,
                    "affect_valence": row.affect_valence,
                    "energy_level": row.energy_level,
                    "sleep_quality": row.sleep_quality,
                    "dominant_theme": row.dominant_theme,
                    "risk_indicators": row.risk_indicators,
                    "somatic_symptoms": row.somatic_symptoms,
                    "interpersonal_status": row.interpersonal_status,
                    "medication_adherence": row.medication_adherence,
                    "model": row.model,
                    "extracted_at": row.extracted_at,
                }
                for row in rows
            ]


def create_conversation_manager(session_factory, cfg: dict, client=None) -> ConversationManager:
    """
    Build a ConversationManager wired to configs/config.yaml's `conversation`,
    `llm`, and `checklist` sections. This is the supported way to construct
    one outside of unit tests — it's the only path that guarantees the
    checklist scanner actually uses its own (cheaper) model from `checklist:`
    instead of silently falling back to extraction's gpt-4o default.

    Pass cfg=load_config() (or an equivalent dict) from the caller.
    """
    conv_cfg = cfg.get("conversation", {})
    llm_cfg = cfg.get("llm", {})
    checklist_cfg = cfg.get("checklist", {})

    extraction_kwargs = {
        "model": llm_cfg.get("model", "gpt-4o"),
        "temperature": llm_cfg.get("temperature", 0),
        "max_tokens": llm_cfg.get("max_tokens", 512),
        "retry_attempts": llm_cfg.get("retry_attempts", 3),
        "retry_delay": llm_cfg.get("retry_delay_seconds", 2.0),
    }
    checklist_kwargs = {
        "model": checklist_cfg.get("model", "gpt-4o-mini"),
        "temperature": checklist_cfg.get("temperature", 0),
        "max_tokens": checklist_cfg.get("max_tokens", 256),
    }

    return ConversationManager(
        session_factory,
        client=client,
        timeout_seconds=conv_cfg.get("timeout_seconds", 600),
        min_turns=conv_cfg.get("min_turns", 6),
        checklist_check_interval=conv_cfg.get("checklist_check_interval", 3),
        extraction_kwargs=extraction_kwargs,
        checklist_kwargs=checklist_kwargs,
    )
