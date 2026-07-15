"""Conversation session lifecycle: turn tracking, end-of-session detection,
and end-of-session structured extraction."""

import time
from datetime import datetime, timezone

from mindlog.agent.checklist import CHECKLIST_FIELDS, check_mentioned
from mindlog.agent.client import build_client
from mindlog.agent.extractor import extract_single
from mindlog.data.models import Extraction, Message, Session


class ConversationManager:
    """Drives a single conversation session against the DB and the extractor.

    A new instance (or a fresh start_session() call) is expected per active
    session; get_history() is the only method that looks across sessions.
    """

    def __init__(
        self,
        session_factory,
        client=None,
        timeout_seconds: float = 600,
        min_turns: int = 6,
        checklist_check_interval: int = 3,
        extraction_kwargs: dict = None,
    ):
        self._session_factory = session_factory
        self._client = client
        self.timeout_seconds = timeout_seconds
        self.min_turns = min_turns
        self.checklist_check_interval = checklist_check_interval
        self.extraction_kwargs = extraction_kwargs or {}

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
        mentioned = check_mentioned(self._get_client(), transcript)
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

        with self._session_factory() as db:
            db.add(
                Extraction(
                    session_id=self.session_id,
                    affect_valence=extraction["affect_valence"],
                    energy_level=extraction["energy_level"],
                    sleep_quality=extraction["sleep_quality"],
                    dominant_theme=extraction["dominant_theme"],
                    risk_indicators=extraction["risk_indicators"],
                    model=model_name,
                )
            )
            db.commit()

        return extraction

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
                    "model": row.model,
                    "extracted_at": row.extracted_at,
                }
                for row in rows
            ]
