"""SQLAlchemy ORM models: users, sessions, messages, extractions.

Relationships:
    users 1:N sessions
    sessions 1:N messages
    sessions 1:0..1 extractions (one extraction per session, at most)
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name: Mapped[str] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(Base):
    """A single conversation session (not to be confused with a SQLAlchemy DB session)."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(default=None)
    end_reason: Mapped[str | None] = mapped_column(String(32), default=None)
    turn_count: Mapped[int] = mapped_column(default=0)

    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    extraction: Mapped["Extraction | None"] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    turn_index: Mapped[int] = mapped_column()
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    session: Mapped["Session"] = relationship(back_populates="messages")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), unique=True)
    affect_valence: Mapped[str] = mapped_column(String(16))
    energy_level: Mapped[str] = mapped_column(String(16))
    sleep_quality: Mapped[str] = mapped_column(String(16))
    dominant_theme: Mapped[str] = mapped_column(String(32))
    risk_indicators: Mapped[str] = mapped_column(String(16))
    # Extended-field track (see mindlog.agent.extended_extractor) — only
    # somatic_symptoms is validated enough to persist so far.
    # medication_adherence/interpersonal_status intentionally have no column yet.
    somatic_symptoms: Mapped[str | None] = mapped_column(Text, default=None)
    model: Mapped[str] = mapped_column(String(64))
    extracted_at: Mapped[datetime] = mapped_column(default=_utcnow)

    session: Mapped["Session"] = relationship(back_populates="extraction")
