"""SQLite engine + session factory for MindLog.

The DB file lives at data/mindlog.db. init_db() creates tables on first use
directly from the ORM models — no separate migration tool yet.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mindlog.data.models import Base
from mindlog.utils.config_loader import get_project_root

DEFAULT_DB_PATH = os.path.join(get_project_root(), "data", "mindlog.db")


def get_engine(db_path: str = None) -> Engine:
    """Build a SQLAlchemy engine. Pass ":memory:" for an in-memory test DB."""
    path = db_path or DEFAULT_DB_PATH

    if path == ":memory:":
        # StaticPool keeps a single connection alive for the engine's lifetime —
        # otherwise each checkout would open a fresh, empty ":memory:" database.
        return create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return create_engine(f"sqlite:///{path}", future=True)


def init_db(engine: Engine = None) -> Engine:
    """Create all tables if they don't already exist."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
