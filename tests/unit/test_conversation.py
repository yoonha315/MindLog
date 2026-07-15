"""Unit tests for mindlog.agent.conversation.ConversationManager — session
lifecycle, end-condition detection, and end-of-session extraction. Uses an
in-memory SQLite DB and the FakeOpenAIClient from conftest.py — no real
network calls."""

import json
import time

import pytest
from mindlog.agent.checklist import CHECKLIST_FIELDS
from mindlog.agent.conversation import ConversationManager, create_conversation_manager
from mindlog.data.db import get_engine, get_session_factory, init_db
from mindlog.data.models import Extraction, Message, Session, User
from mindlog.utils.config_loader import load_config


def _all_checklist_json(value: bool) -> str:
    return json.dumps({field: value for field in CHECKLIST_FIELDS})


@pytest.fixture
def session_factory():
    engine = get_engine(":memory:")
    init_db(engine)
    factory = get_session_factory(engine)

    with factory() as db:
        db.add(User(id="u1", display_name="Test User", timezone="Asia/Seoul"))
        db.commit()

    return factory


def test_full_session_lifecycle_ends_on_timeout_and_saves_extraction(
    session_factory, make_fake_client, valid_extraction_json
):
    client = make_fake_client([valid_extraction_json])
    manager = ConversationManager(
        session_factory,
        client=client,
        timeout_seconds=0.05,
        min_turns=100,  # unreachable — isolate the timeout path
    )

    session_id = manager.start_session("u1")
    manager.add_turn("assistant", "How was your day?")
    manager.add_turn("user", "I feel awful and can't sleep.")

    time.sleep(0.1)
    assert manager.check_session_end() is True

    extraction = manager.end_session("timeout")
    assert extraction["affect_valence"] == "negative"

    with session_factory() as db:
        session_row = db.get(Session, session_id)
        assert session_row.ended_at is not None
        assert session_row.end_reason == "timeout"
        assert session_row.turn_count == 2

        messages = db.query(Message).filter(Message.session_id == session_id).all()
        assert len(messages) == 2
        assert [m.role for m in messages] == ["assistant", "user"]

        extraction_row = db.query(Extraction).filter(Extraction.session_id == session_id).one()
        assert extraction_row.affect_valence == "negative"
        assert extraction_row.model


def test_session_does_not_end_before_min_turns_or_timeout(session_factory, make_fake_client):
    client = make_fake_client([])  # extraction must never be called here
    manager = ConversationManager(
        session_factory,
        client=client,
        timeout_seconds=999,
        min_turns=6,
    )

    manager.start_session("u1")
    manager.add_turn("user", "hi")
    manager.add_turn("assistant", "hello")

    assert manager.check_session_end() is False


def test_session_ends_when_checklist_complete_and_min_turns_reached(
    session_factory, make_fake_client
):
    # checklist_check_interval=1 -> every add_turn triggers a check_mentioned call
    client = make_fake_client([_all_checklist_json(True), _all_checklist_json(True)])
    manager = ConversationManager(
        session_factory,
        client=client,
        timeout_seconds=999,
        min_turns=2,
        checklist_check_interval=1,
    )

    manager.start_session("u1")
    manager.add_turn("user", "msg1")
    assert manager.check_session_end() is False  # checklist complete, but turns < min_turns

    manager.add_turn("assistant", "msg2")
    assert manager.check_session_end() is True  # checklist complete AND turns >= min_turns


def test_session_does_not_end_when_checklist_incomplete_even_past_min_turns(
    session_factory, make_fake_client
):
    client = make_fake_client(
        [_all_checklist_json(False), _all_checklist_json(False), _all_checklist_json(False)]
    )
    manager = ConversationManager(
        session_factory,
        client=client,
        timeout_seconds=999,
        min_turns=2,
        checklist_check_interval=1,
    )

    manager.start_session("u1")
    manager.add_turn("user", "msg1")
    manager.add_turn("assistant", "msg2")
    manager.add_turn("user", "msg3")

    assert manager.check_session_end() is False  # turns >= min_turns, but checklist incomplete


def test_checklist_check_uses_checklist_kwargs_not_extraction_kwargs(
    session_factory, make_fake_client
):
    client = make_fake_client([_all_checklist_json(True)])
    manager = ConversationManager(
        session_factory,
        client=client,
        timeout_seconds=999,
        min_turns=1,
        checklist_check_interval=1,  # every add_turn triggers a check_mentioned call
        extraction_kwargs={"model": "gpt-4o"},
        checklist_kwargs={"model": "gpt-4o-mini", "temperature": 0, "max_tokens": 256},
    )

    manager.start_session("u1")
    manager.add_turn("user", "msg1")  # only the checklist call happens here, not end_session

    sent = client.chat.completions.received_kwargs[-1]
    assert sent["model"] == "gpt-4o-mini"
    assert sent["max_tokens"] == 256


def test_get_history_returns_extractions_most_recent_first(
    session_factory, make_fake_client, valid_extraction_json
):
    client = make_fake_client([valid_extraction_json, valid_extraction_json])
    manager = ConversationManager(session_factory, client=client, timeout_seconds=999, min_turns=1)

    manager.start_session("u1")
    manager.add_turn("user", "first session")
    first_session_id = manager.session_id
    manager.end_session("data_complete")

    manager.start_session("u1")
    manager.add_turn("user", "second session")
    second_session_id = manager.session_id
    manager.end_session("data_complete")

    history = manager.get_history("u1", limit=5)

    assert len(history) == 2
    assert history[0]["session_id"] == second_session_id
    assert history[1]["session_id"] == first_session_id


def test_create_conversation_manager_maps_config_sections(session_factory, make_fake_client):
    client = make_fake_client([])
    cfg = {
        "conversation": {"timeout_seconds": 120, "min_turns": 4, "checklist_check_interval": 2},
        "llm": {
            "model": "gpt-4o",
            "temperature": 0,
            "max_tokens": 512,
            "retry_attempts": 3,
            "retry_delay_seconds": 2.0,
        },
        "checklist": {"model": "gpt-4o-mini", "temperature": 0, "max_tokens": 256},
    }

    manager = create_conversation_manager(session_factory, cfg, client=client)

    assert isinstance(manager, ConversationManager)
    assert manager.checklist_kwargs["model"] == "gpt-4o-mini"
    assert manager.extraction_kwargs["model"] == "gpt-4o"
    assert manager.extraction_kwargs["retry_delay"] == 2.0  # mapped from retry_delay_seconds
    assert manager.timeout_seconds == 120
    assert manager.min_turns == 4
    assert manager.checklist_check_interval == 2


def test_create_conversation_manager_defaults_when_sections_missing(
    session_factory, make_fake_client
):
    client = make_fake_client([])

    manager = create_conversation_manager(session_factory, cfg={}, client=client)

    assert manager.checklist_kwargs["model"] == "gpt-4o-mini"
    assert manager.timeout_seconds == 600
    assert manager.min_turns == 6
    assert manager.checklist_check_interval == 3


def test_create_conversation_manager_from_real_config_uses_cheap_checklist_model(
    session_factory, make_fake_client
):
    """Regression test for the bug this factory exists to make impossible:
    a ConversationManager built without explicitly wiring config.yaml would
    silently use gpt-4o for the checklist scan instead of the configured
    cheaper model."""
    client = make_fake_client([])
    cfg = load_config()

    manager = create_conversation_manager(session_factory, cfg, client=client)

    assert manager.checklist_kwargs["model"] == "gpt-4o-mini"


def test_create_conversation_manager_checklist_call_uses_configured_model(
    session_factory, make_fake_client
):
    cfg = load_config()
    client = make_fake_client([_all_checklist_json(True)])
    manager = create_conversation_manager(session_factory, cfg, client=client)

    manager.start_session("u1")
    for i in range(manager.checklist_check_interval):
        manager.add_turn("user", f"msg{i}")

    sent = client.chat.completions.received_kwargs[-1]
    assert sent["model"] == "gpt-4o-mini"
