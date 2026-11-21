import pytest

from assistant.memory import MemoryStore, slug_user_id


@pytest.fixture()
def mem(tmp_path):
    return MemoryStore(tmp_path / "test.db")


def test_slug_user_id():
    assert slug_user_id("  Sara Ahmed! ") == "sara_ahmed"


def test_new_then_returning_user(mem):
    user, returning = mem.ensure_user("sara", "Sara")
    assert returning is False and user["name"] == "Sara"
    _, returning = mem.ensure_user("sara")
    assert returning is True


def test_profile_merge_unions_lists_and_overwrites_scalars(mem):
    mem.ensure_user("sara")
    mem.update_profile("sara", {"budget_max_aed": 70000, "body_types": ["suv"], "name": "Sara"})
    p = mem.update_profile("sara", {"budget_max_aed": 90000, "body_types": ["SUV", "sedan"], "purpose": None})
    assert p["budget_max_aed"] == 90000                  # scalar overwritten
    assert p["body_types"] == ["suv", "sedan"]            # union, case-insensitive, no duplicates
    assert "purpose" not in p                             # None is ignored
    assert mem.get_user("sara")["name"] == "Sara"


def test_short_term_session_messages_and_state(mem):
    mem.ensure_user("sara")
    sid = mem.create_session("sara")
    mem.add_message(sid, "user", "hi")
    mem.add_message(sid, "assistant", "hello")
    assert [m["role"] for m in mem.get_messages(sid)] == ["user", "assistant"]
    mem.save_state(sid, {"shown": [8, 3], "focus": 8})
    assert mem.get_session(sid)["state"] == {"shown": [8, 3], "focus": 8}


def test_favorites_and_events_survive_new_session(mem):
    mem.ensure_user("sara")
    s1 = mem.create_session("sara")
    mem.add_favorite("sara", 8, "2016 Porsche Cayenne GTS", "loves it")
    mem.log_event("sara", s1, "search", {"criteria": "white suv", "matches": 4})
    s2 = mem.create_session("sara")
    snap = mem.snapshot("sara", current_session=s2)
    assert snap["favorites"][0]["car_id"] == 8
    assert snap["recent_searches"][0]["criteria"] == "white suv"
    assert snap["sessions"] == 2


def test_current_session_events_are_excluded_from_snapshot(mem):
    mem.ensure_user("sara")
    s1 = mem.create_session("sara")
    mem.log_event("sara", s1, "search", {"criteria": "red coupe", "matches": 1})
    assert mem.snapshot("sara", current_session=s1)["recent_searches"] == []


def test_forget_user_wipes_everything(mem):
    mem.ensure_user("sara", "Sara")
    sid = mem.create_session("sara")
    mem.add_message(sid, "user", "hi")
    mem.add_favorite("sara", 8, "x")
    mem.forget_user("sara")
    assert mem.get_user("sara") is None and mem.get_favorites("sara") == []
    assert mem.snapshot("sara")["known"] is False