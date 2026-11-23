import json
from datetime import timedelta

import pytest

from assistant import leads
from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory
from assistant.memory import MemoryStore
from assistant.tools import TOOLS, ToolContext, execute_tool


@pytest.fixture(scope="module")
def inv():
    return Inventory(INVENTORY_PATH)


@pytest.fixture()
def ctx(inv, tmp_path, monkeypatch):
    monkeypatch.setattr(leads, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(leads, "BOOKINGS_CSV", tmp_path / "bookings.csv")
    mem = MemoryStore(tmp_path / "t.db")
    mem.ensure_user("sara", "Sara")
    sid = mem.create_session("sara")
    return ToolContext("sara", sid, {"shown": [], "focus": None}, inv, mem)


def next_weekday(weekday):
    d = leads.today() + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d.isoformat()


def test_every_tool_schema_has_an_implementation():
    from assistant.tools import _IMPL
    assert {t["function"]["name"] for t in TOOLS} == set(_IMPL)


def test_search_sets_short_term_state_and_cards(ctx):
    res = execute_tool(ctx, "search_inventory", {"make": "Mercedes", "body_type": "suv", "limit": 3})
    ids = [c["car_id"] for c in res["results"]]
    assert ctx.state["shown"] == ids and len(ids) == 3          # "the first one" = ids[0]
    assert set(ctx.cards) == set(ids)
    assert ctx.memory.recent_events("sara", "search")[0]["matches"] == res["total_matches"]


def test_search_converts_usd_to_aed(ctx):
    res = execute_tool(ctx, "search_inventory", {"max_price": 30000, "currency": "USD"})
    assert res["results"] and all(c["price_aed"] <= round(30000 * 3.6725) for c in res["results"])


def test_car_details_sets_focus_and_unknown_id_is_an_error(ctx):
    d = execute_tool(ctx, "get_car_details", {"car_id": 3})
    assert d["car_id"] == 3 and ctx.state["focus"] == 3
    assert "error" in execute_tool(ctx, "get_car_details", {"car_id": 9999})


def test_availability_rules(ctx):
    assert execute_tool(ctx, "check_viewing_availability", {"car_id": 3, "date": next_weekday(6)})["open"] is False
    ok = execute_tool(ctx, "check_viewing_availability", {"car_id": 3, "date": next_weekday(1)})
    assert ok["open"] and ok["slots"][0] == "08:00"
    assert "error" in execute_tool(ctx, "check_viewing_availability", {"car_id": 3, "date": "tomorrow"})


def test_booking_updates_profile_lead_and_history(ctx):
    day = next_weekday(2)
    res = execute_tool(ctx, "book_viewing", {"car_id": 3, "date": day, "time": "11:00", "name": "Sara",
                                             "phone": "0501234567", "kind": "test_drive"})
    assert res["success"] and res["lead_score"] == "hot"
    assert ctx.memory.get_user("sara")["profile"]["phone"] == "0501234567"
    assert ctx.memory.recent_events("sara", "booking")[0]["slot"].startswith(day)
    again = execute_tool(ctx, "book_viewing", {"car_id": 3, "date": day, "time": "11:00", "name": "Omar", "phone": "1"})
    assert again["success"] is False and "11:00" not in again["available_slots"]


def test_profile_update_converts_budget_and_records_lead(ctx):
    res = execute_tool(ctx, "update_user_profile", {"budget_max": 20000, "currency": "USD",
                                                    "body_types": ["SUV"], "preferred_colors": None})
    assert res["profile"]["budget_max_aed"] == round(20000 * 3.6725)
    assert res["profile"]["body_types"] == ["suv"]
    assert res["lead_score"] == "warm" and leads.list_leads()[0]["user_id"] == "sara"


def test_favorite_is_saved(ctx):
    res = execute_tool(ctx, "save_favorite_car", {"car_id": 8, "reason": "loves it"})
    assert res["saved"] and ctx.memory.get_favorites("sara")[0]["car_id"] == 8
    assert "error" in execute_tool(ctx, "save_favorite_car", {"car_id": 9999})


def test_execute_tool_never_crashes(ctx):
    assert "error" in execute_tool(ctx, "does_not_exist", {})
    assert "error" in execute_tool(ctx, "search_inventory", "{not json")
    assert "error" in execute_tool(ctx, "get_car_details", {})          # missing required arg
    assert execute_tool(ctx, "search_inventory", json.dumps({"make": "saab"}))["total_matches"] == 0


def test_search_with_unstated_fields_returns_possible_matches_and_shows_them(ctx):
    res = execute_tool(ctx, "search_inventory", {"color": "white", "body_type": "suv", "max_price": 20000, "currency": "USD"})
    assert res["unverified_matches"]                                                # listings that don't state price/colour
    expected = [c["car_id"] for c in res["results"]] + [c["car_id"] for c in res["unverified_matches"]]
    assert ctx.state["shown"] == expected                                           # confirmed first, then possible matches
    assert set(expected) <= set(ctx.cards)                                          # so the UI can render their cards