"""Agent tests with a scripted fake LLM: no API key, no network."""
import json
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from assistant import leads
from assistant.agent import Agent, build_greeting
from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory
from assistant.memory import MemoryStore


@pytest.fixture(scope="module")
def inv():
    return Inventory(INVENTORY_PATH)


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(leads, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(leads, "BOOKINGS_CSV", tmp_path / "bookings.csv")
    return MemoryStore(tmp_path / "m.db")


def tool_call(name, args, call_id="c1"):
    return NS(content=None, tool_calls=[NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))])


def text(t):
    return NS(content=t, tool_calls=None)


class FakeAgent(Agent):
    """Replaces the network call with a scripted list of LLM replies."""

    def __init__(self, inv, mem, script):
        super().__init__(inv, mem)
        self.script = list(script)
        self.prompts = []

    def _llm(self, messages):
        self.prompts.append(messages[0]["content"])
        return NS(choices=[NS(message=self.script.pop(0))])


def next_weekday(weekday):
    d = leads.today() + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d.isoformat()


def test_multi_turn_and_returning_user(inv, mem):
    mem.ensure_user("sara", "Sara")
    s1 = mem.create_session("sara")
    agent = FakeAgent(inv, mem, [
        tool_call("update_user_profile", {"name": "Sara", "budget_max": 20000, "currency": "USD",
                                          "body_types": ["suv"], "preferred_colors": ["white"]}),
        tool_call("search_inventory", {"make": "Porsche"}, "c2"), text("Found the Cayenne #8."),
        tool_call("get_car_details", {"car_id": 8}), text("Mileage is not stated in the listing."),
        tool_call("save_favorite_car", {"car_id": 8, "reason": "loves the GTS"}), text("Saved!"),
    ])
    r1 = agent.chat("sara", s1, "white SUV under $20k, Porsche please")
    assert r1["state"]["shown"] == [8] and r1["cars"][0]["car_id"] == 8
    agent.chat("sara", s1, "what's the mileage on it?")
    assert "Cayenne" in agent.prompts[-1]                       # focus car injected = short-term memory
    agent.chat("sara", s1, "I like it")

    s2 = mem.create_session("sara")                             # brand-new session, same user
    snap = mem.snapshot("sara", current_session=s2)
    greeting = build_greeting(snap, True)
    assert "Sara" in greeting and "Cayenne" in greeting and "white" in greeting
    agent2 = FakeAgent(inv, mem, [text("Welcome back!")])
    agent2.chat("sara", s2, "hi again")
    assert "Cayenne" in agent2.prompts[0] and "Sara" in agent2.prompts[0]     # long-term memory in the prompt


def test_booking_flow_creates_hot_lead(inv, mem):
    mem.ensure_user("omar")
    sid = mem.create_session("omar")
    day = next_weekday(1)
    agent = FakeAgent(inv, mem, [
        tool_call("book_viewing", {"car_id": 3, "date": day, "time": "11:00", "name": "Omar", "phone": "0501234567"}),
        text("Booked!"),
    ])
    out = agent.chat("omar", sid, "book it")
    assert out["tool_calls"][0]["ok"] is True
    assert leads.list_leads()[0]["lead_score"] == "hot"


def test_competitor_question_never_reaches_the_llm(inv, mem):
    mem.ensure_user("x")
    sid = mem.create_session("x")
    agent = FakeAgent(inv, mem, [])                              # empty script: any LLM call would crash
    out = agent.chat("x", sid, "is dubicars better?")
    assert out["guardrail"] is True and "dubizzle" in out["reply"]


def test_competitor_in_llm_output_is_replaced(inv, mem):
    mem.ensure_user("x")
    sid = mem.create_session("x")
    agent = FakeAgent(inv, mem, [text("You could also check OLX!")])
    assert "OLX" not in agent.chat("x", sid, "any other ideas?")["reply"]


def test_unknown_session_raises(inv, mem):
    with pytest.raises(KeyError):
        FakeAgent(inv, mem, []).chat("x", "nope", "hi")


def test_greeting_for_new_user_is_generic():
    assert "Dubi" in build_greeting({"known": False}, False)