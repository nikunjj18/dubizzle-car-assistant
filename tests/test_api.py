"""API tests: FastAPI TestClient, throwaway database, scripted fake LLM. No key, no network."""
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

import main
from assistant import leads
from assistant.agent import Agent
from assistant.memory import MemoryStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(leads, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(leads, "BOOKINGS_CSV", tmp_path / "bookings.csv")
    monkeypatch.setattr(main, "memory", MemoryStore(tmp_path / "api.db"))
    monkeypatch.setattr(main, "agent", Agent(main.inventory, main.memory))
    return TestClient(main.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["listings"] == 189


def test_new_then_returning_user_session(client):
    first = client.post("/sessions", json={"user_id": "Sara Ahmed", "name": "Sara"}).json()
    assert first["user_id"] == "sara_ahmed" and first["is_returning"] is False
    second = client.post("/sessions", json={"user_id": "sara_ahmed"}).json()
    assert second["is_returning"] is True and "Welcome back" in second["greeting"]
    assert second["session_id"] != first["session_id"]


def test_blank_user_id_is_rejected(client):
    assert client.post("/sessions", json={"user_id": "!!!"}).status_code == 400


def test_chat_guardrail_needs_no_llm(client):
    s = client.post("/sessions", json={"user_id": "sara"}).json()
    r = client.post("/chat", json={"user_id": "sara", "session_id": s["session_id"], "message": "is dubicars cheaper?"})
    assert r.status_code == 200 and r.json()["guardrail"] is True


def test_chat_with_fake_llm_returns_reply_and_cars(client, monkeypatch):
    calls = iter([
        NS(content=None, tool_calls=[NS(id="1", function=NS(name="search_inventory",
                                                            arguments=json.dumps({"make": "Porsche"})))]),
        NS(content="Found the Cayenne #8.", tool_calls=None),
    ])
    monkeypatch.setattr(main.agent, "_llm", lambda messages: NS(choices=[NS(message=next(calls))]))
    s = client.post("/sessions", json={"user_id": "sara"}).json()
    r = client.post("/chat", json={"user_id": "sara", "session_id": s["session_id"], "message": "porsche?"}).json()
    assert r["reply"] == "Found the Cayenne #8." and r["cars"][0]["car_id"] == 8
    assert r["tool_calls"][0]["tool"] == "search_inventory"


def test_chat_unknown_session_is_404(client):
    assert client.post("/chat", json={"user_id": "x", "session_id": "nope", "message": "hi"}).status_code == 404


def test_inventory_endpoints(client):
    res = client.get("/inventory", params={"q": "luxury convertible", "limit": 2}).json()
    assert len(res["results"]) == 2
    assert len(client.get("/inventory", params={"limit": 30}).json()["results"]) == 30     # the browse page can ask for more than the chatbot's 8
    assert client.get("/inventory/stats").json()["total_listings"] == 189
    assert client.get("/inventory/8").json()["name"].endswith("Cayenne GTS")
    assert client.get("/inventory/9999").status_code == 404


def test_memory_favorites_and_forget(client):
    client.post("/sessions", json={"user_id": "sara", "name": "Sara"})
    assert client.post("/users/sara/favorites", json={"car_id": 8}).json() == {"saved": True}
    assert client.post("/users/sara/favorites", json={"car_id": 9999}).status_code == 404
    mem = client.get("/users/sara/memory").json()
    assert mem["known"] and mem["favorites"][0]["car_id"] == 8
    assert client.delete("/users/sara/favorites/8").json() == {"removed": True}
    assert client.get("/users/sara/memory").json()["favorites"] == []
    client.post("/users/sara/favorites", json={"car_id": 8})
    client.delete("/users/sara/memory")
    assert client.get("/users/sara/memory").json()["known"] is False


def test_leads_and_bookings_start_empty(client):
    assert client.get("/leads").json() == [] and client.get("/bookings").json() == []