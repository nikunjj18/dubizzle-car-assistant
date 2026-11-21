"""FastAPI backend. Run:  uv run uvicorn main:app --reload
Interactive API docs (Swagger): http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from assistant import leads
from assistant.agent import Agent, build_greeting
from assistant.config import DB_PATH, INVENTORY_PATH, LLM_MODEL
from assistant.inventory import Inventory
from assistant.memory import MemoryStore, slug_user_id
from assistant.schemas import ChatRequest, ChatResponse, FavoriteRequest, SessionInfo, SessionStart

app = FastAPI(title="dubizzle Car Assistant API", version="1.0.0",
              description="Chat agent + inventory search + long-term memory + lead capture for dubizzle cars.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

inventory = Inventory(INVENTORY_PATH)
memory = MemoryStore(DB_PATH)
agent = Agent(inventory, memory)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model": LLM_MODEL, "listings": len(inventory.df)}


# ----------------------------- sessions & chat ----------------------------- #
@app.post("/sessions", response_model=SessionInfo, tags=["chat"])
def start_session(body: SessionStart):
    """Open a new chat session. Recognises returning users by `user_id` and greets them from memory."""
    user_id = slug_user_id(body.user_id)
    if not user_id:
        raise HTTPException(400, "user_id must contain letters or digits")
    _, returning = memory.ensure_user(user_id, body.name)
    session_id = memory.create_session(user_id)
    snap = memory.snapshot(user_id, current_session=session_id)
    greeting = build_greeting(snap, returning)
    memory.add_message(session_id, "assistant", greeting)
    return SessionInfo(session_id=session_id, user_id=user_id, is_returning=returning, greeting=greeting, memory=snap)


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(body: ChatRequest):
    try:
        return agent.chat(slug_user_id(body.user_id), body.session_id, body.message.strip())
    except KeyError:
        raise HTTPException(404, "Unknown session - start one with POST /sessions")
    except RuntimeError as e:
        raise HTTPException(503, str(e))


# ----------------------------- inventory ----------------------------- #
@app.get("/inventory", tags=["inventory"])
def search_inventory(q: Optional[str] = None, make: Optional[str] = None, model: Optional[str] = None,
                     body_type: Optional[str] = None, color: Optional[str] = None,
                     min_year: Optional[int] = None, max_year: Optional[int] = None,
                     max_price_aed: Optional[float] = None, max_mileage_km: Optional[int] = None,
                     limit: int = Query(10, ge=1, le=8 * 10)):
    """Direct (non-LLM) inventory search - same engine the agent's tool uses."""
    return inventory.search(query=q, make=make, model=model, body_type=body_type, color=color, min_year=min_year,
                            max_year=max_year, max_price_aed=max_price_aed, max_mileage_km=max_mileage_km, limit=limit)


@app.get("/inventory/stats", tags=["inventory"])
def inventory_stats():
    return inventory.stats()


@app.get("/inventory/{car_id}", tags=["inventory"])
def car_details(car_id: int):
    d = inventory.details(car_id)
    if d is None:
        raise HTTPException(404, "Listing not found")
    return d


# ----------------------------- users / memory ----------------------------- #
@app.get("/users/{user_id}/memory", tags=["memory"])
def user_memory(user_id: str):
    return memory.snapshot(slug_user_id(user_id))


@app.get("/users/{user_id}/sessions", tags=["memory"])
def user_sessions(user_id: str):
    return memory.list_sessions(slug_user_id(user_id))


@app.post("/users/{user_id}/favorites", tags=["memory"])
def add_favorite(user_id: str, body: FavoriteRequest):
    if not inventory.exists(body.car_id):
        raise HTTPException(404, "Listing not found")
    memory.add_favorite(slug_user_id(user_id), body.car_id, inventory.df.loc[body.car_id, "display_name"], body.reason)
    return {"saved": True}


@app.delete("/users/{user_id}/memory", tags=["memory"])
def forget_user(user_id: str):
    """Right-to-be-forgotten: wipes profile, sessions, favourites and events for this user."""
    memory.forget_user(slug_user_id(user_id))
    return {"deleted": True}


# ----------------------------- leads & bookings (admin view) ----------------------------- #
@app.get("/leads", tags=["leads"])
def get_leads():
    return leads.list_leads()


@app.get("/bookings", tags=["leads"])
def get_bookings():
    return leads.list_bookings()