from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionStart(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=40, description="Stable user identifier (e.g. a username)")
    name: Optional[str] = Field(None, max_length=60)


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    is_returning: bool
    greeting: str
    memory: dict[str, Any]


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    cars: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    guardrail: bool = False
    state: dict[str, Any] = {}


class FavoriteRequest(BaseModel):
    car_id: int
    reason: Optional[str] = None