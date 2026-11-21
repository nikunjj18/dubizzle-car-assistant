"""Persistent memory (SQLite, stdlib only).

Short-term memory  -> `messages` + `sessions.state_json` (shown cars / focus car)
Long-term memory   -> `users.profile_json`, `favorites`, `events` (searches, views, bookings)

Why SQLite: zero setup, ACID, inspectable with any DB browser, and good enough
for a prototype. The `MemoryStore` interface is small, so swapping to Postgres
or Redis later only touches this file.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import LOCAL_TZ

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY, name TEXT, profile_json TEXT DEFAULT '{}',
    created_at TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, user_id TEXT, started_at TEXT, state_json TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT, kind TEXT, payload_json TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS favorites (
    user_id TEXT, car_id INTEGER, car_name TEXT, reason TEXT, ts TEXT, PRIMARY KEY (user_id, car_id));
"""


def now_iso() -> str:
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def slug_user_id(raw: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", raw.strip().lower().replace(" ", "_"))[:40]


class MemoryStore:
    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- users / profile (long-term) ---------------- #
    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not r:
            return None
        return {"user_id": r["user_id"], "name": r["name"], "profile": json.loads(r["profile_json"] or "{}"),
                "created_at": r["created_at"], "last_seen": r["last_seen"]}

    def ensure_user(self, user_id: str, name: Optional[str] = None) -> tuple[dict[str, Any], bool]:
        """Return (user, is_returning)."""
        with self._lock:
            existing = self.get_user(user_id)
            with self._conn() as c:
                if existing:
                    c.execute("UPDATE users SET last_seen=?, name=COALESCE(?, name) WHERE user_id=?",
                              (now_iso(), name or None, user_id))
                else:
                    c.execute("INSERT INTO users(user_id,name,profile_json,created_at,last_seen) VALUES(?,?,?,?,?)",
                              (user_id, name, "{}", now_iso(), now_iso()))
            return self.get_user(user_id), existing is not None

    def update_profile(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge updates into the profile. Lists are unioned, scalars overwritten, None ignored."""
        with self._lock:
            user = self.get_user(user_id) or self.ensure_user(user_id)[0]
            profile = user["profile"]
            name = updates.pop("name", None)
            for k, v in updates.items():
                if v in (None, "", []):
                    continue
                if isinstance(v, list):
                    old = profile.get(k, [])
                    merged = old + [x for x in v if str(x).lower() not in {str(o).lower() for o in old}]
                    profile[k] = merged
                else:
                    profile[k] = v
            with self._conn() as c:
                c.execute("UPDATE users SET profile_json=?, name=COALESCE(?, name), last_seen=? WHERE user_id=?",
                          (json.dumps(profile), name, now_iso(), user_id))
            return {"name": name or user["name"], **profile}

    def forget_user(self, user_id: str) -> None:
        with self._lock, self._conn() as c:
            sids = [r["session_id"] for r in c.execute("SELECT session_id FROM sessions WHERE user_id=?", (user_id,))]
            for sid in sids:
                c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            for t in ("sessions", "events", "favorites", "users"):
                c.execute(f"DELETE FROM {t} WHERE user_id=?", (user_id,))

    # ---------------- sessions (short-term) ---------------- #
    def create_session(self, user_id: str) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO sessions(session_id,user_id,started_at,state_json) VALUES(?,?,?,?)",
                      (sid, user_id, now_iso(), json.dumps({"shown": [], "focus": None})))
        return sid

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not r:
            return None
        return {"session_id": r["session_id"], "user_id": r["user_id"], "started_at": r["started_at"],
                "state": json.loads(r["state_json"] or "{}")}

    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET state_json=? WHERE session_id=?", (json.dumps(state), session_id))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO messages(session_id,role,content,ts) VALUES(?,?,?,?)",
                      (session_id, role, content, now_iso()))

    def get_messages(self, session_id: str, limit: int = 16) -> list[dict[str, str]]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                             (session_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT s.session_id, s.started_at, COUNT(m.id) AS n FROM sessions s "
                "LEFT JOIN messages m ON m.session_id=s.session_id WHERE s.user_id=? "
                "GROUP BY s.session_id ORDER BY s.started_at DESC", (user_id,)).fetchall()
        return [{"session_id": r["session_id"], "started_at": r["started_at"], "messages": r["n"]} for r in rows]

    # ---------------- events + favorites (long-term) ---------------- #
    def log_event(self, user_id: str, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO events(user_id,session_id,kind,payload_json,ts) VALUES(?,?,?,?,?)",
                      (user_id, session_id, kind, json.dumps(payload), now_iso()))

    def recent_events(self, user_id: str, kind: Optional[str] = None, limit: int = 8,
                      exclude_session: Optional[str] = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM events WHERE user_id=?", [user_id]
        if kind:
            q += " AND kind=?"; args.append(kind)
        if exclude_session:
            q += " AND session_id!=?"; args.append(exclude_session)
        q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [{"kind": r["kind"], "ts": r["ts"], "session_id": r["session_id"],
                 **json.loads(r["payload_json"])} for r in rows]

    def add_favorite(self, user_id: str, car_id: int, car_name: str, reason: Optional[str] = None) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO favorites(user_id,car_id,car_name,reason,ts) VALUES(?,?,?,?,?)",
                      (user_id, car_id, car_name, reason, now_iso()))

    def remove_favorite(self, user_id: str, car_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM favorites WHERE user_id=? AND car_id=?", (user_id, car_id))

    def get_favorites(self, user_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT car_id, car_name, reason, ts FROM favorites WHERE user_id=? ORDER BY ts DESC",
                             (user_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------- the "what do we remember" view ---------------- #
    def snapshot(self, user_id: str, current_session: Optional[str] = None) -> dict[str, Any]:
        user = self.get_user(user_id)
        if not user:
            return {"user_id": user_id, "known": False}
        return {
            "user_id": user_id,
            "known": True,
            "name": user["name"],
            "profile": user["profile"],
            "favorites": self.get_favorites(user_id),
            "recent_searches": [e for e in self.recent_events(user_id, "search", 6, current_session)],
            "recent_views": [e for e in self.recent_events(user_id, "view", 6, current_session)],
            "bookings": [e for e in self.recent_events(user_id, "booking", 5)],
            "sessions": len(self.list_sessions(user_id)),
            "last_seen": user["last_seen"],
        }