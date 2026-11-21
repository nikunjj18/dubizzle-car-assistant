"""Lead qualification (CSV) and viewing bookings (CSV + slot rules).

CSV is deliberate: the brief asks for "a local CSV simulating leads being
recorded". One row per user (upsert), re-scored every time we learn more.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import BOOKINGS_CSV, CLOSE_HOUR, LEADS_CSV, LOCAL_TZ, OPEN_HOUR, OPEN_WEEKDAYS, SLOT_MINUTES
from .memory import now_iso

LEAD_COLUMNS = [
    "lead_id", "user_id", "name", "phone", "budget_min_aed", "budget_max_aed", "body_types", "preferred_makes",
    "preferred_colors", "must_haves", "purpose", "timeline", "financing", "interested_car_ids", "viewing_booked",
    "lead_score", "score_reason", "created_at", "updated_at",
]
BOOKING_COLUMNS = ["booking_id", "user_id", "name", "phone", "car_id", "car_name", "slot_start", "slot_end",
                   "type", "status", "created_at"]

_lock = threading.Lock()


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame(columns=cols)


def _join(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    return "" if v is None else str(v)


def score_lead(profile: dict[str, Any], interested: list[Any], booked: bool) -> tuple[str, str]:
    """hot = viewing booked | warm = budget AND (preferences or liked cars) | cold = anything less."""
    has_budget = bool(profile.get("budget_max_aed") or profile.get("budget_min_aed"))
    has_needs = any(profile.get(k) for k in ("body_types", "preferred_makes", "preferred_colors", "must_haves", "purpose"))
    if booked:
        return "hot", "Viewing/test drive booked"
    if has_budget and (has_needs or interested):
        return "warm", "Budget and needs captured"
    if has_budget or has_needs or interested:
        return "cold", "Partial info only (missing budget or needs)"
    return "cold", "No qualifying info yet"


def upsert_lead(user_id: str, name: Optional[str], profile: dict[str, Any], interested_car_ids: list[int],
                booked: bool = False) -> dict[str, Any]:
    with _lock:
        df = _read(LEADS_CSV, LEAD_COLUMNS)
        score, reason = score_lead(profile, interested_car_ids, booked)
        existing = df[df["user_id"] == user_id]
        booked_prev = bool(len(existing)) and existing.iloc[0]["viewing_booked"] == "yes"
        row = {
            "lead_id": existing.iloc[0]["lead_id"] if len(existing) else f"L{len(df) + 1:04d}",
            "user_id": user_id, "name": name or "", "phone": profile.get("phone", ""),
            "budget_min_aed": profile.get("budget_min_aed", ""), "budget_max_aed": profile.get("budget_max_aed", ""),
            "body_types": _join(profile.get("body_types")), "preferred_makes": _join(profile.get("preferred_makes")),
            "preferred_colors": _join(profile.get("preferred_colors")), "must_haves": _join(profile.get("must_haves")),
            "purpose": profile.get("purpose", ""), "timeline": profile.get("timeline", ""),
            "financing": profile.get("financing", ""),
            "interested_car_ids": _join(sorted(set(interested_car_ids))),
            "viewing_booked": "yes" if (booked or booked_prev) else "no",
            "lead_score": "hot" if booked_prev else score, "score_reason": "Viewing/test drive booked" if booked_prev else reason,
            "created_at": existing.iloc[0]["created_at"] if len(existing) else now_iso(), "updated_at": now_iso(),
        }
        df = df[df["user_id"] != user_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[LEAD_COLUMNS]
        df.to_csv(LEADS_CSV, index=False)
        return row


def list_leads() -> list[dict[str, Any]]:
    return _read(LEADS_CSV, LEAD_COLUMNS).to_dict("records")


# ------------------------------ bookings ------------------------------ #
def today() -> date:
    return datetime.now(LOCAL_TZ).date()


def _slot_starts(day: date) -> list[datetime]:
    out, t = [], datetime.combine(day, time(OPEN_HOUR), tzinfo=LOCAL_TZ)
    end = datetime.combine(day, time(CLOSE_HOUR), tzinfo=LOCAL_TZ)
    while t + timedelta(minutes=SLOT_MINUTES) <= end:
        out.append(t)
        t += timedelta(minutes=SLOT_MINUTES)
    return out


def _booked_for(car_id: int, day: date) -> set[str]:
    df = _read(BOOKINGS_CSV, BOOKING_COLUMNS)
    if df.empty:
        return set()
    df = df[(df["car_id"] == str(car_id)) & (df["status"] == "confirmed")]
    return {s[11:16] for s in df["slot_start"] if s[:10] == day.isoformat()}


def parse_day(raw: str) -> date:
    return datetime.strptime(raw.strip()[:10], "%Y-%m-%d").date()


def available_slots(car_id: int, day: date) -> dict[str, Any]:
    if day.weekday() not in OPEN_WEEKDAYS:
        return {"date": day.isoformat(), "weekday": day.strftime("%A"), "open": False,
                "reason": "Viewings are only available Monday to Saturday (8am-8pm). Sunday is closed.", "slots": []}
    if day < today():
        return {"date": day.isoformat(), "weekday": day.strftime("%A"), "open": False,
                "reason": "That date is in the past.", "slots": []}
    taken, now = _booked_for(car_id, day), datetime.now(LOCAL_TZ)
    slots = [s.strftime("%H:%M") for s in _slot_starts(day) if s > now and s.strftime("%H:%M") not in taken]
    return {"date": day.isoformat(), "weekday": day.strftime("%A"), "open": True, "slots": slots,
            "slot_length_minutes": SLOT_MINUTES, "hours": f"{OPEN_HOUR}:00-{CLOSE_HOUR}:00"}


def book_slot(user_id: str, name: str, phone: str, car_id: int, car_name: str, day: date, hhmm: str,
              kind: str = "test_drive") -> dict[str, Any]:
    info = available_slots(car_id, day)
    if not info["open"]:
        return {"success": False, "error": info["reason"]}
    hhmm = hhmm.strip()[:5].rjust(5, "0")
    if hhmm not in info["slots"]:
        return {"success": False, "error": f"{hhmm} is not available on {day.isoformat()}.",
                "available_slots": info["slots"]}
    with _lock:
        df = _read(BOOKINGS_CSV, BOOKING_COLUMNS)
        start = datetime.combine(day, datetime.strptime(hhmm, "%H:%M").time(), tzinfo=LOCAL_TZ)
        row = {"booking_id": f"B{len(df) + 1:04d}", "user_id": user_id, "name": name, "phone": phone,
               "car_id": str(car_id), "car_name": car_name, "slot_start": start.isoformat(timespec="minutes"),
               "slot_end": (start + timedelta(minutes=SLOT_MINUTES)).isoformat(timespec="minutes"),
               "type": kind, "status": "confirmed", "created_at": now_iso()}
        pd.concat([df, pd.DataFrame([row])], ignore_index=True)[BOOKING_COLUMNS].to_csv(BOOKINGS_CSV, index=False)
    return {"success": True, **row}


def list_bookings() -> list[dict[str, Any]]:
    return _read(BOOKINGS_CSV, BOOKING_COLUMNS).to_dict("records")