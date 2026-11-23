"""Tools the LLM can call (function calling) + their implementations.

Design rule: the LLM never touches data directly. It asks for a tool, the tool
returns grounded JSON from the inventory/DB, and the LLM verbalises it. That is
what keeps answers tied to real listings (no hallucinated cars/prices).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from . import leads
from .config import USD_TO_AED
from .inventory import Inventory
from .memory import MemoryStore


@dataclass
class ToolContext:
    user_id: str
    session_id: str
    state: dict[str, Any]                 # short-term: {"shown": [ids], "focus": id}
    inventory: Inventory
    memory: MemoryStore
    cards: dict[int, dict[str, Any]] = field(default_factory=dict)   # cars to render in the UI this turn


TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search_inventory",
        "description": (
            "Search the dubizzle car inventory. Use for ANY request to find, list, filter or compare cars. "
            "Combine structured filters with a free-text query for things like 'sporty', 'luxury', 'family', "
            "'low mileage', 'panoramic roof'. Returns real listings only."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Free-text keywords/semantics, e.g. 'luxury convertible leather seats'"},
            "make": {"type": "string", "description": "e.g. 'Mercedes', 'Toyota', 'Land Rover'"},
            "model": {"type": "string", "description": "e.g. 'Cayenne', 'Pajero', 'C-Class'"},
            "body_type": {"type": "string", "enum": ["suv", "sedan", "coupe", "convertible", "hatchback", "wagon", "pickup", "van"]},
            "color": {"type": "string", "description": "white, black, silver, grey, red, blue, green, brown, gold, beige, orange, yellow"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"},
            "min_price": {"type": "number"}, "max_price": {"type": "number"},
            "currency": {"type": "string", "enum": ["AED", "USD"], "description": "Currency of min/max price. Default AED."},
            "max_mileage_km": {"type": "integer"},
            "specs": {"type": "string", "enum": ["GCC", "Japanese", "American", "European", "Canadian"]},
            "warranty_only": {"type": "boolean"},
            "limit": {"type": "integer", "description": "Max results (default 5, max 8)"},
        }}}},
    {"type": "function", "function": {
        "name": "get_car_details",
        "description": "Get full details (description, price, mileage, warranty, specs) for ONE listing by car_id. "
                       "Use for any follow-up question about a specific car.",
        "parameters": {"type": "object", "properties": {"car_id": {"type": "integer"}}, "required": ["car_id"]}}},
    {"type": "function", "function": {
        "name": "check_viewing_availability",
        "description": "List free viewing/test-drive slots for a car on a date. Viewings: Monday-Saturday, 8am-8pm, 1-hour slots, "
                       "must be pre-booked. Always call this before offering times.",
        "parameters": {"type": "object", "properties": {
            "car_id": {"type": "integer"}, "date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["car_id", "date"]}}},
    {"type": "function", "function": {
        "name": "book_viewing",
        "description": "Book a viewing or test drive. Only call once you have the car, an exact date+time, the customer's name "
                       "and phone number, and they have confirmed.",
        "parameters": {"type": "object", "properties": {
            "car_id": {"type": "integer"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM 24h, e.g. 14:00"},
            "name": {"type": "string"}, "phone": {"type": "string"},
            "kind": {"type": "string", "enum": ["viewing", "test_drive"]}},
            "required": ["car_id", "date", "time", "name", "phone"]}}},
    {"type": "function", "function": {
        "name": "update_user_profile",
        "description": "Save what you learn about the customer (name, phone, budget, needs). Call it whenever the user reveals "
                       "ANY of these - it updates long-term memory AND records/qualifies the sales lead in the leads CSV.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "phone": {"type": "string"},
            "budget_min": {"type": "number"}, "budget_max": {"type": "number"},
            "currency": {"type": "string", "enum": ["AED", "USD"], "description": "Currency of the budget. Default AED."},
            "body_types": {"type": "array", "items": {"type": "string"}},
            "preferred_makes": {"type": "array", "items": {"type": "string"}},
            "preferred_colors": {"type": "array", "items": {"type": "string"}},
            "must_haves": {"type": "array", "items": {"type": "string"}, "description": "e.g. 'low mileage', 'warranty', 'GCC specs'"},
            "purpose": {"type": "string", "description": "e.g. 'daily commute', 'family car', 'weekend fun'"},
            "timeline": {"type": "string", "description": "When they want to buy, e.g. 'this month'"},
            "financing": {"type": "string", "enum": ["cash", "finance", "undecided"]}}}}},
    {"type": "function", "function": {
        "name": "save_favorite_car",
        "description": "Remember a car the user likes/shortlists so it can be recalled in future sessions.",
        "parameters": {"type": "object", "properties": {"car_id": {"type": "integer"}, "reason": {"type": "string"}},
                       "required": ["car_id"]}}},
]


def _aed(amount: Optional[float], currency: Optional[str]) -> Optional[float]:
    if amount is None:
        return None
    return round(amount * USD_TO_AED) if (currency or "AED").upper() == "USD" else amount


def _criteria_text(a: dict[str, Any]) -> str:
    parts = [str(a[k]) for k in ("color", "make", "model", "body_type", "specs") if a.get(k)]
    if a.get("query"):
        parts.append(f"'{a['query']}'")
    if a.get("min_year") or a.get("max_year"):
        parts.append(f"year {a.get('min_year', '')}-{a.get('max_year', '')}")
    if a.get("max_price") or a.get("min_price"):
        cur = a.get("currency", "AED")
        parts.append(f"price {a.get('min_price', '')}-{a.get('max_price', '')} {cur}")
    if a.get("max_mileage_km"):
        parts.append(f"under {a['max_mileage_km']} km")
    return " ".join(parts) or "all cars"


def _interested_ids(ctx: ToolContext) -> list[int]:
    ids = [f["car_id"] for f in ctx.memory.get_favorites(ctx.user_id)]
    ids += [e["car_id"] for e in ctx.memory.recent_events(ctx.user_id, "view", 10) if "car_id" in e]
    return ids


def _sync_lead(ctx: ToolContext, booked: bool = False) -> dict[str, Any]:
    user = ctx.memory.get_user(ctx.user_id) or {"name": None, "profile": {}}
    return leads.upsert_lead(ctx.user_id, user["name"], user["profile"], _interested_ids(ctx), booked)


# ------------------------------- implementations ------------------------------- #
def search_inventory(ctx: ToolContext, **a: Any) -> dict[str, Any]:
    cur = a.get("currency")
    res = ctx.inventory.search(
        query=a.get("query"), make=a.get("make"), model=a.get("model"), body_type=a.get("body_type"),
        color=a.get("color"), min_year=a.get("min_year"), max_year=a.get("max_year"),
        min_price_aed=_aed(a.get("min_price"), cur), max_price_aed=_aed(a.get("max_price"), cur),
        max_mileage_km=a.get("max_mileage_km"), specs=a.get("specs"),
        warranty_only=bool(a.get("warranty_only")), limit=a.get("limit", 5),
    )
    ids = [c["car_id"] for c in res["results"]]
    shown = list(ids)
    for c in res["results"]:
        ctx.cards[c["car_id"]] = c
    if len(ids) < 3:                                   # few confirmed matches: also show the 'possible' ones
        for c in res["unverified_matches"]:
            shown.append(c["car_id"])
            ctx.cards[c["car_id"]] = c
    ctx.state["shown"] = shown                         # numbering the user sees: "the first one" = shown[0]
    ctx.state["focus"] = shown[0] if len(shown) == 1 else ctx.state.get("focus")
    ctx.memory.log_event(ctx.user_id, ctx.session_id, "search",
                         {"criteria": _criteria_text(a), "matches": res["total_matches"]})
    res["note"] = ("`results` CONFIRM every criterion. `unverified_matches` fit the other criteria but their listing does not "
                   "state the fields in `not_stated_in_listing` - present them as 'possible matches', never as confirmed. "
                   "Present cars in the order returned (results first) so users can say 'the first one'.")
    return res


def get_car_details(ctx: ToolContext, car_id: int, **_: Any) -> dict[str, Any]:
    d = ctx.inventory.details(int(car_id))
    if d is None:
        return {"error": f"No listing with car_id {car_id}. Do not invent one."}
    ctx.state["focus"] = int(car_id)
    ctx.cards[int(car_id)] = ctx.inventory.summary(int(car_id))
    ctx.memory.log_event(ctx.user_id, ctx.session_id, "view", {"car_id": int(car_id), "name": d["name"]})
    return d


def check_viewing_availability(ctx: ToolContext, car_id: int, date: str, **_: Any) -> dict[str, Any]:
    if not ctx.inventory.exists(int(car_id)):
        return {"error": f"No listing with car_id {car_id}."}
    try:
        day = leads.parse_day(date)
    except ValueError:
        return {"error": "Invalid date. Use YYYY-MM-DD."}
    ctx.state["focus"] = int(car_id)
    return {"car_id": int(car_id), "car": ctx.inventory.df.loc[int(car_id), "display_name"], **leads.available_slots(int(car_id), day)}


def book_viewing(ctx: ToolContext, car_id: int, date: str, time: str, name: str, phone: str,
                 kind: str = "test_drive", **_: Any) -> dict[str, Any]:
    if not ctx.inventory.exists(int(car_id)):
        return {"success": False, "error": f"No listing with car_id {car_id}."}
    try:
        day = leads.parse_day(date)
    except ValueError:
        return {"success": False, "error": "Invalid date. Use YYYY-MM-DD."}
    car_name = ctx.inventory.df.loc[int(car_id), "display_name"]
    res = leads.book_slot(ctx.user_id, name, phone, int(car_id), car_name, day, time, kind)
    if res.get("success"):
        ctx.memory.update_profile(ctx.user_id, {"name": name, "phone": phone})
        ctx.memory.log_event(ctx.user_id, ctx.session_id, "booking",
                             {"car_id": int(car_id), "name": car_name, "slot": res["slot_start"], "kind": kind})
        lead = _sync_lead(ctx, booked=True)
        res["lead_score"] = lead["lead_score"]
    return res


def update_user_profile(ctx: ToolContext, **a: Any) -> dict[str, Any]:
    cur = a.pop("currency", None)
    updates = {
        "name": a.get("name"), "phone": a.get("phone"),
        "budget_min_aed": _aed(a.get("budget_min"), cur), "budget_max_aed": _aed(a.get("budget_max"), cur),
        "body_types": [x.lower() for x in (a.get("body_types") or [])],
        "preferred_makes": a.get("preferred_makes"),
        "preferred_colors": [x.lower() for x in (a.get("preferred_colors") or [])],
        "must_haves": a.get("must_haves"), "purpose": a.get("purpose"),
        "timeline": a.get("timeline"), "financing": a.get("financing"),
    }
    profile = ctx.memory.update_profile(ctx.user_id, updates)
    lead = _sync_lead(ctx)
    return {"saved": True, "profile": profile, "lead_id": lead["lead_id"], "lead_score": lead["lead_score"],
            "lead_score_reason": lead["score_reason"]}


def save_favorite_car(ctx: ToolContext, car_id: int, reason: Optional[str] = None, **_: Any) -> dict[str, Any]:
    if not ctx.inventory.exists(int(car_id)):
        return {"error": f"No listing with car_id {car_id}."}
    name = ctx.inventory.df.loc[int(car_id), "display_name"]
    ctx.memory.add_favorite(ctx.user_id, int(car_id), name, reason)
    _sync_lead(ctx)
    return {"saved": True, "car_id": int(car_id), "name": name}


_IMPL = {f.__name__: f for f in (search_inventory, get_car_details, check_viewing_availability, book_viewing,
                                 update_user_profile, save_favorite_car)}


def execute_tool(ctx: ToolContext, name: str, raw_args: Any) -> dict[str, Any]:
    fn = _IMPL.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}"}
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
    except json.JSONDecodeError:
        return {"error": "Tool arguments were not valid JSON."}
    try:
        return fn(ctx, **args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # never let a tool crash the chat turn
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}