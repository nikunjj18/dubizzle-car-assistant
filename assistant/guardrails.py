"""Deterministic guardrails that run in code, *around* the LLM.

Prompt instructions alone can be talked around, so the two hardest rules
(never mention competitors, ignore prompt-injection) are also enforced here.
Off-topic handling is done by the LLM (it needs language understanding) and
backed by the system prompt.
"""
from __future__ import annotations

import re
from typing import Optional

COMPETITORS = [
    "dubicars", "yallamotor", "yalla motor", "opensooq", "open sooq", "carswitch", "car switch", "autotrader",
    "auto trader", "cars24", "carmudi", "autodeal", "hatla2ee", "carsome", "olx", "cargurus", "cars.com",
    "carvana", "ebay motors", "facebook marketplace", "dealshaker", "motory", "syarah", "haraj", "carlist",
    "autotempest", "kavak", "expatriates", "dubai drive", "justdubai",
]
_COMP_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(c) for c in COMPETITORS) + r")(?![a-z0-9])", re.I)

_INJECTION_RE = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)|"
    r"disregard (your|the) (instructions|rules|system prompt)|reveal (your|the) (system )?prompt|"
    r"show (me )?(your|the) (system )?prompt|you are now (?!looking)|pretend (to be|you are)|jailbreak|\bDAN mode\b",
    re.I,
)

MAX_INPUT_CHARS = 1500

COMPETITOR_REPLY = (
    "I can only speak about cars listed on dubizzle, so I can't compare or comment on other platforms. "
    "I'm happy to dig into any of our listings with you though - want me to narrow down by budget, body type or make?"
)
INJECTION_REPLY = (
    "I'm dubizzle's car assistant, so I'll stay focused on helping you find and view cars. "
    "Tell me what you're looking for - budget, body type, make - and I'll search our listings."
)
TOO_LONG_REPLY = "That message is a bit long for me - could you shorten it to the key details of the car you're after?"


def check_input(text: str) -> Optional[str]:
    """Return a canned reply if the message must NOT reach the LLM, else None."""
    if len(text) > MAX_INPUT_CHARS:
        return TOO_LONG_REPLY
    if _INJECTION_RE.search(text):
        return INJECTION_REPLY
    if _COMP_RE.search(text):
        return COMPETITOR_REPLY
    return None


def check_output(text: str) -> str:
    """Last line of defence: never let a competitor name reach the user."""
    if _COMP_RE.search(text):
        return COMPETITOR_REPLY
    return text