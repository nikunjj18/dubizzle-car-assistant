"""Central configuration. Everything tweakable lives here or in .env."""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INVENTORY_PATH = Path(os.getenv("INVENTORY_PATH", DATA_DIR / "cars.xlsx"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "memory.db"))
LEADS_CSV = Path(os.getenv("LEADS_CSV", DATA_DIR / "leads.csv"))
BOOKINGS_CSV = Path(os.getenv("BOOKINGS_CSV", DATA_DIR / "bookings.csv"))

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini/gemini-2.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "5"))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "16"))  # messages sent back to the LLM

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Business rules (from the assessment brief)
LOCAL_TZ = ZoneInfo("Asia/Dubai")
OPEN_HOUR, CLOSE_HOUR = 8, 20          # 8am - 8pm
OPEN_WEEKDAYS = {0, 1, 2, 3, 4, 5}     # Monday=0 ... Saturday=5 (Sunday closed)
SLOT_MINUTES = 60
USD_TO_AED = 3.6725                    # AED is pegged to USD