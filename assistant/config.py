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

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-3.8-flash")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini/gemini-3.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))          # seconds per model call
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "5"))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "16"))  # messages sent back to the LLM

# --- RAG: listings table (SQLite) + vector database (ChromaDB) ---
LISTINGS_DB = Path(os.getenv("LISTINGS_DB", DATA_DIR / "listings.db"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-2")
EMBEDDING_FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL", "gemini/gemini-embedding-001")
USE_VECTOR_SEARCH = os.getenv("USE_VECTOR_SEARCH", "true").lower() in ("1", "true", "yes")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Business rules (from the assessment brief)
LOCAL_TZ = ZoneInfo("Asia/Dubai")
OPEN_HOUR, CLOSE_HOUR = 8, 20          # 8am - 8pm
OPEN_WEEKDAYS = {0, 1, 2, 3, 4, 5}     # Monday=0 ... Saturday=5 (Sunday closed)
SLOT_MINUTES = 60
USD_TO_AED = 3.6725                    # AED is pegged to USD