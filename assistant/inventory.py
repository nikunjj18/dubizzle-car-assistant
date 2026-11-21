"""Inventory loading, enrichment and hybrid search.

The supplied dataset has NO price / mileage / warranty columns - those facts only
exist inside the free-text title/description (often messy, sometimes Arabic).
So at startup we derive structured fields with conservative regexes, and keep a
TF-IDF index over the text for semantic-ish ranking. Search = hard structured
filters (make, year, price...) + TF-IDF ranking for the free-text part.

Anything we cannot extract stays None and the agent is told to say "not listed"
rather than guess - this is the core anti-hallucination mechanism.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --------------------------------------------------------------------------- #
# Text cleaning + field extraction
# --------------------------------------------------------------------------- #
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: Any) -> str:
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = _TAG.sub(" ", s)
    s = s.replace("\\t", " ").replace("•", " ")
    return _WS.sub(" ", s).strip()


_NUM = r"(\d{1,3}(?:,\d{3})+|\d{4,8})(?:\.\d+)?"
_PRICE_KEYWORD = re.compile(
    r"(?:price|cash|asking|aed|dhs?|dirhams?|درهم)\s*[:\-]?\s*(?:aed\s*)?" + _NUM, re.I
)
_PRICE_SUFFIX = re.compile(_NUM + r"\s*(?:aed|dhs?|dirhams?|درهم)\b", re.I)
_CASH = re.compile(_NUM + r"\s*(?:in\s+)?cash", re.I)
_MONTHLY_CONTEXT = re.compile(
    r"month|/\s*m\b|monthly|down\s*payment|\bdp\b|emi|instal|deposit|per\s+m|insurance|fee|"
    r"service|warranty\s+cost|مقدم|شهري",
    re.I,
)


def _to_int(raw: str) -> int:
    return int(raw.replace(",", "").split(".")[0])


def extract_price(title: str, desc: str) -> tuple[Optional[int], Optional[int]]:
    """Return (price_aed, monthly_payment_aed). Conservative: None when unsure."""
    text = f"{title} | {desc}"
    monthly: Optional[int] = None
    for m in re.finditer(_NUM + r"[^|]{0,25}?(?:monthly|/\s*month|per\s+month|a\s+month)", text, re.I):
        v = _to_int(m.group(1))
        if 300 <= v <= 60_000:
            monthly = v
            break
    if monthly is None:
        for m in re.finditer(r"(?:monthly|per month|/\s*month)[^|\d]{0,25}" + _NUM, text, re.I):
            v = _to_int(m.group(1))
            if 300 <= v <= 60_000:
                monthly = v
                break

    # 1) explicit cash price wins
    for m in _CASH.finditer(text):
        v = _to_int(m.group(1))
        if 8_000 <= v <= 20_000_000:
            return v, monthly

    # 2) AED/price keyword candidates, skipping monthly / down-payment context
    candidates: list[int] = []
    for pat in (_PRICE_KEYWORD, _PRICE_SUFFIX):
        for m in pat.finditer(text):
            v = _to_int(m.group(1))
            if not 8_000 <= v <= 20_000_000:
                continue
            ctx = text[max(0, m.start() - 25) : m.end() + 30]
            if _MONTHLY_CONTEXT.search(ctx):
                continue
            if _KM_AFTER.match(text[m.end() : m.end() + 12]):
                continue  # "12,000 km" is mileage, not a price
            candidates.append(v)
    return (candidates[0] if candidates else None), monthly


_KM_AFTER = re.compile(r"\s*(?:km|kms|kilomet|كم)", re.I)
_NUM_KM = r"(\d{1,3}(?:,\d{3})+|\d{1,7})(?:\.\d+)?"
_KM = [
    re.compile(r"(?:mileage|odometer|kilomet\w*|millage|المسافة)\s*[:\-=]?\s*" + _NUM_KM, re.I),
    re.compile(r"(\d{1,3}(?:,\d{3})+|\d{2,7})\s*(?:km|kms|kilomet\w*|كم)\b", re.I),
    re.compile(r"\b(\d{2,3})\s*k\s*(?:km|kms)\b", re.I),
]


def extract_mileage(title: str, desc: str) -> Optional[int]:
    text = f"{title} | {desc}"
    if re.search(r"\b0\s*km\b", text, re.I):
        return 0
    for pat in _KM:
        m = pat.search(text)
        if m:
            v = _to_int(m.group(1))
            if pat is _KM[2]:
                v *= 1000
            if 0 <= v <= 600_000:
                return v
    return None


_WARRANTY = re.compile(r"[^.|•\n]{0,45}(?:warrant\w*|ضمان)[^.|•\n]{0,45}", re.I)
_NO_WARRANTY = re.compile(r"\bno\s+warranty\b|without\s+warranty", re.I)


def extract_warranty(title: str, desc: str) -> tuple[Optional[bool], Optional[str]]:
    text = f"{title}. {desc}"
    if _NO_WARRANTY.search(text):
        return False, "Listing says no warranty"
    m = _WARRANTY.search(text)
    if not m:
        return None, None  # unknown, NOT "no warranty"
    return True, _WS.sub(" ", m.group(0)).strip()


_SPECS = {
    "GCC": r"\bgcc\b|خليجي",
    "Japanese": r"japan\w*|ياباني",
    "American": r"\bamerican\b|\busa\b|\bus\s+spec|امريكي|أمريكي",
    "European": r"european|\beuro(?:pean)?\s+spec|\bgerman\s+spec",
    "Canadian": r"canad\w+",
}


def extract_specs(text: str) -> Optional[str]:
    for name, pat in _SPECS.items():
        if re.search(pat, text, re.I):
            return name
    return None


_COLORS = {
    "white": r"\bwhite\b|أبيض|ابيض|\bpearl\b",
    "black": r"\bblack\b|أسود|اسود",
    "silver": r"\bsilver\b|فضي",
    "grey": r"\bgr[ae]y\b|رمادي|\bgunmetal\b",
    "red": r"\bred\b|أحمر|احمر",
    "blue": r"\bblue\b|أزرق|ازرق",
    "green": r"\bgreen\b|أخضر|اخضر",
    "brown": r"\bbrown\b|بني",
    "gold": r"\bgold(?:en)?\b|ذهبي",
    "beige": r"\bbeige\b|بيج",
    "orange": r"\borange\b|برتقالي",
    "yellow": r"\byellow\b|أصفر|اصفر",
}


def extract_colors(title: str, desc: str) -> list[str]:
    """Colours in the title are most reliable; fall back to description mentions."""
    found = [c for c, p in _COLORS.items() if re.search(p, title, re.I)]
    if found:
        return found
    return [c for c, p in _COLORS.items() if re.search(p, desc[:600], re.I)]


_BODY_BY_MODEL = [
    ("convertible", r"convertible|spider|roadster|volante|drophead|\bgtc\b|cabrio|\bdawn\b"),
    ("pickup", r"landtrek|pickup|hilux|\bf-?150\b"),
    ("van", r"hiace|\bvan\b"),
    ("suv", r"\bdbx\b|\bq7\b|bentayga|\bx[1-7]\b|explorer|hummer|tucson|f-pace|wrangler|kaiyi|sportage|"
            r"lr4|range rover|lx-series|\blx\b|aviator|mkx|levante|glc|glk|gls|g-class|countryman|pajero|"
            r"patrol|x-trail|xterra|grandland|cayenne|cullinan|highlander|land cruiser|rav|xc60|territory|"
            r"\bh9\b|velar|\bq[3-8]\b|navigator"),
    ("hatchback", r"megane|golf|beetle|\bcooper\b|\bmini\b"),
    ("coupe", r"camaro|challenger|mustang|vanquish|aventador|revuelto|sf90|675lt|f430|amg gt|\bslr\b|"
              r"wraith|brooklands|\ba5\b|\bamg\b|coupe|huracan|\bgt\b"),
    ("sedan", r"3-series|\bg90\b|genesis|\bq50\b|\bxe\b|c-class|e-class|s-class|altima|versa|camry|"
              r"ghibli|ghost|phantom|flying spur|continental|sedan|\ba[3-8]\b"),
]


def infer_body_type(make: str, model: str, trim: str, title: str) -> str:
    blob = f"{make} {model} {trim}".lower()
    for body, pat in _BODY_BY_MODEL:
        if re.search(pat, blob, re.I):
            return body
    for body, pat in _BODY_BY_MODEL:
        if re.search(pat, title.lower(), re.I):
            return body
    return "unknown"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower().replace("-", " ")).strip()