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
    return int(re.sub(r"[,\s]", "", raw).split(".")[0])

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
_NUM_KM = r"(\d{1,3}(?:[,\s]\d{3})+|\d{1,7})(?:\.\d+)?"
_NOT_SPEED = r"(?!\s*/\s*h|\s*per\s*h)"  # "318km/h" is a top speed, not mileage
_KM = [
    re.compile(r"(?:mileage|odometer|kilomet\w*|millage|المسافة)\s*[:\-=]?\s*" + _NUM_KM, re.I),
    re.compile(r"(\d{1,3}(?:[,\s]\d{3})+|\d{2,7})\s*(?:km|kms|kilomet\w*|كم)\b" + _NOT_SPEED, re.I),
    re.compile(r"\b(\d{2,3})\s*k\s*(?:km|kms)\b" + _NOT_SPEED, re.I),
]


def extract_mileage(title: str, desc: str) -> Optional[int]:
    text = f"{title} | {desc}"
    if re.search(r"\b0\s*km\b" + _NOT_SPEED, text, re.I):
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



# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
class Inventory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.df = self._load()
        self._build_index()

    # ---- loading -------------------------------------------------------- #
    def _load(self) -> pd.DataFrame:
        xl = pd.ExcelFile(self.path)
        sheet = next((s for s in xl.sheet_names if "clean" in s.lower()), xl.sheet_names[0])
        df = pd.read_excel(self.path, sheet_name=sheet)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "listing_id" not in df.columns:
            df.insert(0, "listing_id", range(1, len(df) + 1))
        df = df.rename(columns={"listing_id": "car_id"})
        df["car_id"] = df["car_id"].astype(int)

        for col in ("make", "model", "trim", "title", "description", "photo_url"):
            df[col] = df.get(col, "").apply(clean_text) if col != "photo_url" else df.get(col, "").fillna("")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

        prices, monthlies, kms, wars, warnotes, specs, colors, bodies = ([] for _ in range(8))
        for r in df.itertuples():
            p, mo = extract_price(r.title, r.description)
            prices.append(p); monthlies.append(mo)
            kms.append(extract_mileage(r.title, r.description))
            w, wn = extract_warranty(r.title, r.description)
            wars.append(w); warnotes.append(wn)
            specs.append(extract_specs(f"{r.title} {r.description}"))
            colors.append(extract_colors(r.title, r.description))
            bodies.append(infer_body_type(r.make, r.model, r.trim, r.title))
        df["price_aed"] = pd.array(prices, dtype="Int64")
        df["monthly_payment_aed"] = pd.array(monthlies, dtype="Int64")
        df["mileage_km"] = pd.array(kms, dtype="Int64")
        df["warranty"] = wars
        df["warranty_note"] = warnotes
        df["specs"] = specs
        df["colors"] = colors
        df["body_type"] = bodies
        df["display_name"] = (
            df["year"].astype(str) + " " + df["make"].str.title() + " " + df["model"].str.title()
            + df["trim"].apply(lambda t: "" if t in ("", "other", "standard") else " " + t.upper())
        )
        df["_norm"] = (df["make"] + " " + df["model"] + " " + df["trim"] + " " + df["title"]).apply(_norm)
        return df.set_index("car_id", drop=False)

    def _build_index(self) -> None:
        blob = (
            self.df["display_name"] + " " + self.df["title"] + " " + self.df["body_type"] + " "
            + self.df["colors"].apply(" ".join) + " " + self.df["specs"].fillna("") + " " + self.df["description"]
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform(blob)
        self._pos = {cid: i for i, cid in enumerate(self.df.index)}

    # ---- serialisation -------------------------------------------------- #
    @staticmethod
    def _val(v: Any) -> Any:
        if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
            return None
        if hasattr(v, "item"):
            return v.item()
        return v

    def summary(self, car_id: int) -> dict[str, Any]:
        r = self.df.loc[car_id]
        return {
            "car_id": int(car_id),
            "name": r.display_name,
            "title": r.title,
            "year": int(r.year),
            "make": r.make.title(),
            "model": r.model.title(),
            "trim": r.trim,
            "body_type": r.body_type,
            "price_aed": self._val(r.price_aed),
            "monthly_payment_aed": self._val(r.monthly_payment_aed),
            "mileage_km": self._val(r.mileage_km),
            "specs": self._val(r.specs),
            "colors": list(r.colors),
            "warranty": self._val(r.warranty),
            "photo_url": r.photo_url,
        }

    def details(self, car_id: int) -> Optional[dict[str, Any]]:
        if car_id not in self.df.index:
            return None
        d = self.summary(car_id)
        r = self.df.loc[car_id]
        d["warranty_note"] = self._val(r.warranty_note)
        d["description"] = r.description[:1800]
        d["field_notes"] = (
            "price/mileage/warranty/specs/colors/body_type were derived automatically from the listing text; "
            "null means NOT STATED in the listing. Never guess missing values."
        )
        return d

    def exists(self, car_id: int) -> bool:
        return car_id in self.df.index

    # ---- search --------------------------------------------------------- #
    def search(
        self,
        query: Optional[str] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        body_type: Optional[str] = None,
        color: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_price_aed: Optional[float] = None,
        max_price_aed: Optional[float] = None,
        max_mileage_km: Optional[int] = None,
        specs: Optional[str] = None,
        warranty_only: bool = False,
        limit: int = 5,
    ) -> dict[str, Any]:
        df = self.df
        mask = pd.Series(True, index=df.index)

        if make:
            mk = _norm(make)
            mask &= df["make"].apply(_norm).apply(lambda m: mk in m or m in mk)
        if model:
            toks = _norm(model).split()
            mask &= df["_norm"].apply(lambda s: all(t in s.split() or t in s for t in toks))
        if body_type and body_type.lower() not in ("any", "unknown"):
            bt = body_type.lower().replace("crossover", "suv")
            mask &= df["body_type"] == bt
        if color:
            c = color.lower()
            mask &= df["colors"].apply(lambda cs: c in cs)
        if min_year:
            mask &= df["year"] >= int(min_year)
        if max_year:
            mask &= df["year"] <= int(max_year)
        if specs:
            mask &= df["specs"].fillna("").str.lower() == specs.lower()
        if warranty_only:
            mask &= df["warranty"] == True  # noqa: E712

        excluded_unknown_price = excluded_unknown_km = 0
        if min_price_aed is not None or max_price_aed is not None:
            has_price = df["price_aed"].notna()
            excluded_unknown_price = int((mask & ~has_price).sum())
            mask &= has_price
            if min_price_aed is not None:
                mask &= df["price_aed"].fillna(0) >= min_price_aed
            if max_price_aed is not None:
                mask &= df["price_aed"].fillna(10**12) <= max_price_aed
        if max_mileage_km is not None:
            has_km = df["mileage_km"].notna()
            excluded_unknown_km = int((mask & ~has_km).sum())
            mask &= has_km
            mask &= df["mileage_km"].fillna(10**12) <= max_mileage_km

        ids = list(df.index[mask])
        if query and query.strip() and ids:
            sims = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
            ids.sort(key=lambda cid: sims[self._pos[cid]], reverse=True)
            # when the query adds nothing (all zero similarity) keep newest first
            if all(sims[self._pos[c]] == 0 for c in ids):
                ids.sort(key=lambda cid: -self.df.loc[cid, "year"])
        else:
            ids.sort(key=lambda cid: -self.df.loc[cid, "year"])

        limit = max(1, min(int(limit or 5), 8))
        return {
            "total_matches": len(ids),
            "results": [self.summary(c) for c in ids[:limit]],
            "excluded_because_price_not_listed": excluded_unknown_price,
            "excluded_because_mileage_not_listed": excluded_unknown_km,
        }

    def stats(self) -> dict[str, Any]:
        d = self.df
        return {
            "total_listings": int(len(d)),
            "with_price": int(d["price_aed"].notna().sum()),
            "with_mileage": int(d["mileage_km"].notna().sum()),
            "makes": sorted(d["make"].str.title().unique().tolist()),
            "body_types": d["body_type"].value_counts().to_dict(),
        }