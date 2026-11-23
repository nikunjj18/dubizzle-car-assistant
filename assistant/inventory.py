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

import numpy as np
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


_NUM = r"(\d{1,3}(?:[,.]\d{3})+|\d{4,8})(?:\.\d+)?"
_PRICE_KEYWORD = re.compile(
    r"(?:price|cash|asking|aed|dhs?|dirhams?|درهم)\s*[:\-#=]*\s*(?:aed\s*)?" + _NUM, re.I
)
_PRICE_SUFFIX = re.compile(_NUM + r"\s*(?:aed|dhs?|dirhams?|درهم)\b", re.I)
_CASH = re.compile(_NUM + r"\s*(?:in\s+)?cash", re.I)
# A number is NOT the price if a payment word sits right before or right after it.
_NOT_PRICE_BEFORE = re.compile(r"down\W*payment|instal|deposit|monthly|\bemi\b|\bfee\b|insurance|مقدم|شهري", re.I)
_NOT_PRICE_AFTER = re.compile(r"month|/\s*mo\b|monthly|per\s*m\b|instal|down|\bdp\b|مقدم|شهري", re.I)
_TITLE_TRAILING_PRICE = re.compile(r"(?<![\d,.:+#\-])(\d{5,6})\s*$")


def _to_int(raw: str) -> int:
    raw = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):          # "83.000" = 83,000 (dot used as thousands separator)
        return int(raw.replace(".", ""))
    return int(re.sub(r"[,\s]", "", raw).split(".")[0])


def _is_year(raw: str) -> bool:
    """A bare 4-digit number between 1980 and 2035 is a model year, not an amount."""
    return bool(re.fullmatch(r"\d{4}", raw.strip())) and 1980 <= int(raw) <= 2035


def extract_price(title: str, desc: str) -> tuple[Optional[int], Optional[int]]:
    """Return (price_aed, monthly_payment_aed). Conservative: None when unsure."""
    text = f"{title} | {desc}"
    monthly: Optional[int] = None
    # the number must be directly followed by the per-month wording: "1,430AED / Month", "4,000 P.M", "AED 713/month"
    adjacent = r"(?=\s*(?:aed|dhs?)?\s*(?:/\s*|per\s+|a\s+)?(?:months?\b|monthly(?!\s*(?:installments?|instalments?|payments?|plan))|mo\b|p\.?m\b))"
    for m in re.finditer(_NUM + adjacent, text, re.I):
        v = _to_int(m.group(1))
        if 300 <= v <= 60_000 and not _is_year(m.group(1)):
            monthly = v
            break
    if monthly is None:
        for m in re.finditer(r"(?:monthly|per month|/\s*month)[^|\d]{0,25}" + _NUM, text, re.I):
            v = _to_int(m.group(1))
            if 300 <= v <= 60_000 and not _is_year(m.group(1)):
                monthly = v
                break
    if monthly is None:
        for m in re.finditer(r"(?:installments?|instalments?|\bemi\b)\D{0,14}(\d{3,6})", text, re.I):
            v = int(m.group(1))
            if 300 <= v <= 60_000 and not _is_year(m.group(1)):
                monthly = v
                break

    # 1) explicit cash price wins
    for m in _CASH.finditer(text):
        v = _to_int(m.group(1))
        if 8_000 <= v <= 20_000_000:
            return v, monthly

    # 2) AED/price keyword candidates, skipping payment-plan numbers
    candidates: list[int] = []
    for pat in (_PRICE_KEYWORD, _PRICE_SUFFIX):
        for m in pat.finditer(text):
            v = _to_int(m.group(1))
            if not 8_000 <= v <= 20_000_000:
                continue
            if _NOT_PRICE_BEFORE.search(text[max(0, m.start() - 22) : m.start()]):
                continue
            if _NOT_PRICE_AFTER.search(text[m.end() : m.end() + 14]):
                continue
            if _KM_AFTER.match(text[m.end() : m.end() + 12]):
                continue  # "12,000 km" is mileage, not a price
            candidates.append(v)
    if candidates:
        return candidates[0], monthly

    # 3) last resort: an unlabeled 5-6 digit number ending the TITLE ("Mini Cooper 2018 GCC 120,000 km 30500")
    m = _TITLE_TRAILING_PRICE.search(title)
    if m and 8_000 <= int(m.group(1)) <= 2_000_000:
        return int(m.group(1)), monthly
    return None, monthly


_KM_AFTER = re.compile(r"\s*(?:km|kms|kilomet|كم)", re.I)
_NUM_KM = r"(\d{1,3}(?:[,.\s]\d{3})+|\d{1,7})(?:\.\d+)?"
_NOT_SPEED = r"(?!\s*/\s*h|\s*per\s*h)"  # "318km/h" is a top speed, not mileage
_KM = [
    re.compile(r"(?:mileage|odometer|kilomet\w*|millage|المسافة)\s*[:\-=]?\s*" + _NUM_KM, re.I),
    re.compile(r"(\d{1,3}(?:[,.\s]\d{3})+|\d{2,7})\s*(?:km|kms|kilomet\w*|كم)\b" + _NOT_SPEED, re.I),
    re.compile(r"\b(\d{2,3})\s*k\s*(?:km|kms)\b" + _NOT_SPEED, re.I),
]


_INTERVAL_AFTER = re.compile(r"\W*(?:or|/|per)\s*(?:\d+\s*)?(?:year|yr|month)", re.I)


_CAP_BEFORE = re.compile(r"(?:\bor|/|up\s*to|until|till|within|every|per|service\s*:?)\s*$", re.I)


def extract_mileage(title: str, desc: str) -> Optional[int]:
    """Mileage in km, or None. Ignores warranty/service limits ("warranty until 2029 or 100,000 km"),
    service intervals ("16000 km or 1 year"), top speeds ("318km/h") and model years ("Low Mileage 2024 Toyota")."""
    text = f"{title} | {desc}"
    if re.search(r"(?<![\d.,])0\s*km\b" + _NOT_SPEED, text, re.I):
        return 0
    for idx, pat in enumerate(_KM):
        for m in pat.finditer(text):
            if idx == 0 and _is_year(m.group(1)) and not _KM_AFTER.match(text[m.end(): m.end() + 6]):
                continue
            if idx > 0:
                if _INTERVAL_AFTER.match(text[m.end(): m.end() + 22]) or _CAP_BEFORE.search(text[max(0, m.start() - 14): m.start()]):
                    continue
            v = _to_int(m.group(1))
            if idx == 2:
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
    "Korean": r"korea\w*",
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


_CONVERTIBLE = r"convertible|soft\s*top|spider|spyder|roadster|volante|drophead|\bgtc\b|cabrio|\bdawn\b"
_BODY = {
    "pickup": r"landtrek|pickup|hilux|\bf-?150\b|tundra|wingle|tova|navara|ranger|silverado|\bl200\b|d-?max",
    "van": r"hiace|\bvan\b|sprinter|v-class|vito|urvan|transit",
    "suv": (r"\bdbx\b|\bq7\b|bentayga|\bx[1-7]\b|explorer|hummer|tucson|f-pace|wrangler|kaiyi|sportage|"
            r"lr4|range rover|lx-series|\blx\b|aviator|mkx|levante|glc|glk|gls|g-class|countryman|pajero|"
            r"patrol|x-trail|xterra|grandland|cayenne|cullinan|highlander|land cruiser|rav|xc60|territory|"
            r"\bh9\b|velar|\bq[3-8]\b|navigator|\bgle\b|\bgla\b|\bglb\b|tahoe|prado|kicks|captiva|cx-?5|t-roc|bz4x|"
            r"\bix3\b|urban cruiser|\bedge\b|discovery|\bjs4\b|\bt99\b|evoque|cr-?v|model x|\byu7\b"),
    "hatchback": r"megane|golf|beetle|\bcooper\b|\bmini\b|\ba-class\b|yaris|swift|picanto",
    "wagon": r"station|wagon|estate",
    "coupe": (r"camaro|challenger|mustang|vanquish|aventador|revuelto|sf90|675lt|f430|amg gt|\bslr\b|"
              r"wraith|brooklands|\ba5\b|coupe|huracan|\bgt\b|\bm4\b|\bm2\b|2-series|chiron|bugatti|\br8\b"),
    "sedan": (r"3-series|\bg90\b|genesis|\bq50\b|\bxe\b|c-class|e-class|s-class|altima|versa|camry|"
              r"ghibli|ghost|phantom|flying spur|sedan|\ba[3-8]\b|maxima|sunny|corolla|accent|arrizo|\bhan\b|es-series|"
              r"malibu|charger|\bm5\b|rs7|\bcla\b|cla-class|cls|5-series|7-series|elantra|sonata"),
}


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def infer_body_type(make: str, model: str, trim: str, title: str) -> str:
    """Priority: explicit 'convertible' words -> SUV/pickup/van models -> 'coupe'/'wagon' written in the title ->
    model-name mapping -> any keyword in the title. 'AMG' is a trim badge, NOT a body style."""
    blob = f"{make} {model} {trim}".lower()
    t = title.lower()
    if _has(_CONVERTIBLE, f"{blob} {t}"):
        return "convertible"
    for body in ("pickup", "van", "suv"):
        if _has(_BODY[body], blob):
            return body
    if _has(r"gran\s+coup", t):          # 4-door "Gran Coupé" (e.g. BMW i4) is sold as a sedan
        return "sedan"
    if _has(r"\bcoup[eé]\b", t):
        return "coupe"
    if _has(_BODY["wagon"], t):
        return "wagon"
    for body in ("hatchback", "coupe", "sedan"):
        if _has(_BODY[body], blob):
            return body
    for body, pat in _BODY.items():
        if _has(pat, t):
            return body
    return "unknown"


_BODY_ALIASES = {"crossover": "suv", "saloon": "sedan", "truck": "pickup", "estate": "wagon",
                 "station wagon": "wagon", "convertible": "convertible", "cabriolet": "convertible",
                 "roadster": "convertible", "spider": "convertible"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower().replace("-", " ")).strip()


# Plain keyword scoring doesn't know that "luxury" means Rolls-Royce/Bentley, so we expand a few
# common buyer intents into the brands/words that actually appear in listings. Deterministic and free.
_INTENT_EXPANSION = {
    r"luxur|premium|prestig|high.?end|executive": "bentley rolls royce aston martin maserati lamborghini ferrari mclaren "
                                                  "mercedes porsche range rover lexus genesis maybach full option",
    r"sport|fast|perform|speed|powerful|supercar|track": "ferrari lamborghini mclaren porsche amg gt camaro mustang challenger "
                                                        "v8 v12 turbo twin turbo hp horsepower",
    r"family|7.?seat|seven|seater|spacious|kids": "7 seats seven seater family spacious suv",
    r"economi|fuel|efficien|hybrid": "hybrid economical efficient",
    r"off.?road|4x4|adventure|desert": "4x4 off road awd wrangler pajero patrol land cruiser xterra",
    r"convertible|open.?top|roadster": "convertible roadster spider volante dawn soft top",
}


def expand_query(q: str) -> str:
    extra = [words for pat, words in _INTENT_EXPANSION.items() if re.search(pat, q, re.I)]
    return f"{q} {' '.join(extra)}" if extra else q


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
class Inventory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.df = self._load()
        self._build_index()
        self.rag: Any = None          # optional RagIndex (vector database); keyword search works without it

    def attach_rag(self, rag: Any) -> None:
        self.rag = rag

    # ---- loading -------------------------------------------------------- #
    def _read_sheets(self) -> pd.DataFrame:
        """Read EVERY sheet of the workbook (the file holds two different sets of listings).

        A sheet that has a Listing_ID column keeps those ids, so ids never change; sheets without one get
        sequential ids after the highest existing id. A listing that appears in more than one sheet is kept once.
        `source_sheet` and `excel_row` record where each listing lives, so any answer can be checked in Excel.
        """
        xl = pd.ExcelFile(self.path)
        frames = []
        for sheet in xl.sheet_names:
            d = pd.read_excel(self.path, sheet_name=sheet)
            d.columns = [str(c).strip().lower() for c in d.columns]
            if not {"make", "model", "year"} <= set(d.columns):
                continue
            d = d.dropna(subset=["make", "model"], how="all").copy()
            d["source_sheet"] = sheet
            d["excel_row"] = d.index + 2                      # header is row 1
            frames.append(d)
        frames.sort(key=lambda d: "listing_id" not in d.columns)   # sheets that already have ids go first
        kept, seen = [], set()
        next_id = 1
        for d in frames:
            for _, r in d.iterrows():
                photo = str(r.get("photo_url") or "").strip()
                content = _norm(f"{r.get('make')} {r.get('model')} {r.get('trim')} {r.get('year')} "
                                f"{r.get('title')} {str(r.get('description'))[:300]}")
                if (photo and photo in seen) or content in seen:     # same photo OR same text = same ad re-posted
                    continue
                seen.update({content, photo} - {""})
                row = r.to_dict()
                if "listing_id" in d.columns and pd.notna(r.get("listing_id")):
                    row["car_id"] = int(r["listing_id"])
                    next_id = max(next_id, row["car_id"] + 1)
                else:
                    row["car_id"] = next_id
                    next_id += 1
                kept.append(row)
        return pd.DataFrame(kept).drop(columns=["listing_id"], errors="ignore")

    def _load(self) -> pd.DataFrame:
        df = self._read_sheets()
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
        d["source"] = f"{r.source_sheet} sheet, Excel row {int(r.excel_row)}"
        d["description"] = r.description[:1800]
        d["field_notes"] = (
            "price/mileage/warranty/specs/colors/body_type were derived automatically from the listing text; "
            "null means NOT STATED in the listing. Never guess missing values."
        )
        return d

    def exists(self, car_id: int) -> bool:
        return car_id in self.df.index

    # ---- retrieval helpers ---------------------------------------------- #
    def _fuse(self, tfidf: Optional[np.ndarray], vec: Optional[dict[int, float]]) -> Optional[np.ndarray]:
        """Reciprocal-rank fusion of keyword (TF-IDF) and vector (embedding) similarity."""
        n = len(self.df)
        scores = np.zeros(n)
        used = 0
        for signal in (tfidf, None if vec is None else np.array([vec.get(cid, -1.0) for cid in self.df.index])):
            if signal is None:
                continue
            order = np.argsort(-signal, kind="stable")
            ranks = np.empty(n)
            ranks[order] = np.arange(n)
            scores += 1.0 / (60.0 + ranks)
            used += 1
        return scores if used else None

    def _with_text(self, cid: int, item: dict[str, Any]) -> dict[str, Any]:
        """Attach the seller's own words so the LLM answers from the source text, not from memory."""
        item["listing_text"] = self.df.loc[cid, "description"][:700]
        return item

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
        max_limit: int = 8,
    ) -> dict[str, Any]:
        """Hard filters + TF-IDF ranking.

        Make/model/body/year/specs/warranty are always known, so they are hard filters.
        Price, mileage and colour are often NOT STATED in a listing, so they are 'verifiable' filters:
          - results         = listings that CONFIRM every requested criterion
          - unverified      = listings that match everything else but don't state price/mileage/colour
        This way the agent never says 'nothing found' when the truth is 'the listing doesn't say'.
        """
        df = self.df
        base = pd.Series(True, index=df.index)

        if make:
            mk = _norm(make)
            base &= df["make"].apply(_norm).apply(lambda m: mk in m or m in mk)
        if model:
            toks = _norm(model).split()
            base &= df["_norm"].apply(lambda s: all(t in s.split() or t in s for t in toks))
        if body_type and body_type.lower() not in ("any", "unknown"):
            bt = _BODY_ALIASES.get(body_type.lower().strip(), body_type.lower().strip())
            base &= df["body_type"] == bt
        if min_year:
            base &= df["year"] >= int(min_year)
        if max_year:
            base &= df["year"] <= int(max_year)
        if specs:
            base &= df["specs"].fillna("").str.lower() == specs.lower()
        if warranty_only:
            base &= df["warranty"] == True  # noqa: E712

        strict, relaxed = base.copy(), base.copy()
        missing_if_unverified: dict[str, pd.Series] = {}       # field name -> "listing doesn't state it"
        excluded_unknown_price = excluded_unknown_km = 0

        if min_price_aed is not None or max_price_aed is not None:
            known = df["price_aed"].notna()
            ok = known.copy()
            if min_price_aed is not None:
                ok &= df["price_aed"].fillna(0) >= min_price_aed
            if max_price_aed is not None:
                ok &= df["price_aed"].fillna(10**12) <= max_price_aed
            excluded_unknown_price = int((base & ~known).sum())
            strict &= ok
            relaxed &= ok | ~known
            missing_if_unverified["price"] = ~known
        if max_mileage_km is not None:
            known = df["mileage_km"].notna()
            ok = known & (df["mileage_km"].fillna(10**12) <= max_mileage_km)
            excluded_unknown_km = int((base & ~known).sum())
            strict &= ok
            relaxed &= ok | ~known
            missing_if_unverified["mileage"] = ~known
        if color:
            c = color.lower()
            known = df["colors"].apply(bool)
            ok = df["colors"].apply(lambda cs: c in cs)
            strict &= ok
            relaxed &= ok | ~known
            missing_if_unverified["colour"] = ~known

        sims, retrieval = None, "filters"
        if query and query.strip() and (strict.any() or relaxed.any()):
            tfidf = linear_kernel(self.vectorizer.transform([expand_query(query)]), self.matrix).ravel()
            tfidf = tfidf if tfidf.any() else None
            vec = self.rag.similarities(query) if self.rag is not None else None
            sims = self._fuse(tfidf, vec)
            retrieval = ("hybrid" if (vec and tfidf is not None) else "vector" if vec
                         else "keyword" if tfidf is not None else "filters")

        def rank(ids: list[int], extra_key=None) -> list[int]:
            def key(cid: int):
                sim = sims[self._pos[cid]] if sims is not None else 0.0
                return ((extra_key(cid) if extra_key else 0), -sim, -int(df.loc[cid, "year"]))
            return sorted(ids, key=key)

        strict_ids = rank(list(df.index[strict]))
        unverified_ids = list(df.index[relaxed & ~strict])

        def not_stated(cid: int) -> list[str]:
            return [name for name, m in missing_if_unverified.items() if bool(m.loc[cid])]

        unverified_ids = rank(unverified_ids, extra_key=lambda cid: len(not_stated(cid)))

        limit = max(1, min(int(limit or 5), max_limit))
        unverified = []
        for cid in unverified_ids[:3]:
            item = self._with_text(cid, self.summary(cid))
            item["not_stated_in_listing"] = not_stated(cid)
            unverified.append(item)

        return {
            "total_matches": len(strict_ids),
            "results": [self._with_text(c, self.summary(c)) for c in strict_ids[:limit]],
            "retrieval": retrieval,
            "unverified_total": len(unverified_ids),
            "unverified_matches": unverified,
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