"""dubizzle Cars Assistant - Streamlit client.

Talks to the FastAPI backend over HTTP only (no shared code or state with the backend).
Run:  uv run streamlit run app.py        (backend: uv run uvicorn main:app --reload)

Brand logo: drop your official logo at  assets/logo.png  (or .svg / .jpg / .webp)  and it replaces the text wordmark.
"""
from __future__ import annotations

import base64
import html
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_URL", "http://127.0.0.1:8000")
BRAND = "#E3262E"
ASSETS = Path(__file__).resolve().parent / "assets"
DUBAI = ZoneInfo("Asia/Dubai")
PAGES = ["💬 Assistant", "🔎 Explore cars", "❤️ My shortlist & visits", "📋 Sales view"]

st.set_page_config(page_title="dubizzle Cars Assistant", page_icon="🚗", layout="wide")

st.markdown(f"""
<style>
  #MainMenu, footer {{visibility: hidden;}}
  .block-container {{padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1180px;}}
  /* ---- brand ---- */
  .wordmark {{font-weight: 900; color: {BRAND}; letter-spacing: -1.5px; line-height: 1;}}
  .topbar {{display: flex; align-items: center; justify-content: space-between; padding: 6px 0 14px 0;
            border-bottom: 3px solid {BRAND}; margin-bottom: 18px;}}
  .topbar .tag {{font-size: .72rem; font-weight: 700; color: #fff; background: {BRAND}; padding: 2px 8px;
                 border-radius: 6px; margin-left: 10px; letter-spacing: .5px; vertical-align: middle;}}
  .userpill {{display: inline-flex; align-items: center; gap: 8px; background: #F5F6F8; border-radius: 999px;
              padding: 5px 14px 5px 6px; font-size: .85rem; font-weight: 600; color: #1F2430;}}
  .avatar {{width: 28px; height: 28px; border-radius: 50%; background: {BRAND}; color: #fff; display: inline-flex;
            align-items: center; justify-content: center; font-weight: 800; font-size: .85rem;}}
  /* ---- hero / login ---- */
  .hero h1 {{font-size: 2.6rem; line-height: 1.1; margin: 18px 0 10px 0; color: #1F2430; letter-spacing: -1px;}}
  .hero h1 span {{color: {BRAND};}}
  .hero p.lead {{font-size: 1.08rem; color: #5B6270; max-width: 520px;}}
  .feat {{display: flex; gap: 12px; align-items: flex-start; margin: 14px 0;}}
  .feat .ic {{font-size: 1.4rem; background: {BRAND}14; border-radius: 12px; padding: 8px 10px;}}
  .feat b {{display: block; color: #1F2430;}} .feat small {{color: #5B6270;}}
  /* ---- cards ---- */
  .car-card {{border: 1px solid #E6E8EC; border-radius: 16px; overflow: hidden; background: #fff;
              box-shadow: 0 1px 3px rgba(16,24,40,.06); margin-bottom: 6px;}}
  .car-card img {{width: 100%; height: 160px; object-fit: cover; display: block; background: #EEF0F3;}}
  .car-body {{padding: 12px 14px 12px 14px;}}
  .car-title {{font-weight: 700; font-size: .98rem; line-height: 1.25; color: #1F2430; min-height: 2.4em;}}
  .car-price {{color: {BRAND}; font-weight: 800; font-size: 1.15rem; margin: 4px 0 6px 0;}}
  .car-price small {{color: #8A91A0; font-weight: 600; font-size: .72rem;}}
  .idtag {{font-size: .7rem; font-weight: 700; padding: 1px 7px; border-radius: 6px; background: #EEF0F3;
           color: #5B6270; margin-right: 6px;}}
  .badge {{display: inline-block; font-size: .72rem; padding: 2px 9px; margin: 2px 4px 2px 0; border-radius: 999px;
           background: #F1F3F6; color: #3B4252; font-weight: 600;}}
  .badge.good {{background: #E7F6EC; color: #17803D;}}
  .badge.warn {{background: #FFF4DB; color: #9A6700;}}
  .chip {{display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px; font-size: .75rem;
          background: {BRAND}14; color: {BRAND}; border: 1px solid {BRAND}40; font-weight: 600;}}
  .sidecard {{background: #F5F6F8; border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;}}
  .muted {{color: #6B7280; font-size: .8rem;}}
  div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{width: 100%; border-radius: 10px; font-weight: 600;}}
</style>""", unsafe_allow_html=True)


# ------------------------------- helpers ------------------------------- #
def api(method: str, path: str, **kw: Any) -> Optional[Any]:
    """Call the backend. On failure return None and keep a readable message in session state."""
    try:
        r = httpx.request(method, f"{API}{path}", timeout=200, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            st.session_state["last_error"] = f"{r.status_code}: {detail}"
            return None
        return r.json()
    except httpx.TimeoutException:
        st.session_state["last_error"] = ("The AI took too long to answer. Check the backend terminal for '[llm]' lines, "
                                          "then try again.")
        return None
    except httpx.HTTPError as e:
        st.session_state["last_error"] = (f"Cannot reach the backend at {API} ({type(e).__name__}). "
                                          "Is `uv run uvicorn main:app --reload` running?")
        return None


def logo_html(height: str = "2rem") -> str:
    """Official logo if you put one in /assets, otherwise a text wordmark in the brand colour."""
    for ext, mime in (("svg", "image/svg+xml"), ("png", "image/png"), ("jpg", "image/jpeg"), ("webp", "image/webp")):
        p = ASSETS / f"logo.{ext}"
        if p.exists():
            return f'<img src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}" style="height:{height}" alt="dubizzle">'
    return f'<span class="wordmark" style="font-size:{height}">dubizzle</span>'


def init_state() -> None:
    for key, default in {"user_id": None, "session_id": None, "messages": [], "pending": None, "last_error": None,
                         "nav_page": PAGES[0], "mem": {}, "explore_limit": 12}.items():
        st.session_state.setdefault(key, default)
    # pages are switched by setting `_goto`; it is applied here, BEFORE the navigation radio is created
    goto = st.session_state.pop("_goto", None)
    if goto:
        st.session_state["nav_page"] = goto


def start_session(user_id: str, name: Optional[str] = None) -> bool:
    data = api("POST", "/sessions", json={"user_id": user_id, "name": name or None})
    if not data:
        return False
    st.session_state.update(
        user_id=data["user_id"], session_id=data["session_id"], _goto=PAGES[0],
        messages=[{"role": "assistant", "content": data["greeting"], "cars": [], "trace": []}])
    return True


def sign_out() -> None:
    st.session_state.update(user_id=None, session_id=None, messages=[], mem={}, _goto=PAGES[0])


def ask(prompt: str) -> None:
    """Send a message to the assistant on the next run (works from any page / dialog)."""
    st.session_state.pending = prompt
    st.session_state["_goto"] = PAGES[0]
    st.rerun()


def fmt_price(c: dict[str, Any]) -> str:
    if c.get("price_aed"):
        return f"AED {int(c['price_aed']):,}"
    if c.get("monthly_payment_aed"):
        return f"AED {int(c['monthly_payment_aed']):,}<small> /month · price not listed</small>"
    return "<small>Price not listed</small>"


def _dialog(title: str):
    """st.dialog when available (Streamlit >= 1.37), otherwise render inline."""
    deco = getattr(st, "dialog", None)
    return deco(title) if deco else (lambda fn: fn)


# ------------------------------- car card ------------------------------- #
def car_html(c: dict[str, Any]) -> str:
    badges = []
    if c.get("body_type") and c["body_type"] != "unknown":
        badges.append(f'<span class="badge">{html.escape(str(c["body_type"]).title())}</span>')
    if c.get("mileage_km") is not None:
        txt = "Brand new · 0 km" if c["mileage_km"] == 0 else f"{int(c['mileage_km']):,} km"
        badges.append(f'<span class="badge">{txt}</span>')
    if c.get("specs"):
        badges.append(f'<span class="badge">{html.escape(str(c["specs"]))} specs</span>')
    if c.get("warranty"):
        badges.append('<span class="badge good">✓ Warranty mentioned</span>')
    for field in c.get("not_stated_in_listing", []) or []:
        badges.append(f'<span class="badge warn">{html.escape(field.title())} not stated</span>')
    img = html.escape(c.get("photo_url") or "")
    return (f'<div class="car-card"><img src="{img}" loading="lazy" onerror="this.style.display=\'none\'">'
            f'<div class="car-body"><div class="car-title"><span class="idtag">#{c["car_id"]}</span>{html.escape(c["name"])}</div>'
            f'<div class="car-price">{fmt_price(c)}</div>{"".join(badges)}</div></div>')


def car_card(c: dict[str, Any], key: str) -> None:
    st.markdown(car_html(c), unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    if b1.button("Details", key=f"det_{key}"):
        details_dialog(c["car_id"])
    if b2.button("❤️ Save", key=f"fav_{key}"):
        api("POST", f"/users/{st.session_state.user_id}/favorites", json={"car_id": c["car_id"]})
        st.toast(f"Saved #{c['car_id']} to your shortlist ❤️")
    if b3.button("📅 Book", key=f"book_{key}"):
        booking_dialog(c)


def render_cars(cars: list[dict[str, Any]], key: str, per_row: int = 3) -> None:
    for start in range(0, len(cars), per_row):
        for col, car in zip(st.columns(per_row), cars[start:start + per_row]):
            with col:
                car_card(car, f"{key}_{car['car_id']}")


# ------------------------------- dialogs ------------------------------- #
@_dialog("Listing details")
def details_dialog(car_id: int) -> None:
    d = api("GET", f"/inventory/{car_id}")
    if not d:
        st.error(st.session_state.get("last_error") or "Could not load this listing.")
        return
    st.markdown(car_html(d), unsafe_allow_html=True)
    facts = [("Year", d.get("year")), ("Make / model", f"{d.get('make')} {d.get('model')}"),
             ("Mileage", f"{int(d['mileage_km']):,} km" if d.get("mileage_km") is not None else "not stated"),
             ("Price", f"AED {int(d['price_aed']):,}" if d.get("price_aed") else "not stated"),
             ("Warranty", d.get("warranty_note") or ("mentioned" if d.get("warranty") else "not stated")),
             ("Specs", d.get("specs") or "not stated")]
    st.markdown("".join(f"**{k}:** {v}  \n" for k, v in facts))
    st.markdown("**Seller's description**")
    st.write(d.get("description", ""))
    st.caption(f"Source in your Excel file: {d.get('source', 'n/a')}")
    c1, c2 = st.columns(2)
    if c1.button("💬 Ask Dubi about this car", key=f"ask_{car_id}"):
        ask(f"Tell me more about car #{car_id} ({d['name']}).")
    if c2.button("❤️ Save to shortlist", key=f"dsave_{car_id}"):
        api("POST", f"/users/{st.session_state.user_id}/favorites", json={"car_id": car_id})
        st.toast("Saved ❤️")


@_dialog("Book a test drive")
def booking_dialog(c: dict[str, Any]) -> None:
    st.markdown(f"**#{c['car_id']} · {c['name']}**")
    st.caption("Viewings & test drives are pre-booked · Monday to Saturday · 8am to 8pm (Dubai time)")
    today = datetime.now(DUBAI).date()
    default = today + timedelta(days=1)
    while default.weekday() == 6:
        default += timedelta(days=1)
    kind = st.radio("What would you like?", ["Test drive", "Viewing only"], horizontal=True, key=f"bk_kind_{c['car_id']}")
    d = st.date_input("Date", value=default, min_value=today, key=f"bk_date_{c['car_id']}")
    slot = st.selectbox("Time", [f"{h:02d}:00" for h in range(8, 20)], index=6, key=f"bk_time_{c['car_id']}")
    prof = (st.session_state.get("mem") or {}).get("profile", {})
    name = st.text_input("Your name", value=(st.session_state.get("mem") or {}).get("name") or "", key=f"bk_name_{c['car_id']}")
    phone = st.text_input("Phone number", value=prof.get("phone", ""), placeholder="05x xxx xxxx", key=f"bk_phone_{c['car_id']}")
    if d.weekday() == 6:
        st.warning("We're closed on Sundays. Please pick Monday to Saturday.")
    if st.button("Request booking", type="primary", key=f"bk_go_{c['car_id']}"):
        if d.weekday() == 6:
            st.error("Sunday is not available.")
        elif not name.strip() or not phone.strip():
            st.error("Please enter your name and phone number.")
        else:
            ask(f"Please book a {kind.lower()} for car #{c['car_id']} ({c['name']}) on {d.isoformat()} at {slot}. "
                f"My name is {name.strip()} and my phone number is {phone.strip()}.")


# ------------------------------- sidebar ------------------------------- #
def sidebar() -> str:
    with st.sidebar:
        st.markdown(f'<div style="padding:4px 0 8px 0">{logo_html("1.9rem")}</div>', unsafe_allow_html=True)
        st.caption("Cars assistant · powered by AI")
        if not st.session_state.user_id:
            return PAGES[0]

        mem = api("GET", f"/users/{st.session_state.user_id}/memory") or {}
        st.session_state.mem = mem
        name = mem.get("name") or st.session_state.user_id
        st.markdown(f'<div class="sidecard"><span class="userpill"><span class="avatar">{html.escape(name[:1].upper())}</span>'
                    f'{html.escape(name)}</span><div class="muted" style="margin-top:6px">ID: {html.escape(st.session_state.user_id)}'
                    f' · session {st.session_state.session_id}</div></div>', unsafe_allow_html=True)

        choice = st.radio("Navigate", PAGES, key="nav_page", label_visibility="collapsed")

        st.divider()
        st.markdown("**🧠 What I remember**")
        prof = mem.get("profile", {}) if mem.get("known") else {}
        chips = [x for k in ("body_types", "preferred_makes", "preferred_colors", "must_haves") for x in prof.get(k, [])]
        if prof.get("purpose"):
            chips.append(prof["purpose"])
        lo, hi = prof.get("budget_min_aed"), prof.get("budget_max_aed")
        if lo or hi:
            st.markdown("**Budget:** " + (f"AED {int(float(lo)):,} – " if lo else "up to ") + (f"{int(float(hi)):,}" if hi else "any"))
        if chips:
            st.markdown("".join(f'<span class="chip">{html.escape(str(c))}</span>' for c in chips), unsafe_allow_html=True)
        for f in (mem.get("favorites") or [])[:3]:
            st.caption(f"❤️ #{f['car_id']} {f['car_name']}")
        for s in (mem.get("recent_searches") or [])[:2]:
            st.caption(f"🔍 {s['criteria']}")
        if not (chips or lo or hi or mem.get("favorites") or mem.get("recent_searches")):
            st.caption("Nothing yet. Tell Dubi what you're after!")

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("🔄 New chat", help="Same user, fresh conversation. Dubi still remembers you."):
            start_session(st.session_state.user_id)
            st.rerun()
        if c2.button("🚪 Sign out"):
            sign_out()
            st.rerun()
        if st.button("🗑️ Forget me", help="Delete everything stored about this user"):
            api("DELETE", f"/users/{st.session_state.user_id}/memory")
            sign_out()
            st.rerun()
    return choice


def topbar(title: str) -> None:
    name = (st.session_state.get("mem") or {}).get("name") or st.session_state.user_id
    pill = (f'<span class="userpill"><span class="avatar">{html.escape(str(name)[:1].upper())}</span>{html.escape(str(name))}</span>'
            if st.session_state.user_id else "")
    st.markdown(f'<div class="topbar"><div>{logo_html("2.1rem")}<span class="tag">CARS</span>'
                f'<span class="muted" style="margin-left:12px">{html.escape(title)}</span></div>{pill}</div>',
                unsafe_allow_html=True)


# ------------------------------- login ------------------------------- #
def login_page() -> None:
    topbar("Welcome")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(f"""<div class="hero"><h1>Find your next car.<br><span>Just ask Dubi.</span></h1>
        <p class="lead">Search real dubizzle listings in plain English, get honest answers about each car, and book a test drive in a minute.</p>
        <div class="feat"><div class="ic">🔎</div><div><b>Smart search</b><small>Filters + AI similarity over every listing. Try “a luxury open-top car for weekends”.</small></div></div>
        <div class="feat"><div class="ic">🧠</div><div><b>It remembers you</b><small>Come back later with the same ID and Dubi recalls your budget and the cars you liked.</small></div></div>
        <div class="feat"><div class="ic">📅</div><div><b>Test drives, sorted</b><small>Real slots Monday to Saturday, 8am to 8pm. No phone tag.</small></div></div>
        <div class="feat"><div class="ic">✅</div><div><b>Straight answers</b><small>If a listing doesn't say something, Dubi tells you instead of guessing.</small></div></div></div>""",
                    unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("### 👋 Sign in")
            st.caption("New here? Pick any ID. Use the same one next time and Dubi remembers you.")
            with st.form("login_form"):
                uid = st.text_input("User ID", placeholder="e.g. sara")
                nm = st.text_input("Your name (optional)", placeholder="e.g. Sara")
                go = st.form_submit_button("Start chatting →", type="primary")
            if go:
                if not uid.strip():
                    st.warning("Please enter a User ID.")
                elif start_session(uid, nm):
                    st.rerun()
            if st.session_state.last_error:
                st.error(st.session_state.last_error)
                st.session_state.last_error = None
        st.caption("🔒 Prototype · no password needed · data stays on this machine")


# ------------------------------- pages ------------------------------- #
SUGGESTIONS = [("🚙 White SUV under $20k", "I'm looking for a white SUV under $20,000"),
               ("🏎️ Luxury convertibles", "Show me luxury convertibles"),
               ("🛡️ Cars with warranty", "Which cars mention a warranty?"),
               ("📅 Book a test drive", "I'd like to book a test drive")]


def chat_page() -> None:
    topbar("Chat with Dubi")
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"], avatar="🚗" if m["role"] == "assistant" else None):
            st.markdown(m["content"])
            if m.get("cars"):
                render_cars(m["cars"], f"m{i}")
            if m.get("trace"):
                with st.expander(f"🛠️ How Dubi answered ({len(m['trace'])} tool call{'s' if len(m['trace']) != 1 else ''})"):
                    for t in m["trace"]:
                        st.markdown(f"{'✅' if t['ok'] else '⚠️'} `{t['tool']}`")
                        st.code(t["arguments"], language="json")
            if m.get("guardrail"):
                st.caption("🛡️ Handled by a code-level guardrail (the AI model was not called)")

    if len(st.session_state.messages) <= 1:
        st.markdown("**Try asking:**")
        for col, (label, prompt) in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            if col.button(label, key=f"sg_{label}"):
                ask(prompt)

    prompt = st.chat_input("Ask about cars, e.g. “white SUV under $20k” …") or st.session_state.pending
    st.session_state.pending = None
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🚗"):
        with st.spinner("Dubi is checking the inventory…"):
            res = api("POST", "/chat", json={"user_id": st.session_state.user_id,
                                             "session_id": st.session_state.session_id, "message": prompt})
    if res:
        st.session_state.messages.append({"role": "assistant", "content": res["reply"], "cars": res["cars"],
                                          "trace": res["tool_calls"], "guardrail": res.get("guardrail")})
    else:
        st.session_state.messages.append({"role": "assistant", "cars": [], "trace": [],
                                          "content": "⚠️ " + (st.session_state.last_error or "Something went wrong.")})
        st.session_state.last_error = None
    st.rerun()


def explore_page() -> None:
    topbar("Explore cars")
    stats = api("GET", "/inventory/stats") or {}
    if stats:
        a, b, c = st.columns(3)
        a.metric("Listings", stats.get("total_listings", 0))
        b.metric("With a stated price", stats.get("with_price", 0))
        c.metric("With stated mileage", stats.get("with_mileage", 0))
    with st.container(border=True):
        f1, f2, f3 = st.columns([1.4, 1, 1])
        q = f1.text_input("Search", placeholder="e.g. luxury open-top, 7 seats, low mileage…")
        make = f2.selectbox("Make", ["Any"] + stats.get("makes", []))
        body = f3.selectbox("Body type", ["Any", "suv", "sedan", "coupe", "convertible", "hatchback", "wagon", "pickup", "van"])
        g1, g2 = st.columns(2)
        year_from = g1.slider("Year from", 2000, 2026, 2000)
        price = g2.number_input("Max price (AED, 0 = any)", min_value=0, step=10000, value=0)
    params = {"limit": st.session_state.explore_limit}
    if q.strip():
        params["q"] = q.strip()
    if make != "Any":
        params["make"] = make
    if body != "Any":
        params["body_type"] = body
    if year_from > 2000:
        params["min_year"] = year_from
    if price:
        params["max_price_aed"] = price
    res = api("GET", "/inventory", params=params)
    if not res:
        st.error(st.session_state.get("last_error") or "Could not load the inventory.")
        return
    note = f"**{res['total_matches']}** listings match"
    if res.get("retrieval") and q.strip():
        note += f" · ranked by {res['retrieval']} retrieval"
    st.markdown(note)
    if price and res.get("excluded_because_price_not_listed"):
        st.info(f"{res['excluded_because_price_not_listed']} more cars match your other filters but don't state a price in the listing.")
    render_cars(res["results"], "ex")
    if res["total_matches"] > len(res["results"]):
        if st.button("Show more cars"):
            st.session_state.explore_limit += 12
            st.rerun()
    if res.get("unverified_matches"):
        st.markdown("#### Possible matches (the listing doesn't state everything you asked for)")
        render_cars(res["unverified_matches"], "exu")


def shortlist_page() -> None:
    topbar("My shortlist & visits")
    uid = st.session_state.user_id
    mem = st.session_state.get("mem") or {}
    st.subheader("❤️ Shortlist")
    favs = mem.get("favorites") or []
    if not favs:
        st.info("Nothing saved yet. Tap ❤️ Save on any car and it shows up here.")
    cars = [d for d in (api("GET", f"/inventory/{f['car_id']}") for f in favs) if d]
    render_cars(cars, "sl")
    for f in favs:
        if st.button(f"Remove #{f['car_id']} {f['car_name']}", key=f"rm_{f['car_id']}"):
            api("DELETE", f"/users/{uid}/favorites/{f['car_id']}")
            st.rerun()
    st.subheader("📅 My visits")
    mine = [b for b in (api("GET", "/bookings") or []) if b.get("user_id") == uid]
    if mine:
        df = pd.DataFrame(mine)[["booking_id", "car_name", "slot_start", "type", "status"]]
        df.columns = ["Booking", "Car", "When", "Type", "Status"]
        st.dataframe(df, hide_index=True)
    else:
        st.info("No visits booked yet. Ask Dubi to book a test drive.")
    st.subheader("🔍 Past searches")
    searches = mem.get("recent_searches") or []
    if searches:
        for s in searches:
            st.write(f"• {s['criteria']}  ·  {s['matches']} matches  ·  {s['ts'][:10]}")
    else:
        st.caption("No searches yet.")


def sales_page() -> None:
    topbar("Sales view")
    st.caption("Live view of `data/leads.csv` and `data/bookings.csv`, written automatically by the assistant.")
    leads = api("GET", "/leads") or []
    bookings = api("GET", "/bookings") or []
    a, b, c, d = st.columns(4)
    a.metric("Leads", len(leads))
    b.metric("🔥 Hot", sum(1 for x in leads if x["lead_score"] == "hot"))
    c.metric("🌤️ Warm", sum(1 for x in leads if x["lead_score"] == "warm"))
    d.metric("Bookings", len(bookings))
    st.subheader("Qualified leads")
    if leads:
        st.dataframe(pd.DataFrame(leads), hide_index=True)
    else:
        st.info("No leads yet. Chat with Dubi to create one.")
    st.subheader("Viewing bookings")
    if bookings:
        st.dataframe(pd.DataFrame(bookings), hide_index=True)
    else:
        st.info("No bookings yet.")


# ------------------------------- main ------------------------------- #
init_state()
page = sidebar()
if not st.session_state.user_id:
    login_page()
elif page == PAGES[0]:
    chat_page()
elif page == PAGES[1]:
    explore_page()
elif page == PAGES[2]:
    shortlist_page()
else:
    sales_page()