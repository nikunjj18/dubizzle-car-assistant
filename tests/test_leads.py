from datetime import timedelta

import pytest

from assistant import leads


@pytest.fixture(autouse=True)
def csv_files(tmp_path, monkeypatch):
    monkeypatch.setattr(leads, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(leads, "BOOKINGS_CSV", tmp_path / "bookings.csv")


def next_weekday(weekday: int):
    """Next date (strictly in the future) that falls on weekday: Mon=0 ... Sun=6."""
    d = leads.today() + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


# ---------------- lead scoring ---------------- #
def test_score_cold_warm_hot():
    assert leads.score_lead({}, [], False)[0] == "cold"
    assert leads.score_lead({"budget_max_aed": 70000}, [], False)[0] == "cold"
    assert leads.score_lead({"budget_max_aed": 70000, "body_types": ["suv"]}, [], False)[0] == "warm"
    assert leads.score_lead({}, [], True)[0] == "hot"


def test_upsert_keeps_one_row_per_user_and_rescores():
    leads.upsert_lead("sara", "Sara", {"budget_max_aed": 70000}, [])
    row = leads.upsert_lead("sara", "Sara", {"budget_max_aed": 70000, "body_types": ["suv"]}, [8])
    rows = leads.list_leads()
    assert len(rows) == 1 and rows[0]["lead_score"] == "warm" and rows[0]["interested_car_ids"] == "8"
    assert row["lead_id"] == "L0001"


def test_booked_lead_stays_hot_after_later_updates():
    leads.upsert_lead("omar", "Omar", {}, [], booked=True)
    leads.upsert_lead("omar", "Omar", {"budget_max_aed": 50000}, [])
    assert leads.list_leads()[0]["lead_score"] == "hot"


# ---------------- slot rules ---------------- #
def test_sunday_is_closed():
    info = leads.available_slots(1, next_weekday(6))
    assert info["open"] is False and info["slots"] == []


def test_monday_to_saturday_slots_run_8am_to_8pm():
    for wd in range(6):
        slots = leads.available_slots(1, next_weekday(wd))["slots"]
        assert slots[0] == "08:00" and slots[-1] == "19:00" and len(slots) == 12


def test_past_dates_are_rejected():
    info = leads.available_slots(1, leads.today() - timedelta(days=1))
    assert info["open"] is False


# ---------------- booking ---------------- #
def test_booking_saves_row_and_blocks_double_booking():
    day = next_weekday(2)
    ok = leads.book_slot("sara", "Sara", "0501234567", 3, "2018 Land Rover Velar", day, "10:00")
    assert ok["success"] and ok["booking_id"] == "B0001"
    assert len(leads.list_bookings()) == 1
    assert "10:00" not in leads.available_slots(3, day)["slots"]       # slot is gone for this car
    assert "10:00" in leads.available_slots(4, day)["slots"]           # other cars unaffected
    again = leads.book_slot("omar", "Omar", "0509999999", 3, "x", day, "10:00")
    assert again["success"] is False


def test_booking_outside_hours_or_on_sunday_fails():
    assert leads.book_slot("a", "A", "1", 3, "x", next_weekday(1), "20:00")["success"] is False
    assert leads.book_slot("a", "A", "1", 3, "x", next_weekday(1), "07:00")["success"] is False
    assert leads.book_slot("a", "A", "1", 3, "x", next_weekday(6), "10:00")["success"] is False