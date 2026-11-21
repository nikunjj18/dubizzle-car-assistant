import pytest

from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory


@pytest.fixture(scope="module")
def inv():
    return Inventory(INVENTORY_PATH)


def test_loads_all_100_listings(inv):
    assert len(inv.df) == 100


def test_make_and_body_filter(inv):
    r = inv.search(make="mercedes", body_type="suv")
    assert r["total_matches"] > 3
    assert all(c["make"] == "Mercedes-Benz" and c["body_type"] == "suv" for c in r["results"])


def test_price_filter_reports_listings_without_price(inv):
    r = inv.search(max_price_aed=100_000)
    assert r["excluded_because_price_not_listed"] > 50
    assert all(c["price_aed"] <= 100_000 for c in r["results"])


def test_unknown_make_returns_nothing(inv):
    assert inv.search(make="honda")["total_matches"] == 0


def test_text_relevance_ranking(inv):
    top = inv.search(query="open top roadster convertible luxury", limit=3)["results"]
    assert top[0]["body_type"] == "convertible"


def test_known_listing_details(inv):
    velar = inv.details(3)
    assert velar["price_aed"] == 119750 and velar["mileage_km"] == 68000
    cayenne = inv.details(8)
    assert cayenne["mileage_km"] is None      # not stated in the listing -> None, never guessed
    assert inv.details(9999) is None