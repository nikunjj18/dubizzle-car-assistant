import pytest

from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory


@pytest.fixture(scope="module")
def inv():
    return Inventory(INVENTORY_PATH)


def test_loads_both_sheets_with_stable_ids(inv):
    # the workbook holds two different sets of listings: 100 "cleaned" + 99 "raw" (10 re-posted ads are collapsed)
    assert len(inv.df) == 189 and inv.df.index.is_unique
    assert inv.df["source_sheet"].value_counts().to_dict() == {"cleaned dataset": 100, "raw dataset": 89}
    assert inv.details(7)["name"].endswith("G-Class Brabus") and inv.details(90)["name"].endswith("Pajero GLX")


def test_make_and_body_filter(inv):
    r = inv.search(make="mercedes", body_type="suv")
    assert r["total_matches"] > 3
    assert all(c["make"] == "Mercedes-Benz" and c["body_type"] == "suv" for c in r["results"])


def test_price_filter_reports_listings_without_price(inv):
    r = inv.search(max_price_aed=100_000)
    assert r["excluded_because_price_not_listed"] > 50
    assert all(c["price_aed"] <= 100_000 for c in r["results"])


def test_unknown_make_returns_nothing(inv):
    assert inv.search(make="saab")["total_matches"] == 0


def test_text_relevance_ranking(inv):
    top = inv.search(query="open top roadster convertible luxury", limit=3)["results"]
    assert top[0]["body_type"] == "convertible"


def test_known_listing_details(inv):
    velar = inv.details(3)
    assert velar["price_aed"] == 119750 and velar["mileage_km"] == 68000
    cayenne = inv.details(8)
    assert cayenne["mileage_km"] is None      # not stated in the listing -> None, never guessed
    assert inv.details(9999) is None


# ---------------- data-quality + soft-filter behaviour ---------------- #
def test_unverified_matches_when_listing_does_not_state_price_or_colour(inv):
    r = inv.search(color="white", body_type="suv", max_price_aed=73_450)     # "white SUV under $20k"
    assert r["unverified_total"] > 0
    unverified = r["unverified_matches"]
    assert all(c["not_stated_in_listing"] for c in unverified)
    assert len(unverified[0]["not_stated_in_listing"]) <= len(unverified[-1]["not_stated_in_listing"])   # fewest unknowns first


def test_body_type_is_not_fooled_by_amg_badge(inv):
    body = lambda i: inv.details(i)["body_type"]
    assert body(52) == "sedan" and body(4) == "coupe" and body(20) == "convertible" and body(34) == "wagon"


def test_prices_found_in_awkward_formats(inv):
    assert inv.details(15)["price_aed"] == 1349999 and inv.details(15)["monthly_payment_aed"] == 25683
    assert inv.details(60)["price_aed"] == 32000          # "Price # 32000 Aed"
    assert inv.details(6)["price_aed"] == 30500           # bare number ending the title


def test_service_interval_is_not_mileage(inv):
    assert inv.details(22)["mileage_km"] is None


def test_luxury_query_ranks_luxury_brands_first(inv):
    top = inv.search(query="luxury sedan", body_type="sedan", limit=4)["results"]
    assert any(c["make"] == "Rolls-Royce" for c in top)


# ---------------- second sheet ("raw dataset") ---------------- #
def test_honda_from_the_raw_sheet_is_searchable(inv):
    hondas = inv.search(make="honda")["results"]
    assert len(hondas) == 1 and hondas[0]["name"].endswith("Cr-V EX-L")
    assert inv.details(hondas[0]["car_id"])["source"] == "raw dataset sheet, Excel row 88"      # checkable in Excel


def test_reposted_ads_are_collapsed(inv):
    assert inv.search(make="audi", model="q8")["total_matches"] == 1       # the same dealer ad was posted 5 times


def test_warranty_limits_are_not_read_as_mileage(inv):
    territory = inv.search(make="ford", model="territory")["results"]
    assert territory and all(c["mileage_km"] != 100_000 for c in territory)
    h9 = inv.search(make="haval")["results"][0]
    assert h9["mileage_km"] == 18                                           # "Brand New - Only 18 km mileage"