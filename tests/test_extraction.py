from assistant.inventory import (clean_text, extract_colors, extract_mileage, extract_price,
                                 extract_specs, extract_warranty, infer_body_type)


def test_clean_text():
    assert clean_text("Hello<br>world &amp; co") == "Hello world & co"


def test_price_prefers_cash_and_ignores_monthly():
    text = "Option 1 - AED 1,950 monthly for 5 years. Option 3 - AED 95,000 cash."
    assert extract_price("Toyota", text) == (95000, 1950)


def test_mileage_is_not_a_price():
    assert extract_price("DONE ONLY 12,000km", "") == (None, None)
    assert extract_mileage("DONE ONLY 12,000km", "") == 12000


def test_mileage_short_numbers_and_new_cars():
    assert extract_mileage("", "Odometer : 970 Kms") == 970
    assert extract_mileage("2026 Mercedes GLC 0KM", "") == 0


def test_warranty_unknown_is_not_no_warranty():
    assert extract_warranty("", "Nice car, call me")[0] is None
    assert extract_warranty("", "Agency warranty till 2027")[0] is True
    assert extract_warranty("", "sold with no warranty")[0] is False


def test_specs_colors_body():
    assert extract_specs("Lexus LX 600 GCC specs") == "GCC"
    assert extract_colors("2020 White Range Rover", "") == ["white"]
    assert infer_body_type("porsche", "cayenne", "gts", "") == "suv"
    assert infer_body_type("rolls-royce", "dawn", "standard", "") == "convertible"



def test_mileage_with_space_thousands_separator():
    assert extract_mileage("", "Mileage: 23 900 kms Exterior: White") == 23900


def test_top_speed_is_not_mileage():
    assert extract_mileage("", "Max Speed 318km/h, 0 to 100 KM/H in 2.5 s") is None