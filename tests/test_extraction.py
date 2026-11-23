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


def test_price_awkward_formats():
    assert extract_price("", "1,349,999 AED / 25,683 AED per Month with 20% Down Payment") == (1349999, 25683)
    assert extract_price("", "Price # 32000 Aed")[0] == 32000
    assert extract_price("Mini Cooper 2018 GCC 120,000 km 30500", "")[0] == 30500
    assert extract_price("", "From AED 5,805/mo | Up to 3Y Warranty") == (None, 5805)


def test_service_interval_and_warranty_cap_are_not_mileage():
    assert extract_mileage("", "SERVICE CONTRACT (16000 Km or 1 year, whichever first)") is None
    assert extract_mileage("", "Warranty up to 100,000 km") is None


def test_body_type_priority():
    assert infer_body_type("mercedes-benz", "c-class", "c63 amg", "MERCEDES S63 AMG 2014") == "sedan"
    assert infer_body_type("mercedes-benz", "e-class", "e 400", "MERCEDES E 450 COUPE 2019") == "coupe"
    assert infer_body_type("ford", "mustang", "v6", "2014 FORD MUSTANG SOFT TOP CONVERTIBLE") == "convertible"


def test_warranty_limits_service_odometer_and_years_are_not_mileage():
    assert extract_mileage("", "Ford Warranty until 6/2028 or 100,000 kms") is None
    assert extract_mileage("", "Warranty: Until 24-June-2029 or 105,000 KM") is None
    assert extract_mileage("", "Last Service: 34,000 Km (05/02/2026)") is None
    assert extract_mileage("", "Rear Camera - Low Mileage 2024 Toyota Tundra") is None
    assert extract_mileage("", "Brand New - Only 18 km mileage") == 18


def test_dot_as_thousands_separator():
    assert extract_mileage("", "83.000KM 25.000AED Monthly Installment 550 aed") == 83000


def test_monthly_payment_formats():
    assert extract_price("Nissan Sunny", "83.000KM 25.000AED Monthly Installment 550 aed") == (None, 550)
    assert extract_price("", "Mustang Convertable 2021 0% Downpayment EMI 1375")[1] == 1375
    assert extract_price("", "AED 2,340 P.M | 2024 BMW i4")[1] == 2340
    assert extract_price("1,430AED / Month | TERRITORY", "")[1] == 1430


def test_more_body_types():
    assert infer_body_type("mercedes-benz", "gle-class", "gle 400 4matic", "AED 2887/month | GLE 400") == "suv"
    assert infer_body_type("mercedes-benz", "cla-class", "cla 250 4matic", "MERCEDES CLA250 2021") == "sedan"
    assert infer_body_type("bmw", "i4", "edrive35", "2024 BMW i4 Gran Coupé eDrive35") == "sedan"
    assert infer_body_type("mercedes-benz", "a-class", "a 45 amg", "2014 Mercedes-Benz A-Class A 45 AMG") == "hatchback"