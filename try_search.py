from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory

inv = Inventory(INVENTORY_PATH)
print(inv.stats()["total_listings"], "listings loaded\n")

res = inv.search(body_type="suv", make="mercedes", limit=3)
print("Mercedes SUVs:", res["total_matches"])
for c in res["results"]:
    print(f"  #{c['car_id']} {c['name']} | price={c['price_aed']} | km={c['mileage_km']}")

res = inv.search(max_price_aed=100_000)
print("\nUnder AED 100k:", res["total_matches"], "| listings with no price:", res["excluded_because_price_not_listed"])
for c in res["results"]:
    print(f"  #{c['car_id']} {c['name']} | AED {c['price_aed']:,}")