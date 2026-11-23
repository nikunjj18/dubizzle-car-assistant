"""Build the RAG index by hand and prove it works:  uv run python scripts/ingest.py

  Excel -> cleaned table (SQLite: data/listings.db) -> embeddings -> vector DB (ChromaDB: data/chroma)
Then runs a few similarity searches so you can see what gets retrieved.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `assistant` imports work from /scripts

from assistant.config import INVENTORY_PATH  # noqa: E402
from assistant.inventory import Inventory  # noqa: E402
from assistant.rag import RagIndex  # noqa: E402

inv = Inventory(INVENTORY_PATH)
rag = RagIndex()
print("Building index ...")
info = rag.build(inv)
for k, v in info.items():
    print(f"  {k}: {v}")
if not rag.ready:
    sys.exit("\nVector index NOT ready - see 'error' above. (Keyword search still works.)")

inv.attach_rag(rag)
for q in ["luxury open-top car for weekends", "spacious family SUV with 7 seats", "fast sporty coupe", "rugged off-road 4x4"]:
    res = inv.search(query=q, limit=4)
    print(f"\n'{q}'  [{res['retrieval']} retrieval]")
    for c in res["results"]:
        print(f"   #{c['car_id']:<3} {c['name']}")