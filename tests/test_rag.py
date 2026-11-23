"""RAG layer tests: SQLite table + vector DB + hybrid search. A fake embedder replaces Gemini (no key, no network)."""
import sqlite3
import zlib

import numpy as np
import pytest

from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory
from assistant.rag import Embedder, MemoryVectorStore, RagIndex, listing_document


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words 'embeddings' (hashing trick): similar words -> similar vectors."""
    DIM = 512

    def __init__(self):
        super().__init__(models=("fake/model",))
        self.texts_embedded = 0

    def probe(self):
        self.model = "fake/model"
        return self.model

    def embed(self, texts):
        self.texts_embedded += len(texts)
        out = []
        for t in texts:
            v = np.zeros(self.DIM)
            for w in t.lower().replace(".", " ").replace(",", " ").split():
                v[zlib.crc32(w.encode()) % self.DIM] += 1.0
            out.append((v / (np.linalg.norm(v) or 1.0)).tolist())
        return out


class BrokenEmbedder(Embedder):
    def __init__(self):
        super().__init__(models=("broken/model",))

    def probe(self):
        raise RuntimeError("No embedding model worked. broken/model: NotFoundError")


@pytest.fixture()
def inv():
    return Inventory(INVENTORY_PATH)


@pytest.fixture()
def rag(tmp_path):
    return RagIndex(embedder=FakeEmbedder(), db_path=tmp_path / "listings.db", chroma_dir=tmp_path / "chroma")


def test_build_writes_sql_table_and_vector_index(inv, rag):
    info = rag.build(inv)
    assert rag.ready and info["table_rows"] == 189 and info["vectors"] == 189 and info["newly_embedded"] == 189
    with sqlite3.connect(rag.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 189
        row = conn.execute("SELECT make, model, price_aed, mileage_km FROM listings WHERE car_id = 3").fetchone()
        assert row == ("land rover", "range rover velar", 119750, 68000)
        assert conn.execute("SELECT mileage_km FROM listings WHERE car_id = 8").fetchone()[0] is None   # NULL, not guessed


def test_vectors_are_cached_by_content_hash(inv, rag):
    rag.build(inv)
    first = rag.embedder.texts_embedded
    again = rag.build(inv)                                   # nothing changed -> no new embeddings
    assert again["newly_embedded"] == 0 and rag.embedder.texts_embedded == first
    inv.df.loc[8, "description"] = inv.df.loc[8, "description"] + " Brand new tyres fitted."
    changed = rag.build(inv)                                 # one listing edited -> exactly one re-embedded
    assert changed["newly_embedded"] == 1


def test_similarity_search_finds_the_right_listing(inv, rag):
    rag.build(inv)
    sims = rag.similarities("Porsche Cayenne GTS")
    assert len(sims) == 189 and max(sims, key=sims.get) == 8


def test_hybrid_search_uses_vectors_and_keeps_filters(inv, rag):
    rag.build(inv)
    inv.attach_rag(rag)
    res = inv.search(query="porsche cayenne", limit=3)
    assert res["retrieval"] == "hybrid" and res["results"][0]["car_id"] == 8
    filtered = inv.search(query="luxury car", body_type="convertible", limit=5)
    assert filtered["results"] and all(c["body_type"] == "convertible" for c in filtered["results"])   # filters still hard


def test_results_carry_the_listing_text_for_the_llm(inv):
    top = inv.search(make="mercedes", model="gls 63", limit=1)["results"][0]
    assert top["car_id"] == 92 and "Mileage: 23 900 kms" in top["listing_text"]      # the seller's own words


def test_falls_back_to_keyword_search_when_embeddings_fail(inv, tmp_path):
    broken = RagIndex(embedder=BrokenEmbedder(), db_path=tmp_path / "l.db", chroma_dir=tmp_path / "c")
    info = broken.build(inv)
    assert broken.ready is False and "error" in info and info["table_rows"] == 189     # table is still written
    inv.attach_rag(broken)
    res = inv.search(query="open top roadster convertible", limit=3)
    assert res["retrieval"] == "keyword" and res["results"][0]["body_type"] == "convertible"


def test_memory_store_fallback_behaves_like_chroma(inv, tmp_path):
    rag = RagIndex(embedder=FakeEmbedder(), db_path=tmp_path / "l.db", chroma_dir=tmp_path / "c", use_chroma=False)
    rag.build(inv)
    assert isinstance(rag.store, MemoryVectorStore)
    sims = rag.similarities("Porsche Cayenne GTS")
    assert max(sims, key=sims.get) == 8


def test_listing_document_contains_the_facts_we_embed(inv):
    doc = listing_document(inv.df.loc[92])
    assert "GLS 63" in doc and "23 900" in doc


def test_api_builds_index_on_startup_and_reports_status(tmp_path, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, "rag", RagIndex(embedder=FakeEmbedder(), db_path=tmp_path / "l.db", chroma_dir=tmp_path / "c"))
    monkeypatch.setattr(main.inventory, "rag", None)
    with TestClient(main.app) as client:                      # `with` runs the startup (lifespan) code
        for _ in range(100):                                  # index builds in a background thread
            if client.get("/rag/status").json()["ready"]:
                break
            time.sleep(0.1)
        status = client.get("/rag/status").json()
        assert status["ready"] and status["vectors"] == 189 and status["table_rows"] == 189
        assert client.get("/health").json()["vector_search"] is True
        res = client.get("/inventory", params={"q": "porsche cayenne", "limit": 2}).json()
        assert res["retrieval"] == "hybrid" and res["results"][0]["car_id"] == 8
    main.inventory.rag = None