"""RAG layer: listings table (SQLite) + embeddings (Gemini via LiteLLM) + vector database (ChromaDB).

Pipeline
  ingest : Excel -> clean/enrich -> SQLite table `listings` -> one embedding per listing -> ChromaDB
  query  : user text -> embedding -> cosine similarity against every listing vector -> scores
  answer : Inventory blends these scores with structured filters + keyword score, the agent hands the retrieved
           listings (with their own text) to the LLM, which writes the reply from them.

Vectors are cached by content hash, so a restart only embeds listings that are new or changed.
If embeddings are unavailable (no key, model retired, quota) everything keeps working on keyword search.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import CHROMA_DIR, EMBEDDING_FALLBACK_MODEL, EMBEDDING_MODEL, LISTINGS_DB


# --------------------------------------------------------------------------- #
# What we embed
# --------------------------------------------------------------------------- #
def listing_document(row: Any) -> str:
    """The text that represents one listing in vector space."""
    return (f"{row['display_name']}. {row['title']}. Body type: {row['body_type']}. "
            f"{str(row['description'])[:1500]}")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _field(obj: Any, name: str) -> Any:
    return getattr(obj, name) if hasattr(obj, name) else obj[name]


class Embedder:
    """Text -> vectors through LiteLLM. Tries the primary model, then the fallback; retries rate limits."""

    def __init__(self, models: Optional[tuple[str, ...]] = None, batch_size: int = 32):
        self.models = [m for m in (models or (EMBEDDING_MODEL, EMBEDDING_FALLBACK_MODEL)) if m]
        self.batch_size = batch_size
        self.model: Optional[str] = None

    def _call(self, model: str, texts: list[str]) -> list[list[float]]:
        import litellm

        litellm.suppress_debug_info = True
        litellm.drop_params = True
        last: Optional[Exception] = None
        for wait in (0, 5, 15):
            if wait:
                time.sleep(wait)
            try:
                resp = litellm.embedding(model=model, input=texts, timeout=60)
                data = sorted(resp.data, key=lambda d: _field(d, "index"))
                return [list(_field(d, "embedding")) for d in data]
            except (litellm.RateLimitError, litellm.Timeout, litellm.ServiceUnavailableError,
                    litellm.InternalServerError, litellm.APIConnectionError) as e:
                last = e
        raise last  # type: ignore[misc]

    def probe(self) -> str:
        """Find the first embedding model that actually works for this API key."""
        if self.model:
            return self.model
        errors = []
        for m in self.models:
            try:
                self._call(m, ["probe"])
                self.model = m
                return m
            except Exception as e:  # noqa: BLE001 - we want to report every failure reason
                errors.append(f"{m}: {type(e).__name__}: {str(e)[:140]}")
        raise RuntimeError("No embedding model worked. " + " | ".join(errors))

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self.probe()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out += self._call(model, texts[i:i + self.batch_size])
        return out


# --------------------------------------------------------------------------- #
# Vector stores (same tiny interface)
# --------------------------------------------------------------------------- #
class MemoryVectorStore:
    """Fallback if chromadb is not installed: NumPy cosine similarity, not persisted."""
    persistent = False

    def __init__(self) -> None:
        self._vecs: dict[int, np.ndarray] = {}
        self._hash: dict[int, str] = {}

    def hashes(self) -> dict[int, str]:
        return dict(self._hash)

    def upsert(self, ids: list[int], vectors: list[list[float]], hashes: list[str], docs: list[str]) -> None:
        for i, v, h in zip(ids, vectors, hashes):
            self._vecs[i] = np.asarray(v, dtype=float)
            self._hash[i] = h

    def delete(self, ids: list[int]) -> None:
        for i in ids:
            self._vecs.pop(i, None)
            self._hash.pop(i, None)

    def count(self) -> int:
        return len(self._vecs)

    def query_all(self, vector: list[float]) -> dict[int, float]:
        q = np.asarray(vector, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        return {i: float(v @ q / (np.linalg.norm(v) or 1.0)) for i, v in self._vecs.items()}


class ChromaVectorStore:
    """ChromaDB persistent collection using cosine distance."""
    persistent = True

    def __init__(self, name: str, path: Path):
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._col = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"})

    def hashes(self) -> dict[int, str]:
        res = self._col.get(include=["metadatas"])
        return {int(i): (m or {}).get("hash", "") for i, m in zip(res["ids"], res["metadatas"])}

    def upsert(self, ids: list[int], vectors: list[list[float]], hashes: list[str], docs: list[str]) -> None:
        self._col.upsert(ids=[str(i) for i in ids], embeddings=vectors, documents=docs,
                         metadatas=[{"hash": h} for h in hashes])

    def delete(self, ids: list[int]) -> None:
        if ids:
            self._col.delete(ids=[str(i) for i in ids])

    def count(self) -> int:
        return self._col.count()

    def query_all(self, vector: list[float]) -> dict[int, float]:
        n = self._col.count()
        if n == 0:
            return {}
        res = self._col.query(query_embeddings=[list(vector)], n_results=n, include=["distances"])
        return {int(i): 1.0 - float(d) for i, d in zip(res["ids"][0], res["distances"][0])}


def make_store(name: str, chroma_dir: Path, use_chroma: bool = True):
    if use_chroma:
        try:
            return ChromaVectorStore(name, chroma_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[rag] ChromaDB unavailable ({type(e).__name__}: {str(e)[:100]}) - using in-memory vector store", flush=True)
    return MemoryVectorStore()


# --------------------------------------------------------------------------- #
# The index: SQLite table + vector DB
# --------------------------------------------------------------------------- #
class RagIndex:
    def __init__(self, embedder: Optional[Embedder] = None, db_path: Optional[Path] = None,
                 chroma_dir: Optional[Path] = None, use_chroma: bool = True):
        self.embedder = embedder or Embedder()
        self.db_path = Path(db_path or LISTINGS_DB)
        self.chroma_dir = Path(chroma_dir or CHROMA_DIR)
        self.use_chroma = use_chroma
        self.store: Any = None
        self.ready = False
        self.error: Optional[str] = None
        self.info: dict[str, Any] = {}
        self._qcache: dict[str, dict[int, float]] = {}
        self._lock = threading.Lock()

    # ---- 1) the table ---- #
    def write_table(self, df: pd.DataFrame) -> int:
        """Persist the cleaned + enriched listings as a real SQL table (inspect it with any SQLite viewer)."""
        cols = ["car_id", "year", "make", "model", "trim", "display_name", "body_type", "price_aed",
                "monthly_payment_aed", "mileage_km", "specs", "colors", "warranty", "warranty_note",
                "title", "description", "photo_url"]
        out = df[cols].copy()
        out["colors"] = out["colors"].apply(lambda cs: ", ".join(cs))
        out["warranty"] = out["warranty"].map({True: 1, False: 0})
        for c in ("price_aed", "monthly_payment_aed", "mileage_km", "warranty"):
            out[c] = out[c].astype(object).where(out[c].notna(), None)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            out.to_sql("listings", conn, if_exists="replace", index=False)
        return len(out)

    # ---- 2) embeddings -> vector DB ---- #
    def build(self, inventory: Any) -> dict[str, Any]:
        with self._lock:
            started = time.time()
            self.ready, self.error = False, None
            info: dict[str, Any] = {"table": str(self.db_path), "table_rows": self.write_table(inventory.df)}
            try:
                model = self.embedder.probe()
                name = ("listings_" + re.sub(r"[^A-Za-z0-9._-]+", "-", model))[:100].strip("-._") or "listings"
                self.store = make_store(name, self.chroma_dir, self.use_chroma)
                docs = {int(cid): listing_document(row) for cid, row in inventory.df.iterrows()}
                hashes = {cid: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for cid, t in docs.items()}
                have = self.store.hashes()
                todo = [cid for cid, h in hashes.items() if have.get(cid) != h]
                stale = [cid for cid in have if cid not in docs]
                if todo:
                    vecs = self.embedder.embed([docs[c] for c in todo])
                    self.store.upsert(todo, vecs, [hashes[c] for c in todo], [docs[c] for c in todo])
                if stale:
                    self.store.delete(stale)
                self._qcache.clear()
                self.ready = True
                info.update(embedding_model=model, vector_store=type(self.store).__name__, vectors=self.store.count(),
                            newly_embedded=len(todo))
            except Exception as e:  # noqa: BLE001 - never take the app down because embeddings failed
                self.error = f"{type(e).__name__}: {str(e)[:400]}"
                info["error"] = self.error
            info["seconds"] = round(time.time() - started, 1)
            self.info = info
            print(f"[rag] {info}", flush=True)
            return info

    # ---- 3) similarity search ---- #
    def similarities(self, query: str) -> Optional[dict[int, float]]:
        """car_id -> cosine similarity for EVERY listing, or None if vector search is unavailable."""
        if not self.ready or self.store is None:
            return None
        key = query.strip().lower()
        if key in self._qcache:
            return self._qcache[key]
        try:
            sims = self.store.query_all(self.embedder.embed([query])[0])
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {str(e)[:300]}"
            return None
        if len(self._qcache) > 200:
            self._qcache.clear()
        self._qcache[key] = sims
        return sims

    def status(self) -> dict[str, Any]:
        return {"ready": self.ready, "error": self.error, **self.info}