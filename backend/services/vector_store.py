# backend/services/vector_store.py
import os
import logging
from typing import List, Dict, Optional
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

try:
    # Primary option: chromadb
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMA_AVAILABLE = False

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    S_BERT_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    S_BERT_AVAILABLE = False


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class VectorStoreService:
    """
    Vector store abstraction. Tries Chromadb if available, otherwise uses a simple in-memory index.
    Exposes:
      - upsert_evidence_batch(batch)
      - search_similar_evidence(query_text, top_k)
      - get_index_stats()
    `batch` format expected by callers:
      [{"id": "<id>", "text": "...", "metadata": {...}}, ...]
    """

    def __init__(self):
        self.provider = Config.VECTOR_PROVIDER.lower() if hasattr(Config, "VECTOR_PROVIDER") else "chromadb"
        self.embedding_model_name = getattr(Config, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # init embedding model
        self._init_embedder()

        # initialize store
        if self.provider == "chromadb" and CHROMA_AVAILABLE:
            try:
                settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=getattr(Config, "CHROMA_PERSIST_DIR", None))
                self.client = chromadb.Client(settings)
                self.collection = self.client.get_or_create_collection(name="veripulse_evidence")
                logger.info("Chromadb collection ready")
                self._mode = "chromadb"
            except Exception as e:
                logger.warning(f"Chromadb init failed ({e}), falling back to in-memory store")
                self._init_inmemory()
        else:
            self._init_inmemory()

    def _init_embedder(self):
        if S_BERT_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(self.embedding_model_name)
                logger.info(f"SentenceTransformer loaded: {self.embedding_model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers model ({e}), embeddings disabled")
        # fallback: None -> we will use naive text hashing to produce vector-like values
        self.embedder = None

    def _init_inmemory(self):
        self._mode = "inmemory"
        # list of {"id","embedding":np.array,"text","metadata"}
        self._index = []
        logger.info("Initialized in-memory vector store")

    def upsert_evidence_batch(self, batch: List[Dict]):
        """
        Batch items expected:
          {"id": ..., "text": "...", "metadata": {...}}
        """
        if not batch:
            return

        if self._mode == "chromadb":
            ids = [b["id"] for b in batch]
            texts = [b["text"] for b in batch]
            metadatas = [b.get("metadata", {}) for b in batch]

            # compute embeddings
            embeddings = [self._embed_text(t) for t in texts]
            # chromadb expects a list of embeddings (list of floats)
            try:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            except Exception as e:
                logger.warning(f"Chromadb upsert failed: {e}")
        else:
            for b in batch:
                emb = self._embed_text(b["text"])
                self._index.append({
                    "id": b["id"],
                    "embedding": emb,
                    "text": b["text"],
                    "metadata": b.get("metadata", {})
                })

    def search_similar_evidence(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Returns a list of results with fields:
          id, score (similarity), metadata, text
        """
        if not query_text:
            return []

        q_emb = self._embed_text(query_text)

        if self._mode == "chromadb":
            try:
                results = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    include=['metadatas', 'documents', 'distances', 'ids']
                )
                # chromadb returns nested lists per query
                docs = []
                for i, _id in enumerate(results['ids'][0]):
                    docs.append({
                        "id": _id,
                        "score": 1 - results['distances'][0][i] if results['distances'][0][i] is not None else 0,
                        "metadata": results['metadatas'][0][i] or {},
                        "text": results['documents'][0][i] or ""
                    })
                return docs
            except Exception as e:
                logger.warning(f"Chromadb query failed: {e}")
                return []

        # in-memory search
        sims = []
        for item in self._index:
            score = _cosine_sim(q_emb, item["embedding"])
            sims.append((score, item))
        sims.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in sims[:top_k]:
            results.append({
                "id": item["id"],
                "score": float(score),
                "metadata": item.get("metadata", {}),
                "text": item.get("text", "")
            })
        return results

    def get_index_stats(self) -> Dict:
        if self._mode == "chromadb":
            try:
                size = self.collection.count()
            except Exception:
                size = None
            return {"provider": "chromadb", "collections": 1, "size": size}
        else:
            return {"provider": "inmemory", "collections": 1, "size": len(self._index)}

    def _embed_text(self, text: str) -> List[float]:
        """Return embedding as list[float]"""
        if not text:
            return np.zeros(384).tolist()  # safe default

        if self.embedder is not None:
            try:
                vec = self.embedder.encode(text, convert_to_numpy=True)
                # normalize
                if np.linalg.norm(vec) > 0:
                    vec = vec / np.linalg.norm(vec)
                return vec.tolist()
            except Exception as e:
                logger.warning(f"Embedding error: {e}")

        # fallback: hashed vector (deterministic, low quality)
        h = abs(hash(text)) % (10 ** 8)
        rng = np.random.RandomState(h)
        v = rng.normal(size=(384,))
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.tolist()
