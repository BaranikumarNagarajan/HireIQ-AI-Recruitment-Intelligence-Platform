"""Semantic Search Retriever."""
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL, TOP_K
from utils.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)

# SentenceTransformer cached at module level — loading takes several seconds
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def retrieve_legal_context(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant legal context from ChromaDB."""
    try:
        model = _get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        query_embeddings = [np.asarray(query_embedding, dtype=np.float32).tolist()]

        collection = get_chroma_client().get_collection(name="legal_documents")
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(n_results, TOP_K),
        )

        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        if not documents or not metadatas or not distances:
            return []

        docs = documents[0]
        metas = metadatas[0]
        dists = distances[0]
        return [
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "unknown"),
                "relevance_score": 1 - dist,
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    except Exception as e:
        logger.error("Failed to retrieve legal context: %s", e)
        return []
