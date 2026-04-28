"""
HybridRetriever — Combines vector similarity search with graph-based retrieval.

Supports:
  - Sentence-transformer embeddings (local, no API key needed)
  - Adaptive top-k based on confidence
  - Metadata filtering
  - Provenance tracking
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any

import numpy as np

from grag.core.models import RetrievedDocument, QueryParsed
from grag.core.config import GRAGConfig
from grag.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Lightweight in-memory vector store using cosine similarity.
    Drop-in replacement — swap with FAISS, Pinecone, Weaviate, etc.
    """

    def __init__(self):
        self._docs: List[RetrievedDocument] = []
        self._embeddings: Optional[np.ndarray] = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded SentenceTransformer: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using TF-IDF fallback. Install via: pip install grag[ml]"
                )
                self._encoder = _TFIDFFallback()
        return self._encoder

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Parameters
        ----------
        docs : list of dicts with keys: 'content', 'source', 'metadata' (optional)
        """
        encoder = self._get_encoder()
        texts = [d["content"] for d in docs]
        embeddings = encoder.encode(texts)

        for i, doc in enumerate(docs):
            doc_id = hashlib.md5(doc["content"].encode()).hexdigest()[:8]
            self._docs.append(RetrievedDocument(
                doc_id=doc_id,
                content=doc["content"],
                score=0.0,
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata", {}),
            ))

        if self._embeddings is None:
            self._embeddings = np.array(embeddings)
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        logger.debug(f"Added {len(docs)} documents. Total: {len(self._docs)}")

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Return top-k most similar documents to the query."""
        if not self._docs:
            return []

        encoder = self._get_encoder()
        q_emb = np.array(encoder.encode([query])[0])
        scores = _cosine_similarity(q_emb, self._embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self._docs[idx]
            results.append(RetrievedDocument(
                doc_id=doc.doc_id,
                content=doc.content,
                score=float(scores[idx]),
                source=doc.source,
                metadata=doc.metadata,
            ))
        return results

    def __len__(self):
        return len(self._docs)


class HybridRetriever:
    """
    Hybrid retrieval combining:
      1. Vector similarity search (semantic)
      2. Knowledge graph neighbor expansion (structural)

    Adaptive k: increases top_k if average confidence < threshold.

    Example
    -------
    >>> retriever = HybridRetriever(config, knowledge_graph)
    >>> retriever.add_documents([{"content": "...", "source": "wiki"}])
    >>> docs = retriever.retrieve(parsed_query)
    """

    def __init__(self, config: GRAGConfig, knowledge_graph: KnowledgeGraph):
        self.config = config
        self.kg = knowledge_graph
        self.vector_store = VectorStore()

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Index a list of documents into the vector store."""
        self.vector_store.add_documents(docs)

    def retrieve(
        self,
        parsed_query: QueryParsed,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:
        """
        Perform hybrid retrieval.

        1. Vector search on semantic_query
        2. Entity-neighbor expansion via knowledge graph
        3. Adaptive k if confidence low
        4. Metadata filtering
        5. Deduplication + re-ranking
        """
        k = self.config.top_k

        # Step 1: Vector retrieval
        vector_results = self.vector_store.search(parsed_query.semantic_query, top_k=k)

        # Adaptive k: if confidence is low, fetch more
        avg_score = np.mean([d.score for d in vector_results]) if vector_results else 0.0
        if avg_score < self.config.retrieval_noise_threshold and k < 20:
            logger.debug(f"Low avg score ({avg_score:.2f}), expanding k to {k * 2}")
            vector_results = self.vector_store.search(
                parsed_query.semantic_query, top_k=min(k * 2, 20)
            )

        # Step 2: Graph-expanded retrieval (entity-aware)
        graph_expanded = self._graph_expand(parsed_query.entities, vector_results)

        # Step 3: Metadata filtering
        if metadata_filter:
            graph_expanded = [
                d for d in graph_expanded
                if all(d.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        # Step 4: Deduplicate + re-rank
        final = _deduplicate(graph_expanded)
        final.sort(key=lambda d: d.score, reverse=True)

        logger.debug(f"Retrieved {len(final)} documents for query: '{parsed_query.raw_query[:60]}'")
        return final[:self.config.top_k * 2]

    def _graph_expand(
        self,
        entities: List[str],
        base_results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Boost scores of documents that mention graph-neighboring entities.
        """
        neighbors = set()
        for entity in entities:
            neighbors.update(self.kg.get_neighbors(entity, depth=1))

        boosted = []
        for doc in base_results:
            boost = sum(
                1 for n in neighbors
                if n.lower() in doc.content.lower()
            ) * 0.05
            boosted.append(RetrievedDocument(
                doc_id=doc.doc_id,
                content=doc.content,
                score=min(doc.score + boost, 1.0),
                source=doc.source,
                metadata=doc.metadata,
            ))
        return boosted


# ─── Utilities ────────────────────────────────────────────────────────────────

def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normed = matrix / norms
    return normed @ (vec / (np.linalg.norm(vec) + 1e-8))


def _deduplicate(docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
    seen = set()
    unique = []
    for d in docs:
        if d.doc_id not in seen:
            seen.add(d.doc_id)
            unique.append(d)
    return unique


class _TFIDFFallback:
    """Minimal TF-IDF encoder for when sentence-transformers is unavailable."""

    def __init__(self):
        self._fitted = False
        self._vocab: Dict[str, int] = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            words = set()
            for t in texts:
                words.update(t.lower().split())
            self._vocab = {w: i for i, w in enumerate(sorted(words))}
            self._fitted = True

        vecs = []
        for t in texts:
            vec = np.zeros(len(self._vocab))
            for word in t.lower().split():
                if word in self._vocab:
                    vec[self._vocab[word]] += 1
            norm = np.linalg.norm(vec)
            vecs.append(vec / (norm + 1e-8))
        return np.array(vecs)
