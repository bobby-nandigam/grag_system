"""
GRAGConfig — Central configuration for the GRAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRAGConfig:
    """
    Central configuration for the GRAG pipeline.

    Attributes
    ----------
    top_k : int
        Number of documents retrieved per query.
    max_hops : int
        Maximum graph traversal depth for multi-hop reasoning.
    confidence_threshold : float
        Minimum confidence score to accept an answer (0–1).
    max_refinement_iterations : int
        Max number of self-refinement loops.
    embedding_model : str
        Sentence embedding model name (HuggingFace or OpenAI).
    llm_model : str
        LLM model used for generation.
    use_gpu : bool
        Whether to use CUDA acceleration.
    memory_path : str
        Path to persist memory store.
    verbose : bool
        Enable detailed logging.
    """

    top_k: int = 5
    max_hops: int = 3
    confidence_threshold: float = 0.7
    max_refinement_iterations: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    use_gpu: bool = False
    memory_path: str = ".grag_memory"
    verbose: bool = False
    reward_decay: float = 0.95
    hallucination_penalty: float = -1.0
    faithfulness_reward: float = 1.0
    retrieval_noise_threshold: float = 0.3
    contradiction_penalty: float = -0.5
    graph_weight: float = 0.6
    vector_weight: float = 0.4

    def __post_init__(self):
        assert 0 < self.confidence_threshold <= 1, "confidence_threshold must be in (0, 1]"
        assert self.graph_weight + self.vector_weight == 1.0, \
            "graph_weight + vector_weight must equal 1.0"

    @classmethod
    def fast(cls) -> "GRAGConfig":
        """Lightweight config for quick prototyping."""
        return cls(top_k=3, max_hops=2, max_refinement_iterations=1, verbose=False)

    @classmethod
    def production(cls) -> "GRAGConfig":
        """High-accuracy config for production deployments."""
        return cls(top_k=10, max_hops=4, max_refinement_iterations=5,
                   confidence_threshold=0.85, verbose=True)
