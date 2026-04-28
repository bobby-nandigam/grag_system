"""
GRAG - Graph Retrieval-Augmented Generation System
===================================================
A production-grade, self-improving RAG system combining:
  - Knowledge Graph reasoning
  - Vector similarity search
  - Reinforcement learning feedback loops
  - Explainable AI outputs

Author: GRAG Contributors
License: MIT
"""

__version__ = "1.0.0"
__author__ = "GRAG Contributors"
__license__ = "MIT"

from grag.core.pipeline import GRAGPipeline
from grag.core.config import GRAGConfig
from grag.graph.knowledge_graph import KnowledgeGraph
from grag.retrieval.hybrid_retriever import HybridRetriever
from grag.reasoning.graph_reasoner import GraphReasoner
from grag.evaluation.critic import CriticModule
from grag.rl.reward_engine import RewardEngine
from grag.memory.memory_store import MemoryStore

__all__ = [
    "GRAGPipeline",
    "GRAGConfig",
    "KnowledgeGraph",
    "HybridRetriever",
    "GraphReasoner",
    "CriticModule",
    "RewardEngine",
    "MemoryStore",
]
