"""
GRAG Data Models — Typed structures used across the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class FailureType(Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    REASONING_FAILURE = "reasoning_failure"
    FUSION_ERROR = "fusion_error"
    HALLUCINATION = "hallucination"
    NONE = "none"


@dataclass
class QueryParsed:
    """Output of the Query Understanding module."""
    raw_query: str
    intent: str
    entities: List[str]
    relationships: List[str]
    constraints: Dict[str, Any]
    semantic_query: str
    graph_query: str
    ambiguity_score: float = 0.0


@dataclass
class RetrievedDocument:
    """A single retrieved document chunk."""
    doc_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPath:
    """A multi-hop path in the knowledge graph."""
    path: List[Tuple[str, str, str]]   # (entity, relation, entity) triples
    confidence: float
    recency_score: float = 1.0

    def to_string(self) -> str:
        parts = []
        for s, r, o in self.path:
            parts.append(f"{s} --[{r}]--> {o}")
        return " | ".join(parts)


@dataclass
class FusedContext:
    """Merged context from graph + documents."""
    graph_facts: List[str]
    document_chunks: List[RetrievedDocument]
    contradictions: List[str]
    confidence: float


@dataclass
class GRAGAnswer:
    """
    Final output of the GRAG pipeline.

    Fields
    ------
    answer : str
        The generated answer text.
    graph_path : GraphPath
        Supporting knowledge graph path.
    document_summary : str
        Summary of supporting documents.
    confidence : float
        Overall confidence score (0–1).
    entities_used : List[str]
        Key entities involved in reasoning.
    failure_type : FailureType
        Indicates if a failure occurred.
    iterations : int
        Number of refinement iterations used.
    """
    answer: str
    graph_path: Optional[GraphPath]
    document_summary: str
    confidence: float
    entities_used: List[str]
    failure_type: FailureType = FailureType.NONE
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        gp = self.graph_path.to_string() if self.graph_path else "N/A"
        return (
            f"\n{'='*60}\n"
            f"ANSWER:\n{self.answer}\n\n"
            f"GRAPH PATH:\n{gp}\n\n"
            f"DOCUMENT SUPPORT:\n{self.document_summary}\n\n"
            f"CONFIDENCE: {self.confidence:.2f}\n"
            f"ITERATIONS: {self.iterations}\n"
            f"{'='*60}"
        )


@dataclass
class EvaluationResult:
    """Output of the Critic Module."""
    faithfulness: float
    relevance: float
    completeness: float
    consistency: float
    overall_score: float
    failure_type: FailureType = FailureType.NONE
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.7
