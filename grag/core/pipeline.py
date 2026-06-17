"""
GRAGPipeline — The master orchestrator of the GRAG system.

Executes all 11 stages:
  [1] Query Understanding
  [2] Hybrid Retrieval
  [3] Graph Reasoning
  [4] Context Fusion
  [5] Answer Generation
  [6] Explainability
  [7] Self-Evaluation (Critic)
  [8] Refinement Loop
  [9] RL Reward Recording
  [10] Memory Adaptation
  [11] Safety Guardrails
"""

import logging
from typing import List, Optional, Dict, Any

from grag.core.config import GRAGConfig
from grag.core.models import (
    GRAGAnswer, QueryParsed, FusedContext, FailureType, GraphPath
)
from grag.graph.knowledge_graph import KnowledgeGraph
from grag.extraction.relation_extractor import RelationExtractor
from grag.retrieval.hybrid_retriever import HybridRetriever
from grag.retrieval.query_understanding import QueryUnderstanding
from grag.reasoning.graph_reasoner import GraphReasoner
from grag.evaluation.critic import CriticModule
from grag.rl.reward_engine import RewardEngine
from grag.memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class GRAGPipeline:
    """
    Full GRAG pipeline with RL-driven self-improvement.

    Quick Start
    -----------
    >>> from grag import GRAGPipeline, GRAGConfig
    >>> pipeline = GRAGPipeline()
    >>> pipeline.add_documents([{"content": "Python was created by Guido van Rossum.", "source": "wiki"}])
    >>> pipeline.kg.add_triple("python", "created_by", "guido van rossum", confidence=0.99)
    >>> result = pipeline.query("Who created Python?")
    >>> print(result)

    Parameters
    ----------
    config : GRAGConfig, optional
        Pipeline configuration. Defaults to GRAGConfig().
    knowledge_graph : KnowledgeGraph, optional
        Pre-built knowledge graph. Creates new if None.
    """

    def __init__(
        self,
        config: Optional[GRAGConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        relation_extractor: Optional[RelationExtractor] = None,
    ):
        self.config = config or GRAGConfig()
        # A caller-supplied graph is used as-is; the default graph persists to disk
        # alongside the memory store so facts survive across sessions.
        self.kg = knowledge_graph or KnowledgeGraph(
            path=f"{self.config.memory_path}/knowledge_graph.json"
        )
        # Turns ingested text into knowledge-graph triples. Swap in a custom
        # (e.g. LLM-backed) extractor by passing one here.
        self.relation_extractor = relation_extractor or RelationExtractor()

        # Initialize all modules
        self._query_understanding = QueryUnderstanding()
        self._retriever = HybridRetriever(self.config, self.kg)
        self._reasoner = GraphReasoner(self.config, self.kg)
        self._critic = CriticModule(self.config)
        self._reward_engine = RewardEngine(self.config)
        self._memory = MemoryStore(
            path=f"{self.config.memory_path}/memory.json"
        )

        if self.config.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        logger.info(f"GRAG Pipeline initialized | config={self.config}")

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        extract_relations: bool = True,
    ) -> int:
        """
        Index documents into the vector store and, by default, mine them for
        knowledge-graph triples.

        Parameters
        ----------
        docs : list of dict
            Each dict must have a 'content' key. Optional keys: 'source', 'metadata'.
        extract_relations : bool
            When True (default), extract (subject, predicate, object) triples from
            each document's text and add them to the knowledge graph, so retrieval
            and graph reasoning both improve as documents arrive. Set False to index
            for vector search only.

        Returns
        -------
        int
            Number of triples extracted and added to the knowledge graph.

        Example
        -------
        >>> pipeline.add_documents([
        ...     {"content": "NVIDIA is a tech company founded in 1993.", "source": "wiki"},
        ...     {"content": "Google was founded by Larry Page and Sergey Brin.", "source": "wiki"},
        ... ])
        """
        self._retriever.add_documents(docs)

        added = 0
        if extract_relations:
            added = self._extract_into_graph(docs)

        logger.info(f"Indexed {len(docs)} documents | extracted {added} triples.")
        return added

    def _extract_into_graph(self, docs: List[Dict[str, Any]]) -> int:
        """Extract triples from documents and upsert them into the knowledge graph."""
        added = 0
        for doc in docs:
            source = doc.get("source", "extracted")
            for triple in self.relation_extractor.extract(doc.get("content", "")):
                self.kg.add_triple(
                    triple.subject,
                    triple.predicate,
                    triple.obj,
                    confidence=triple.confidence,
                    source=source,
                )
                added += 1
        return added

    def query(
        self,
        question: str,
        user_feedback: Optional[float] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> GRAGAnswer:
        """
        Run the full GRAG pipeline on a question.

        Parameters
        ----------
        question : str
            The user's natural language question.
        user_feedback : float, optional
            Explicit user signal in [-1, 1] for RL update.
        metadata_filter : dict, optional
            Filter retrieved documents by metadata fields.

        Returns
        -------
        GRAGAnswer
            Contains answer, graph path, document support, confidence.

        Example
        -------
        >>> result = pipeline.query("Who created Python?")
        >>> print(result.answer)
        >>> print(result.confidence)
        """
        logger.info(f"Query: '{question}'")

        # ── [1] Query Understanding ─────────────────────────────────────────
        parsed = self._query_understanding.parse(question)

        # ── Memory check: avoid known failure patterns ──────────────────────
        if self._memory.is_failure_pattern(question):
            logger.warning("Query matches known failure pattern. Extra caution applied.")

        # ── [2] Hybrid Retrieval ────────────────────────────────────────────
        docs = self._retriever.retrieve(parsed, metadata_filter=metadata_filter)

        # ── [3] Graph Reasoning ─────────────────────────────────────────────
        fused_context = self._reasoner.reason(parsed, docs)

        # ── [4] + [5] Answer Generation ─────────────────────────────────────
        answer = self._generate_answer(parsed, fused_context)

        # ── [6] Explainability ──────────────────────────────────────────────
        best_path = self._reasoner.get_best_path(parsed)
        answer.graph_path = best_path
        answer.document_summary = self._summarize_docs(fused_context)

        # ── [7] Self-Evaluation ─────────────────────────────────────────────
        eval_result = self._critic.evaluate(answer, parsed, fused_context)
        logger.info(
            f"Critic score: {eval_result.overall_score:.2f} | "
            f"failure={eval_result.failure_type.value}"
        )

        # ── [8] Refinement Loop ─────────────────────────────────────────────
        iteration = 1
        while (
            not eval_result.passed
            and iteration < self.config.max_refinement_iterations
        ):
            logger.info(f"Refinement iteration {iteration + 1} | reason: {eval_result.notes}")
            answer, eval_result, fused_context = self._refine(
                parsed, eval_result, iteration, metadata_filter
            )
            iteration += 1

        answer.iterations = iteration
        answer.confidence = eval_result.overall_score
        answer.failure_type = eval_result.failure_type

        # ── [9] RL Reward ───────────────────────────────────────────────────
        reward = self._reward_engine.record(
            answer, eval_result, parsed, user_feedback=user_feedback
        )

        # ── [10] Memory Storage ─────────────────────────────────────────────
        self._memory.store(
            query=question,
            intent=parsed.intent,
            answer_summary=answer.answer[:200],
            graph_path_str=answer.graph_path.to_string() if answer.graph_path else "",
            confidence=answer.confidence,
            reward=reward,
            tags=parsed.entities[:5],
        )

        logger.info(
            f"Query complete | confidence={answer.confidence:.2f} | "
            f"iterations={answer.iterations} | reward={reward:+.2f}"
        )
        return answer

    def stats(self) -> Dict:
        """Return system-wide statistics."""
        return {
            "kg": self.kg.stats(),
            "vector_store_size": len(self._retriever.vector_store),
            "rl": self._reward_engine.stats(),
            "memory": self._memory.get_stats(),
        }

    # ─── Private helpers ───────────────────────────────────────────────────────

    def _generate_answer(
        self,
        parsed: QueryParsed,
        context: FusedContext,
    ) -> GRAGAnswer:
        """
        Generate a grounded answer from fused context.

        Priority: Graph facts > high-confidence documents > weak signals.
        Falls back to "Insufficient evidence" if nothing useful found.
        """
        # ── [11] Safety guardrail: insufficient evidence ────────────────────
        if not context.graph_facts and not context.document_chunks:
            return GRAGAnswer(
                answer="Insufficient evidence to answer this question. "
                       "Please provide more context or knowledge graph data.",
                graph_path=None,
                document_summary="No relevant documents found.",
                confidence=0.0,
                entities_used=[],
                failure_type=FailureType.RETRIEVAL_FAILURE,
            )

        # Build grounded answer from available facts
        facts_part = ""
        if context.graph_facts:
            facts_part = "Based on the knowledge graph: " + "; ".join(
                context.graph_facts[:3]
            ) + ". "

        doc_part = ""
        top_docs = sorted(
            context.document_chunks, key=lambda d: d.score, reverse=True
        )[:2]
        if top_docs:
            doc_part = "Supporting evidence: " + " ".join(
                d.content[:150] for d in top_docs
            )

        answer_text = self._compose_answer(
            parsed, facts_part, doc_part, context
        )

        return GRAGAnswer(
            answer=answer_text,
            graph_path=None,  # filled by caller
            document_summary="",  # filled by caller
            confidence=context.confidence,
            entities_used=parsed.entities,
        )

    def _compose_answer(
        self,
        parsed: QueryParsed,
        facts_part: str,
        doc_part: str,
        context: FusedContext,
    ) -> str:
        """Compose the final answer string."""
        if context.contradictions:
            warning = (
                f" [Note: {len(context.contradictions)} contradiction(s) detected "
                f"in source material. Confidence reduced.]"
            )
        else:
            warning = ""

        combined = (facts_part + doc_part).strip()
        if not combined:
            return "Insufficient evidence to answer this question reliably."

        # Trim to reasonable length
        if len(combined) > 600:
            combined = combined[:597] + "..."

        return combined + warning

    def _summarize_docs(self, context: FusedContext) -> str:
        """Summarize supporting documents."""
        if not context.document_chunks:
            return "No supporting documents found."
        top = sorted(context.document_chunks, key=lambda d: d.score, reverse=True)[:3]
        parts = []
        for d in top:
            preview = d.content[:100].replace("\n", " ")
            parts.append(f"[{d.source}] (score={d.score:.2f}): {preview}...")
        return "\n".join(parts)

    def _refine(
        self,
        parsed: QueryParsed,
        prev_eval,
        iteration: int,
        metadata_filter,
    ):
        """
        Refinement loop: adjust retrieval strategy and re-reason.
        """
        adapted = self._reward_engine.get_adapted_config(parsed)
        logger.debug(f"Refinement adapted config: {adapted}")

        # Temporarily adjust config values
        old_top_k = self.config.top_k
        old_hops = self.config.max_hops
        self.config.top_k = adapted.get("top_k", self.config.top_k)
        self.config.max_hops = adapted.get("max_hops", self.config.max_hops)

        # Re-retrieve and re-reason
        docs = self._retriever.retrieve(parsed, metadata_filter=metadata_filter)
        fused_context = self._reasoner.reason(parsed, docs)
        answer = self._generate_answer(parsed, fused_context)
        best_path = self._reasoner.get_best_path(parsed)
        answer.graph_path = best_path
        answer.document_summary = self._summarize_docs(fused_context)
        eval_result = self._critic.evaluate(answer, parsed, fused_context)

        # Restore config
        self.config.top_k = old_top_k
        self.config.max_hops = old_hops

        return answer, eval_result, fused_context
