"""
CriticModule — Self-evaluation engine for GRAG outputs.

Evaluates generated answers on:
  - Faithfulness (no hallucination)
  - Relevance (answers user intent)
  - Completeness
  - Logical consistency
"""

import re
import logging
from typing import List

from grag.core.models import (
    GRAGAnswer, QueryParsed, FusedContext, EvaluationResult, FailureType
)
from grag.core.config import GRAGConfig

logger = logging.getLogger(__name__)


class CriticModule:
    """
    Critic / Self-Evaluation module.

    Assigns scores to generated answers and identifies failure types
    for the refinement loop.

    Scoring
    -------
    - faithfulness  : are all claims grounded in the context?
    - relevance     : does it answer the user's query intent?
    - completeness  : are all key entities mentioned?
    - consistency   : no logical contradictions?

    Overall = weighted combination of above.

    Example
    -------
    >>> critic = CriticModule(config)
    >>> result = critic.evaluate(answer, parsed_query, fused_context)
    >>> result.passed   # True if score >= threshold
    """

    def __init__(self, config: GRAGConfig):
        self.config = config
        self._weights = {
            "faithfulness": 0.35,
            "relevance": 0.30,
            "completeness": 0.20,
            "consistency": 0.15,
        }

    def evaluate(
        self,
        answer: GRAGAnswer,
        parsed_query: QueryParsed,
        fused_context: FusedContext,
    ) -> EvaluationResult:
        """
        Run all evaluation metrics and return an EvaluationResult.
        """
        faith_score = self._faithfulness(answer, fused_context)
        rel_score = self._relevance(answer, parsed_query)
        comp_score = self._completeness(answer, parsed_query)
        cons_score = self._consistency(answer, fused_context)

        overall = (
            self._weights["faithfulness"] * faith_score +
            self._weights["relevance"] * rel_score +
            self._weights["completeness"] * comp_score +
            self._weights["consistency"] * cons_score
        )

        failure_type = FailureType.NONE
        notes = []

        if faith_score < 0.5:
            failure_type = FailureType.HALLUCINATION
            notes.append(f"Low faithfulness ({faith_score:.2f}): answer may contain hallucinations.")
        elif rel_score < 0.5:
            failure_type = FailureType.RETRIEVAL_FAILURE
            notes.append(f"Low relevance ({rel_score:.2f}): retrieved docs may not match query.")
        elif comp_score < 0.5:
            failure_type = FailureType.FUSION_ERROR
            notes.append(f"Low completeness ({comp_score:.2f}): key entities missing from answer.")
        elif cons_score < 0.5:
            failure_type = FailureType.REASONING_FAILURE
            notes.append(f"Low consistency ({cons_score:.2f}): contradictions detected.")

        result = EvaluationResult(
            faithfulness=faith_score,
            relevance=rel_score,
            completeness=comp_score,
            consistency=cons_score,
            overall_score=overall,
            failure_type=failure_type,
            notes=" | ".join(notes) if notes else "All checks passed.",
        )

        logger.debug(
            f"Critic: faith={faith_score:.2f} rel={rel_score:.2f} "
            f"comp={comp_score:.2f} cons={cons_score:.2f} overall={overall:.2f}"
        )
        return result

    def _faithfulness(self, answer: GRAGAnswer, ctx: FusedContext) -> float:
        """
        Check that answer content is grounded in graph facts + documents.
        Penalizes if answer introduces entities not in context.
        """
        if not answer.answer or answer.answer.startswith("Insufficient"):
            return 1.0  # safely abstained

        context_text = " ".join(ctx.graph_facts)
        context_text += " ".join(d.content for d in ctx.document_chunks)
        context_words = set(re.findall(r"\b\w+\b", context_text.lower()))

        answer_words = set(re.findall(r"\b\w{4,}\b", answer.answer.lower()))
        stop = {
            "this", "that", "with", "from", "have", "been", "they", "their",
            "which", "where", "when", "also", "more", "some", "used", "uses",
            "based", "known", "many", "such", "other", "into", "because",
            "these", "those", "then", "than", "about", "after", "before",
        }
        answer_words -= stop

        if not answer_words:
            return 0.8

        grounded = answer_words & context_words
        score = len(grounded) / len(answer_words)
        return min(score * 1.2, 1.0)  # slight boost since short answers score lower

    def _relevance(self, answer: GRAGAnswer, parsed_query: QueryParsed) -> float:
        """Check if the answer addresses the query intent and entities."""
        answer_lower = answer.answer.lower()
        query_lower = parsed_query.raw_query.lower()

        # Entity coverage
        entity_hits = sum(
            1 for e in parsed_query.entities
            if e.lower() in answer_lower
        )
        entity_score = (
            entity_hits / len(parsed_query.entities)
            if parsed_query.entities else 0.7
        )

        # Intent alignment
        intent_score = self._intent_alignment(parsed_query.intent, answer.answer)

        # Query keyword overlap
        query_words = set(re.findall(r"\b\w{4,}\b", query_lower))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer_lower))
        keyword_overlap = (
            len(query_words & answer_words) / len(query_words)
            if query_words else 0.5
        )

        return (0.4 * entity_score + 0.4 * intent_score + 0.2 * keyword_overlap)

    def _intent_alignment(self, intent: str, answer_text: str) -> float:
        """Check if answer structure aligns with the query intent."""
        answer_lower = answer_text.lower()
        intent_signals = {
            "definition": ["is", "refers to", "defined as", "means", "type of"],
            "comparison": ["whereas", "while", "compared to", "unlike", "both", "however"],
            "causal": ["because", "due to", "causes", "leads to", "results in", "reason"],
            "temporal": ["in", "year", "since", "during", "before", "after", "when"],
            "entity_info": ["is a", "was", "has", "works", "created", "born"],
            "listing": ["first", "second", "also", "include", "•", "1.", "2."],
            "factual": ["%", "million", "billion", "number", "count", "total"],
            "general": [],
        }
        signals = intent_signals.get(intent, [])
        if not signals:
            return 0.75
        hits = sum(1 for s in signals if s in answer_lower)
        return min(hits / max(len(signals) * 0.4, 1), 1.0)

    def _completeness(self, answer: GRAGAnswer, parsed_query: QueryParsed) -> float:
        """Check if all key entities from the query appear in the answer."""
        if not parsed_query.entities:
            return 0.8

        answer_lower = answer.answer.lower()
        covered = sum(
            1 for e in parsed_query.entities
            if e.lower() in answer_lower
        )
        base_score = covered / len(parsed_query.entities)

        # Penalize very short answers
        word_count = len(answer.answer.split())
        if word_count < 10:
            base_score *= 0.7
        elif word_count > 20:
            base_score = min(base_score + 0.1, 1.0)

        return base_score

    def _consistency(self, answer: GRAGAnswer, ctx: FusedContext) -> float:
        """Penalize contradictions in the fused context."""
        if not ctx.contradictions:
            return 1.0
        penalty = len(ctx.contradictions) * 0.15
        return max(1.0 - penalty, 0.0)
