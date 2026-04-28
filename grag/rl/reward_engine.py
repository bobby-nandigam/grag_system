"""
RewardEngine — Reinforcement learning feedback loop for GRAG.

Maintains a reward table keyed by (intent, query_pattern) pairs.
Updates retrieval weights and traversal depth based on reward signals.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from grag.core.models import GRAGAnswer, EvaluationResult, QueryParsed, FailureType
from grag.core.config import GRAGConfig

logger = logging.getLogger(__name__)


class RewardEngine:
    """
    RL reward/penalty engine for GRAG self-improvement.

    Maintains cumulative reward signals per query pattern.
    Adapts:
      - graph_weight vs vector_weight balance
      - retrieval top_k
      - traversal max_hops

    Reward Signals
    --------------
    +1.0  Correct, faithful answer
    +0.5  Partially correct
    -1.0  Hallucination detected
    -0.5  Contradiction in output
    -0.3  Missed key entities
    -0.2  Irrelevant retrieval

    Example
    -------
    >>> engine = RewardEngine(config)
    >>> engine.record(answer, eval_result, parsed_query, user_feedback=1.0)
    >>> engine.get_adapted_config(parsed_query)
    """

    def __init__(self, config: GRAGConfig):
        self.config = config
        self._rewards: Dict[str, List[float]] = defaultdict(list)
        self._strategy: Dict[str, Dict] = defaultdict(lambda: {
            "graph_weight": config.graph_weight,
            "vector_weight": config.vector_weight,
            "top_k": config.top_k,
            "max_hops": config.max_hops,
        })
        self._failure_counts: Dict[FailureType, int] = defaultdict(int)
        self._total_queries = 0
        self._total_reward = 0.0

    def record(
        self,
        answer: GRAGAnswer,
        eval_result: EvaluationResult,
        parsed_query: QueryParsed,
        user_feedback: Optional[float] = None,
    ) -> float:
        """
        Record a reward signal and update strategy.

        Parameters
        ----------
        answer : GRAGAnswer
        eval_result : EvaluationResult
        parsed_query : QueryParsed
        user_feedback : float in [-1, 1], optional explicit user signal

        Returns
        -------
        float : total reward computed
        """
        reward = self._compute_reward(eval_result, user_feedback)
        key = self._pattern_key(parsed_query)

        self._rewards[key].append(reward)
        self._failure_counts[eval_result.failure_type] += 1
        self._total_queries += 1
        self._total_reward += reward

        self._update_strategy(key, eval_result)

        logger.debug(
            f"RL reward={reward:+.2f} | key={key} | "
            f"failure={eval_result.failure_type.value}"
        )
        return reward

    def get_adapted_config(self, parsed_query: QueryParsed) -> Dict:
        """
        Return adapted strategy parameters for the given query pattern.
        Used by the pipeline to adjust retrieval on refinement loops.
        """
        key = self._pattern_key(parsed_query)
        return dict(self._strategy[key])

    def average_reward(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return self._total_reward / self._total_queries

    def stats(self) -> Dict:
        return {
            "total_queries": self._total_queries,
            "average_reward": round(self.average_reward(), 3),
            "failure_breakdown": {
                k.value: v for k, v in self._failure_counts.items()
            },
            "patterns_learned": len(self._rewards),
        }

    def save(self, path: str) -> None:
        data = {
            "rewards": dict(self._rewards),
            "strategy": dict(self._strategy),
            "total_queries": self._total_queries,
            "total_reward": self._total_reward,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._rewards = defaultdict(list, data.get("rewards", {}))
        self._strategy = defaultdict(
            lambda: {
                "graph_weight": self.config.graph_weight,
                "vector_weight": self.config.vector_weight,
                "top_k": self.config.top_k,
                "max_hops": self.config.max_hops,
            },
            data.get("strategy", {})
        )
        self._total_queries = data.get("total_queries", 0)
        self._total_reward = data.get("total_reward", 0.0)

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _compute_reward(
        self,
        eval_result: EvaluationResult,
        user_feedback: Optional[float],
    ) -> float:
        reward = 0.0

        # Faithfulness rewards/penalties
        reward += self.config.faithfulness_reward * eval_result.faithfulness

        # Hallucination penalty
        if eval_result.failure_type == FailureType.HALLUCINATION:
            reward += self.config.hallucination_penalty

        # Contradiction penalty
        if eval_result.failure_type == FailureType.REASONING_FAILURE:
            reward += self.config.contradiction_penalty

        # Retrieval failure
        if eval_result.failure_type == FailureType.RETRIEVAL_FAILURE:
            reward -= 0.2

        # Completeness bonus
        reward += 0.3 * eval_result.completeness

        # User explicit feedback (weighted heavily)
        if user_feedback is not None:
            reward += 0.5 * user_feedback

        # Decay over time
        reward *= self.config.reward_decay

        return round(reward, 4)

    def _update_strategy(self, key: str, eval_result: EvaluationResult) -> None:
        """
        Adapt retrieval strategy based on failure type.

        - RETRIEVAL_FAILURE → increase vector_weight, expand top_k
        - REASONING_FAILURE → increase graph_weight, expand max_hops
        - HALLUCINATION     → reduce both weights, tighten thresholds
        - FUSION_ERROR      → rebalance weights
        """
        strategy = self._strategy[key]

        if eval_result.failure_type == FailureType.RETRIEVAL_FAILURE:
            strategy["vector_weight"] = min(strategy["vector_weight"] + 0.05, 0.8)
            strategy["graph_weight"] = 1.0 - strategy["vector_weight"]
            strategy["top_k"] = min(strategy["top_k"] + 2, 20)

        elif eval_result.failure_type == FailureType.REASONING_FAILURE:
            strategy["graph_weight"] = min(strategy["graph_weight"] + 0.05, 0.85)
            strategy["vector_weight"] = 1.0 - strategy["graph_weight"]
            strategy["max_hops"] = min(strategy["max_hops"] + 1, 5)

        elif eval_result.failure_type == FailureType.HALLUCINATION:
            # Tighten — be more conservative
            strategy["top_k"] = max(strategy["top_k"] - 1, 3)
            strategy["max_hops"] = max(strategy["max_hops"] - 1, 1)

        elif eval_result.failure_type == FailureType.FUSION_ERROR:
            # Rebalance toward defaults
            strategy["graph_weight"] = self.config.graph_weight
            strategy["vector_weight"] = self.config.vector_weight

        self._strategy[key] = strategy

    def _pattern_key(self, parsed_query: QueryParsed) -> str:
        """Create a stable string key from intent + top entity."""
        top_entity = parsed_query.entities[0] if parsed_query.entities else "unknown"
        raw = f"{parsed_query.intent}:{top_entity.lower()[:20]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
