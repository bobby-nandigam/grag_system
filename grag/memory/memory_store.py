"""
MemoryStore — Episodic memory for successful query patterns and graph paths.

Stores:
  - High-quality (query, answer) pairs
  - Successful graph paths
  - Failure cases for avoidance
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    query: str
    intent: str
    answer_summary: str
    graph_path_str: str
    confidence: float
    reward: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = field(default_factory=list)


class MemoryStore:
    """
    Episodic memory for the GRAG system.

    Stores successful (query, answer) pairs indexed by intent.
    Avoids repeating failure patterns.
    Can suggest cached graph paths for known query types.

    Example
    -------
    >>> mem = MemoryStore()
    >>> mem.store(query, answer, graph_path, reward=0.9)
    >>> suggestion = mem.retrieve_similar("Who created Python?")
    """

    def __init__(self, path: Optional[str] = None):
        self._entries: List[MemoryEntry] = []
        self._failure_patterns: List[str] = []
        self._path = path

        if path and Path(path).exists():
            self.load(path)

    def store(
        self,
        query: str,
        intent: str,
        answer_summary: str,
        graph_path_str: str,
        confidence: float,
        reward: float,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a successful query-answer pair."""
        if reward < 0:
            self._failure_patterns.append(query[:80])
            logger.debug(f"Stored failure pattern: '{query[:60]}'")
            return

        entry = MemoryEntry(
            query=query,
            intent=intent,
            answer_summary=answer_summary[:300],
            graph_path_str=graph_path_str,
            confidence=confidence,
            reward=reward,
            tags=tags or [],
        )
        self._entries.append(entry)
        logger.debug(f"Memory stored: '{query[:60]}' | reward={reward:.2f}")

        if self._path:
            self.save(self._path)

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        """
        Return top-k similar memory entries by keyword overlap.
        """
        if not self._entries:
            return []

        query_words = set(query.lower().split())
        scored = []
        for entry in self._entries:
            entry_words = set(entry.query.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def is_failure_pattern(self, query: str) -> bool:
        """Check if this query resembles a known failure pattern."""
        q_words = set(query.lower().split())
        for pattern in self._failure_patterns:
            p_words = set(pattern.lower().split())
            overlap = len(q_words & p_words) / max(len(p_words), 1)
            if overlap > 0.5:
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self._entries),
            "failure_patterns": len(self._failure_patterns),
            "avg_confidence": (
                sum(e.confidence for e in self._entries) / len(self._entries)
                if self._entries else 0.0
            ),
            "avg_reward": (
                sum(e.reward for e in self._entries) / len(self._entries)
                if self._entries else 0.0
            ),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [asdict(e) for e in self._entries],
            "failure_patterns": self._failure_patterns,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._entries = [MemoryEntry(**e) for e in data.get("entries", [])]
        self._failure_patterns = data.get("failure_patterns", [])
        logger.info(f"Memory loaded: {self.get_stats()}")
