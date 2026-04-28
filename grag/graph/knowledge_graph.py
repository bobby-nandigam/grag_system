"""
KnowledgeGraph — In-memory + persistent knowledge graph with multi-hop traversal.

Uses NetworkX under the hood. Can be swapped with Neo4j for production.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

import networkx as nx

from grag.core.models import GraphPath

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Directed knowledge graph supporting:
      - Triple (subject, predicate, object) ingestion
      - Multi-hop path traversal
      - Confidence-weighted edge scoring
      - Subgraph extraction
      - Persistence (JSON)

    Example
    -------
    >>> kg = KnowledgeGraph()
    >>> kg.add_triple("Python", "created_by", "Guido van Rossum", confidence=0.99)
    >>> kg.add_triple("Guido van Rossum", "works_at", "Google", confidence=0.95)
    >>> paths = kg.find_paths("Python", "Google", max_hops=2)
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "manual",
        recency: float = 1.0,
    ) -> None:
        """Add a (subject, predicate, object) triple to the graph."""
        subject = subject.strip().lower()
        obj = obj.strip().lower()
        predicate = predicate.strip().lower()

        self.graph.add_node(subject, label=subject)
        self.graph.add_node(obj, label=obj)
        self.graph.add_edge(
            subject, obj,
            relation=predicate,
            confidence=confidence,
            source=source,
            recency=recency,
            weight=confidence * recency,
        )
        self._entity_index[subject].add(obj)
        logger.debug(f"Triple added: ({subject}) --[{predicate}]--> ({obj})")

    def add_triples_bulk(self, triples: List[Tuple[str, str, str]], **kwargs) -> None:
        """Bulk-add a list of (subject, predicate, object) triples."""
        for s, p, o in triples:
            self.add_triple(s, p, o, **kwargs)

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        top_k: int = 5,
    ) -> List[GraphPath]:
        """
        Find top-k multi-hop paths between source and target.

        Returns
        -------
        List[GraphPath], sorted by confidence descending.
        """
        source = source.strip().lower()
        target = target.strip().lower()

        if source not in self.graph or target not in self.graph:
            return []

        try:
            all_simple = list(
                nx.all_simple_paths(self.graph, source, target, cutoff=max_hops)
            )
        except nx.NetworkXNoPath:
            return []

        paths = []
        for node_path in all_simple:
            triples = []
            total_confidence = 1.0
            recency = 1.0

            for i in range(len(node_path) - 1):
                u, v = node_path[i], node_path[i + 1]
                edges = self.graph[u][v]
                best_edge = max(edges.values(), key=lambda e: e.get("weight", 0))
                relation = best_edge.get("relation", "related_to")
                conf = best_edge.get("confidence", 1.0)
                rec = best_edge.get("recency", 1.0)
                triples.append((u, relation, v))
                total_confidence *= conf
                recency = min(recency, rec)

            paths.append(GraphPath(
                path=triples,
                confidence=total_confidence,
                recency_score=recency,
            ))

        paths.sort(key=lambda p: p.confidence * p.recency_score, reverse=True)
        return paths[:top_k]

    def get_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """Get all neighbors of an entity up to given depth."""
        entity = entity.strip().lower()
        if entity not in self.graph:
            return []
        neighbors = set()
        frontier = {entity}
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(self.graph.successors(node))
                new_frontier.update(self.graph.predecessors(node))
            neighbors.update(new_frontier)
            frontier = new_frontier
        neighbors.discard(entity)
        return list(neighbors)

    def extract_subgraph(self, entities: List[str], max_hops: int = 2) -> "KnowledgeGraph":
        """Extract a minimal subgraph containing given entities."""
        subgraph = KnowledgeGraph()
        nodes = set(e.lower() for e in entities)

        for e in list(nodes):
            nodes.update(self.get_neighbors(e, depth=max_hops))

        sg = self.graph.subgraph(nodes & set(self.graph.nodes()))
        for u, v, data in sg.edges(data=True):
            subgraph.add_triple(
                u, data.get("relation", "related_to"), v,
                confidence=data.get("confidence", 1.0),
                source=data.get("source", "subgraph"),
                recency=data.get("recency", 1.0),
            )
        return subgraph

    def validate_relationship(self, subject: str, predicate: str, obj: str) -> float:
        """Return confidence if (subject, predicate, obj) exists, else 0."""
        s = subject.lower()
        o = obj.lower()
        p = predicate.lower()
        if not self.graph.has_edge(s, o):
            return 0.0
        for edge_data in self.graph[s][o].values():
            if edge_data.get("relation", "") == p:
                return edge_data.get("confidence", 0.0)
        return 0.0

    def entity_exists(self, entity: str) -> bool:
        return entity.strip().lower() in self.graph

    def stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }

    def save(self, path: str) -> None:
        """Persist graph to JSON."""
        data = nx.node_link_data(self.graph)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Graph saved to {path}")

    def load(self, path: str) -> None:
        """Load graph from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True, multigraph=True)
        logger.info(f"Graph loaded from {path} | {self.stats()}")
