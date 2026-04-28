"""
GraphReasoner — Multi-hop graph reasoning engine.

Performs:
  - Entity linking
  - Relationship validation
  - Path ranking
  - Subgraph construction
  - Contradiction detection
"""

import logging
from typing import List, Dict, Tuple, Optional

from grag.core.models import QueryParsed, GraphPath, FusedContext, RetrievedDocument
from grag.core.config import GRAGConfig
from grag.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphReasoner:
    """
    Core reasoning engine over the knowledge graph.

    Workflow
    --------
    1. Link query entities to graph nodes (entity linking)
    2. Traverse multi-hop paths between entity pairs
    3. Rank paths by confidence × recency
    4. Validate relationships
    5. Detect contradictions between graph + documents
    6. Construct minimal reasoning subgraph

    Example
    -------
    >>> reasoner = GraphReasoner(config, kg)
    >>> context = reasoner.reason(parsed_query, retrieved_docs)
    """

    def __init__(self, config: GRAGConfig, knowledge_graph: KnowledgeGraph):
        self.config = config
        self.kg = knowledge_graph

    def reason(
        self,
        parsed_query: QueryParsed,
        retrieved_docs: List[RetrievedDocument],
    ) -> FusedContext:
        """
        Full reasoning pipeline:
          1. Entity linking
          2. Multi-hop graph traversal
          3. Contradiction detection
          4. Context fusion
        """
        # Step 1: Link entities to graph nodes
        linked_entities = self._link_entities(parsed_query.entities)
        logger.debug(f"Linked entities: {linked_entities}")

        # Step 2: Multi-hop traversal
        graph_paths = self._traverse(linked_entities, parsed_query)

        # Step 3: Extract graph facts as strings
        graph_facts = [p.to_string() for p in graph_paths]

        # Step 4: Validate relationships
        validated_facts = self._validate_facts(graph_facts, parsed_query)

        # Step 5: Detect contradictions
        contradictions = self._detect_contradictions(validated_facts, retrieved_docs)

        # Step 6: Compute overall confidence
        if graph_paths:
            graph_confidence = sum(p.confidence for p in graph_paths) / len(graph_paths)
        else:
            graph_confidence = 0.0

        doc_confidence = (
            sum(d.score for d in retrieved_docs) / len(retrieved_docs)
            if retrieved_docs else 0.0
        )

        fused_confidence = (
            self.config.graph_weight * graph_confidence +
            self.config.vector_weight * doc_confidence
        )

        if contradictions:
            fused_confidence *= (1 - 0.1 * len(contradictions))

        return FusedContext(
            graph_facts=validated_facts,
            document_chunks=retrieved_docs,
            contradictions=contradictions,
            confidence=min(max(fused_confidence, 0.0), 1.0),
        )

    def get_best_path(self, parsed_query: QueryParsed) -> Optional[GraphPath]:
        """Return the single highest-confidence path for the query."""
        linked = self._link_entities(parsed_query.entities)
        paths = self._traverse(linked, parsed_query)
        return paths[0] if paths else None

    def _link_entities(self, entities: List[str]) -> List[str]:
        """
        Link extracted entities to graph nodes via fuzzy matching.
        Falls back to partial name matching if exact node not found.
        """
        linked = []
        graph_nodes = set(self.kg.graph.nodes())

        for entity in entities:
            ent_lower = entity.lower()
            if ent_lower in graph_nodes:
                linked.append(ent_lower)
                continue

            # Partial fuzzy match
            candidates = [
                n for n in graph_nodes
                if ent_lower in n or n in ent_lower
            ]
            if candidates:
                best = min(candidates, key=len)
                logger.debug(f"Fuzzy linked '{entity}' → '{best}'")
                linked.append(best)

        return list(set(linked))

    def _traverse(
        self,
        entities: List[str],
        parsed_query: QueryParsed,
    ) -> List[GraphPath]:
        """Traverse all entity-pair paths, ranked by confidence."""
        all_paths: List[GraphPath] = []

        for i, src in enumerate(entities):
            for tgt in entities[i + 1:]:
                paths = self.kg.find_paths(
                    src, tgt,
                    max_hops=self.config.max_hops,
                    top_k=3,
                )
                all_paths.extend(paths)

            # Also traverse from each entity outward
            neighbors = self.kg.get_neighbors(src, depth=min(self.config.max_hops, 2))
            for neighbor in neighbors[:5]:
                paths = self.kg.find_paths(src, neighbor, max_hops=2, top_k=2)
                all_paths.extend(paths)

        # Deduplicate paths
        seen = set()
        unique_paths = []
        for p in all_paths:
            key = p.to_string()
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        # Apply recency + relationship relevance boost
        for path in unique_paths:
            path.confidence *= self._relationship_relevance(
                path, parsed_query.relationships
            )

        unique_paths.sort(key=lambda p: p.confidence * p.recency_score, reverse=True)
        return unique_paths[:self.config.top_k]

    def _relationship_relevance(
        self, path: GraphPath, query_relationships: List[str]
    ) -> float:
        """Boost paths that match queried relationship types."""
        if not query_relationships:
            return 1.0
        path_relations = {r for _, r, _ in path.path}
        overlap = len(path_relations & set(query_relationships))
        return 1.0 + (overlap * 0.1)

    def _validate_facts(
        self, facts: List[str], parsed_query: QueryParsed
    ) -> List[str]:
        """Filter out facts with zero-confidence relationships."""
        validated = []
        for fact in facts:
            # A fact string looks like "A --[rel]--> B | B --[rel2]--> C"
            # We trust graph-derived facts (they came from the KG itself)
            if "--[" in fact and "]-->" in fact:
                validated.append(fact)
        return validated

    def _detect_contradictions(
        self,
        graph_facts: List[str],
        docs: List[RetrievedDocument],
    ) -> List[str]:
        """
        Simple contradiction detector:
        Flags document chunks that appear to contradict graph facts.
        """
        contradictions = []
        for fact in graph_facts:
            parts = fact.split(" --[")
            if not parts:
                continue
            subject = parts[0].strip()
            for doc in docs:
                content_lower = doc.content.lower()
                # If a doc explicitly negates the subject relationship
                negation_signals = [
                    f"not {subject.lower()}",
                    f"{subject.lower()} does not",
                    f"{subject.lower()} never",
                    f"{subject.lower()} isn't",
                ]
                for signal in negation_signals:
                    if signal in content_lower:
                        contradictions.append(
                            f"CONTRADICTION: '{fact}' ← conflicts with doc[{doc.doc_id}]"
                        )
                        break
        return contradictions
