"""
QueryUnderstanding — Parses raw queries into structured QueryParsed objects.

Uses spaCy NER when available, with a regex fallback.
"""

import re
import logging
from typing import List, Dict, Any

from grag.core.models import QueryParsed

logger = logging.getLogger(__name__)

# Common English stop words to filter from entity extraction
_STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "in", "of", "for", "to", "and", "or", "but", "on",
    "with", "by", "at", "from", "as", "into", "that", "this", "it", "he",
    "she", "they", "we", "i", "you", "has", "have", "had", "does", "do",
    "did", "can", "could", "will", "would", "should", "tell", "me", "give",
    "list", "show", "describe", "explain", "find", "search",
}

# Intent keyword mapping
_INTENT_PATTERNS = {
    "definition": r"\b(what is|define|meaning of|explain)\b",
    "comparison": r"\b(compare|difference|vs|versus|better|worse)\b",
    "causal": r"\b(why|cause|reason|because|leads to|result)\b",
    "temporal": r"\b(when|history|timeline|year|date|since|until)\b",
    "entity_info": r"\b(who is|tell me about|describe|information on)\b",
    "relationship": r"\b(how (is|are|does)|relationship|connected|related)\b",
    "listing": r"\b(list|examples|types of|kinds of|what are)\b",
    "factual": r"\b(how many|how much|what (number|amount|percentage))\b",
}

# Relationship keywords
_RELATIONSHIP_KEYWORDS = [
    "created_by", "developed_by", "founded_by", "used_in",
    "part_of", "belongs_to", "related_to", "works_for",
    "located_in", "successor_of", "predecessor_of",
]


class QueryUnderstanding:
    """
    Transforms a raw user query into a structured QueryParsed object.

    Steps
    -----
    1. Detect intent from keyword patterns
    2. Extract named entities (spaCy or regex fallback)
    3. Identify relationship keywords
    4. Extract constraints (time, domain)
    5. Generate semantic + graph query variants
    6. Score ambiguity

    Example
    -------
    >>> qu = QueryUnderstanding()
    >>> parsed = qu.parse("Who created Python and where does Guido work?")
    >>> parsed.intent
    'entity_info'
    >>> parsed.entities
    ['Python', 'Guido']
    """

    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NER loaded (en_core_web_sm)")
            except Exception:
                logger.warning(
                    "spaCy not available — using regex entity extraction. "
                    "For better NER: pip install grag[nlp] && python -m spacy download en_core_web_sm"
                )
                self._nlp = None
        return self._nlp

    def parse(self, query: str) -> QueryParsed:
        """Parse a raw query string into a structured QueryParsed object."""
        query = query.strip()
        intent = self._detect_intent(query)
        entities = self._extract_entities(query)
        relationships = self._extract_relationships(query)
        constraints = self._extract_constraints(query)
        semantic_query = self._build_semantic_query(query, intent, entities)
        graph_query = self._build_graph_query(entities, relationships)
        ambiguity = self._score_ambiguity(query, entities, intent)

        parsed = QueryParsed(
            raw_query=query,
            intent=intent,
            entities=entities,
            relationships=relationships,
            constraints=constraints,
            semantic_query=semantic_query,
            graph_query=graph_query,
            ambiguity_score=ambiguity,
        )

        if ambiguity > 0.7:
            logger.warning(f"High ambiguity ({ambiguity:.2f}) in query: '{query[:60]}'")

        logger.debug(f"Parsed query: intent={intent}, entities={entities}, ambiguity={ambiguity:.2f}")
        return parsed

    def _detect_intent(self, query: str) -> str:
        q_lower = query.lower()
        for intent, pattern in _INTENT_PATTERNS.items():
            if re.search(pattern, q_lower):
                return intent
        return "general"

    def _extract_entities(self, query: str) -> List[str]:
        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(query)
            return list({ent.text for ent in doc.ents if len(ent.text) > 1})

        # Regex fallback: capitalized words / quoted phrases
        entities = []
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for q in quoted:
            entities.append((q[0] or q[1]).strip())

        words = query.split()
        for word in words:
            clean = re.sub(r"[^a-zA-Z0-9_\-]", "", word)
            if (
                clean
                and clean[0].isupper()
                and clean.lower() not in _STOPWORDS
                and len(clean) > 1
            ):
                entities.append(clean)

        return list(dict.fromkeys(entities))  # deduplicate while preserving order

    def _extract_relationships(self, query: str) -> List[str]:
        q_lower = query.lower()
        found = []
        for rel in _RELATIONSHIP_KEYWORDS:
            keyword = rel.replace("_", " ")
            if keyword in q_lower or rel in q_lower:
                found.append(rel)

        # Extract verb phrases like "works at", "created by"
        verb_patterns = [
            r"\b(created|developed|founded|built|invented)\s+by\b",
            r"\b(works?\s+(?:at|for|with))\b",
            r"\b(part\s+of|belongs?\s+to)\b",
            r"\b(located\s+in|based\s+in)\b",
        ]
        for pat in verb_patterns:
            m = re.search(pat, q_lower)
            if m:
                found.append(m.group().replace(" ", "_"))

        return list(set(found))

    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        q_lower = query.lower()

        # Time constraints
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        if year_match:
            constraints["year"] = int(year_match.group())

        for kw in ["recent", "latest", "current", "today", "now"]:
            if kw in q_lower:
                constraints["recency"] = "recent"
                break

        # Domain hints
        domain_keywords = {
            "ml": ["machine learning", "deep learning", "neural", "model", "training"],
            "nlp": ["language model", "nlp", "text", "tokenizer", "bert", "gpt"],
            "cv": ["image", "vision", "cnn", "convolutional", "object detection"],
            "software": ["python", "java", "code", "library", "framework", "api"],
            "science": ["research", "paper", "experiment", "hypothesis", "study"],
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in q_lower for kw in keywords):
                constraints["domain"] = domain
                break

        return constraints

    def _build_semantic_query(self, query: str, intent: str, entities: List[str]) -> str:
        """Augment the query for better embedding similarity."""
        if intent == "definition":
            return f"definition explanation of {' '.join(entities)} — {query}"
        elif intent == "comparison":
            return f"comparison between {' vs '.join(entities)} — {query}"
        elif intent == "causal":
            return f"causal relationship reason why {' '.join(entities)} — {query}"
        return query

    def _build_graph_query(self, entities: List[str], relationships: List[str]) -> str:
        """Build a structured graph traversal query string."""
        if not entities:
            return ""
        if relationships:
            return f"MATCH ({entities[0]})-[:{relationships[0].upper()}]->(?)"
        return f"MATCH ({entities[0]})-[*]->(?) WHERE entities IN {entities}"

    def _score_ambiguity(self, query: str, entities: List[str], intent: str) -> float:
        """
        Score ambiguity of the query.
        Higher = more ambiguous.
        """
        score = 0.0
        if intent == "general":
            score += 0.4
        if len(entities) == 0:
            score += 0.3
        if len(query.split()) < 3:
            score += 0.2
        if "?" not in query and intent not in ("listing", "factual"):
            score += 0.1
        return min(score, 1.0)
