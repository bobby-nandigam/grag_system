"""
RelationExtractor — Extract (subject, predicate, object) triples from raw text.

This is what lets free-form documents and chat messages automatically populate the
knowledge graph, instead of every fact having to be added by hand.

Two backends, selected automatically:
  * spaCy dependency parsing when the ``nlp`` extra is installed (more accurate).
  * A dependency-free regular-expression fallback otherwise.

Both backends are best-effort: rule-based extraction is inherently noisy, so every
triple carries a confidence score and the graph keeps each fact's provenance. The
class is designed to be subclassed — e.g. an LLM-backed extractor only needs to
override :meth:`extract`.

Example
-------
>>> rx = RelationExtractor()
>>> rx.extract("Python was created by Guido van Rossum.")
[Triple(subject='python', predicate='created_by', obj='guido van rossum', ...)]
"""

import re
import logging
from typing import List, Optional

from grag.core.models import Triple

logger = logging.getLogger(__name__)

# Authorship/creation verbs. These are canonicalized to a "<verb>_by" predicate
# pointing from the created thing to its agent, regardless of voice — so
# "Python was created by Guido" and "Guido created Python" both yield
# (python, created_by, guido).
_AUTHORSHIP_VERBS = {
    "create": "created_by",
    "develop": "developed_by",
    "design": "designed_by",
    "found": "founded_by",
    "invent": "invented_by",
    "write": "written_by",
    "author": "written_by",
    "build": "built_by",
    "introduce": "introduced_by",
    "pioneer": "pioneered_by",
    "discover": "discovered_by",
    "publish": "published_by",
}

# Direct relational verbs keep their subject -> object direction.
_DIRECT_VERBS = {
    "produce": "produces",
    "make": "makes",
    "own": "owns",
    "use": "uses",
    "support": "supports",
    "acquire": "acquired",
    "release": "released",
}

# Past-participle forms (including irregulars) mapped to canonical "_by" predicates,
# used by the regex fallback to match passive "... created by X" constructions.
_PARTICIPLE_TO_PRED = {
    "created": "created_by",
    "developed": "developed_by",
    "designed": "designed_by",
    "founded": "founded_by",
    "invented": "invented_by",
    "written": "written_by",
    "authored": "written_by",
    "built": "built_by",
    "made": "made_by",
    "introduced": "introduced_by",
    "pioneered": "pioneered_by",
    "discovered": "discovered_by",
    "published": "published_by",
}

# Determiners / leading filler stripped from extracted entity spans.
_LEADING_FILLER = {
    "a", "an", "the", "this", "that", "these", "those", "its", "their",
    "his", "her", "our", "your", "my", "some", "any", "such",
}

# Lowercase connector words allowed inside multi-word proper names (e.g. "van").
_NAME_CONNECTORS = {"van", "von", "de", "der", "di", "da", "del", "la", "le", "bin"}


class RelationExtractor:
    """
    Extract knowledge-graph triples from natural-language text.

    Parameters
    ----------
    min_confidence : float
        Triples scored below this threshold are discarded.
    use_spacy : bool
        If False, always use the regex fallback (useful for deterministic tests).
    """

    SPACY_CONFIDENCE = 0.75
    REGEX_CONFIDENCE = 0.6

    def __init__(self, min_confidence: float = 0.5, use_spacy: bool = True):
        self.min_confidence = min_confidence
        self._use_spacy = use_spacy
        self._nlp = None
        self._nlp_loaded = False

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(self, text: str) -> List[Triple]:
        """Extract triples from a block of text (may contain multiple sentences)."""
        if not text or not text.strip():
            return []

        nlp = self._get_nlp()
        if nlp is not None:
            triples = self._extract_spacy(text, nlp)
        else:
            triples = self._extract_regex(text)

        kept = [t for t in triples if t.confidence >= self.min_confidence]
        return self._dedupe(kept)

    def extract_many(self, texts: List[str]) -> List[Triple]:
        """Extract triples from many texts, de-duplicated across the batch."""
        all_triples: List[Triple] = []
        for text in texts:
            all_triples.extend(self.extract(text))
        return self._dedupe(all_triples)

    # ── spaCy backend ───────────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp_loaded:
            return self._nlp
        self._nlp_loaded = True
        if not self._use_spacy:
            self._nlp = None
            return None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("RelationExtractor: spaCy backend loaded (en_core_web_sm)")
        except Exception:
            logger.warning(
                "RelationExtractor: spaCy unavailable — using regex fallback. "
                "For better extraction: pip install grag-system[nlp] && "
                "python -m spacy download en_core_web_sm"
            )
            self._nlp = None
        return self._nlp

    def _extract_spacy(self, text: str, nlp) -> List[Triple]:
        triples: List[Triple] = []
        doc = nlp(text)
        # Map every token to the noun chunk that contains it, for clean spans.
        chunk_of = {}
        for chunk in doc.noun_chunks:
            for tok in chunk:
                chunk_of[tok.i] = chunk

        for sent in doc.sents:
            for token in sent:
                if token.pos_ not in ("VERB", "AUX"):
                    continue
                triples.extend(self._triples_from_verb(token, chunk_of))
        return triples

    def _triples_from_verb(self, verb, chunk_of) -> List[Triple]:
        out: List[Triple] = []
        lemma = verb.lemma_.lower()

        subj = [c for c in verb.children if c.dep_ == "nsubj"]
        subj_pass = [c for c in verb.children if c.dep_ == "nsubjpass"]
        dobj = [c for c in verb.children if c.dep_ in ("dobj", "obj")]
        attrs = [c for c in verb.children if c.dep_ in ("attr", "acomp")]
        agents = []
        for c in verb.children:
            if c.dep_ == "agent":  # the word "by" in a passive clause
                agents.extend([g for g in c.children if g.dep_ == "pobj"])

        # Passive authorship: "X was created by Y" -> (X, created_by, Y)
        if lemma in _AUTHORSHIP_VERBS and subj_pass and agents:
            pred = _AUTHORSHIP_VERBS[lemma]
            thing = self._phrase(subj_pass[0], chunk_of)
            for ag in agents:
                out.append(self._mk(thing, pred, self._phrase(ag, chunk_of)))
            return out

        # Active authorship: "Y created X" -> (X, created_by, Y)
        if lemma in _AUTHORSHIP_VERBS and subj and dobj:
            pred = _AUTHORSHIP_VERBS[lemma]
            agent = self._phrase(subj[0], chunk_of)
            for o in dobj:
                out.append(self._mk(self._phrase(o, chunk_of), pred, agent))
            return out

        # Direct relation: "X produces Y" -> (X, produces, Y)
        if lemma in _DIRECT_VERBS and subj and dobj:
            pred = _DIRECT_VERBS[lemma]
            s = self._phrase(subj[0], chunk_of)
            for o in dobj:
                out.append(self._mk(s, pred, self._phrase(o, chunk_of)))
            return out

        # Copular: "X is a Y" -> (X, is_a, Y)
        if lemma == "be" and subj and attrs:
            s = self._phrase(subj[0], chunk_of)
            for a in attrs:
                if a.pos_ in ("NOUN", "PROPN"):
                    out.append(self._mk(s, "is_a", self._phrase(a, chunk_of)))
            return out

        # Prepositional relations: "X works at Y", "X is located in Y".
        if subj:
            s = self._phrase(subj[0], chunk_of)
            for prep in (c for c in verb.children if c.dep_ == "prep"):
                pobjs = [g for g in prep.children if g.dep_ == "pobj"]
                pred = self._prep_predicate(lemma, prep.text.lower())
                if pred:
                    for o in pobjs:
                        out.append(self._mk(s, pred, self._phrase(o, chunk_of)))
        return out

    @staticmethod
    def _prep_predicate(lemma: str, prep: str) -> Optional[str]:
        if lemma == "work" and prep in ("at", "for"):
            return "works_at"
        if lemma in ("locate", "base") and prep == "in":
            return "located_in"
        if lemma == "be" and prep == "in":
            return "located_in"
        return None

    def _phrase(self, token, chunk_of) -> str:
        """Return a clean lowercased entity span for a token."""
        chunk = chunk_of.get(token.i)
        text = chunk.text if chunk is not None else token.text
        return self._clean(text)

    def _mk(self, subject: str, predicate: str, obj: str) -> Triple:
        return Triple(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=self.SPACY_CONFIDENCE,
            source="extracted:spacy",
        )

    # ── Regex fallback ──────────────────────────────────────────────────────

    def _extract_regex(self, text: str) -> List[Triple]:
        triples: List[Triple] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip()
            if sentence:
                triples.extend(self._regex_sentence(sentence))
        return triples

    def _regex_sentence(self, sentence: str) -> List[Triple]:
        out: List[Triple] = []
        subject = self._leading_entity(sentence)
        if not subject:
            return out

        # Passive authorship: "... created/founded/developed by <agent(s)>".
        # The participle need not follow the subject directly (it may sit inside a
        # reduced relative clause, e.g. "Python is a language created by Guido").
        participles = "|".join(_PARTICIPLE_TO_PRED)
        for m in re.finditer(rf"\b({participles})\s+by\s+(.+)", sentence, re.IGNORECASE):
            pred = _PARTICIPLE_TO_PRED[m.group(1).lower()]
            for agent in self._split_entities(m.group(2)):
                out.append(self._mk_regex(subject, pred, agent))

        # Founding/establishment date: "... founded/established in <year>".
        m = re.search(r"\b(?:founded|established|released|born)\s+in\s+((?:19|20)\d{2})",
                      sentence, re.IGNORECASE)
        if m:
            out.append(self._mk_regex(subject, "founded_in", m.group(1)))

        # Direct relation: "<subject> produces/makes/uses/... <object>".
        direct = "|".join(v + r"s?" for v in _DIRECT_VERBS)
        m = re.search(rf"\b({direct})\b\s+(.+)", sentence, re.IGNORECASE)
        if m:
            verb_lemma = re.sub(r"s$", "", m.group(1).lower())
            pred = _DIRECT_VERBS.get(verb_lemma, verb_lemma)
            for obj in self._split_entities(m.group(2)):
                out.append(self._mk_regex(subject, pred, obj))

        # Prepositional: "<subject> works at/for <object>".
        m = re.search(r"\bworks?\s+(?:at|for)\s+(.+)", sentence, re.IGNORECASE)
        if m:
            for obj in self._split_entities(m.group(1)):
                out.append(self._mk_regex(subject, "works_at", obj))

        # Copular type: "<subject> is a/an <type>" (object trimmed before any clause).
        m = re.search(r"\b(?:is|was|are|were)\s+(?:a|an)\s+(.+)", sentence, re.IGNORECASE)
        if m:
            obj = self._trim_object(m.group(1))
            if obj:
                out.append(self._mk_regex(subject, "is_a", obj))

        return out

    def _mk_regex(self, subject: str, predicate: str, obj: str) -> Triple:
        return Triple(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=self.REGEX_CONFIDENCE,
            source="extracted:regex",
        )

    @staticmethod
    def _leading_entity(fragment: str) -> str:
        """
        Take the leading proper-name run from a fragment (allowing connectors like
        "van"), else fall back to the first token for lowercase/casual text.
        """
        fragment = fragment.strip()
        if not fragment:
            return ""
        tokens = fragment.split()
        if tokens and tokens[0][:1].isupper():
            name = [tokens[0]]
            for tok in tokens[1:]:
                if tok[:1].isupper() or tok.lower() in _NAME_CONNECTORS:
                    name.append(tok)
                else:
                    break
            # Drop a trailing connector (e.g. "Page and" -> "Page").
            while name and name[-1].lower() in _NAME_CONNECTORS:
                name.pop()
            return RelationExtractor._clean(" ".join(name))
        # Lowercase text (e.g. casual chat): fall back to the first token.
        return RelationExtractor._clean(tokens[0])

    def _split_entities(self, fragment: str) -> List[str]:
        """Split an object span on 'and'/commas, then take each leading entity.

        ``_leading_entity`` already stops at a trailing lowercase word (e.g. the
        "in" of "... in 1991"), so no separate trimming is needed here.
        """
        fragment = fragment.strip().rstrip(".!?")
        parts = re.split(r"\s+and\s+|\s*,\s*", fragment)
        entities = []
        for part in parts:
            ent = self._leading_entity(part)
            if ent:
                entities.append(ent)
        return entities

    @staticmethod
    def _trim_object(fragment: str) -> str:
        """Cut an object span at a trailing preposition, relative clause, or year."""
        fragment = fragment.strip().rstrip(".!?")
        # Stop before a reduced relative clause: "... language created by ...".
        participles = "|".join(_PARTICIPLE_TO_PRED)
        fragment = re.split(rf"\s+\b(?:{participles})\b\s+", fragment, maxsplit=1)[0]
        fragment = re.split(
            r"\s+\b(?:in|on|at|since|during|that|which|who|when|while|and|by)\b\s+",
            fragment, maxsplit=1,
        )[0]
        fragment = re.sub(r"\b(?:19|20)\d{2}\b.*$", "", fragment).strip()
        return RelationExtractor._clean(fragment)

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[^\w\s.&'-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        words = text.split()
        while words and words[0] in _LEADING_FILLER:
            words.pop(0)
        return " ".join(words)

    @staticmethod
    def _dedupe(triples: List[Triple]) -> List[Triple]:
        best = {}
        for t in triples:
            if not t.subject or not t.obj or t.subject == t.obj:
                continue
            key = (t.subject, t.predicate, t.obj)
            if key not in best or t.confidence > best[key].confidence:
                best[key] = t
        return list(best.values())
