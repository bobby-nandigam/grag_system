"""
GRAG Test Suite — Unit + integration tests.

Run with: pytest tests/ -v --cov=grag
"""

import pytest
from unittest.mock import patch, MagicMock

from grag.core.config import GRAGConfig
from grag.core.models import (
    QueryParsed, GRAGAnswer, GraphPath, FusedContext,
    EvaluationResult, FailureType, RetrievedDocument,
)
from grag.graph.knowledge_graph import KnowledgeGraph
from grag.retrieval.query_understanding import QueryUnderstanding
from grag.retrieval.hybrid_retriever import HybridRetriever, VectorStore
from grag.reasoning.graph_reasoner import GraphReasoner
from grag.evaluation.critic import CriticModule
from grag.rl.reward_engine import RewardEngine
from grag.memory.memory_store import MemoryStore
from grag.core.pipeline import GRAGPipeline


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return GRAGConfig(top_k=3, max_hops=2, max_refinement_iterations=2)


@pytest.fixture
def kg():
    g = KnowledgeGraph()
    g.add_triple("python", "created_by", "guido van rossum", confidence=0.99)
    g.add_triple("guido van rossum", "works_at", "google", confidence=0.95)
    g.add_triple("nvidia", "produces", "gpus", confidence=0.99)
    g.add_triple("cuda", "developed_by", "nvidia", confidence=0.99)
    g.add_triple("pytorch", "created_by", "meta ai", confidence=0.99)
    return g


@pytest.fixture
def pipeline(config, kg):
    p = GRAGPipeline(config=config, knowledge_graph=kg)
    p.add_documents([
        {"content": "Python was created by Guido van Rossum in 1991.", "source": "wiki"},
        {"content": "NVIDIA designs GPUs and CUDA for parallel computing.", "source": "wiki"},
        {"content": "PyTorch is a deep learning framework by Meta AI.", "source": "wiki"},
        {"content": "Guido van Rossum works at Google.", "source": "wiki"},
    ])
    return p


# ─── KnowledgeGraph Tests ─────────────────────────────────────────────────────

class TestKnowledgeGraph:

    def test_add_triple(self, kg):
        assert kg.entity_exists("python")
        assert kg.entity_exists("guido van rossum")

    def test_find_paths_direct(self, kg):
        paths = kg.find_paths("python", "guido van rossum", max_hops=1)
        assert len(paths) > 0
        assert paths[0].confidence > 0

    def test_find_paths_multihop(self, kg):
        paths = kg.find_paths("python", "google", max_hops=2)
        assert len(paths) > 0
        assert "google" in paths[0].to_string()

    def test_get_neighbors(self, kg):
        neighbors = kg.get_neighbors("nvidia", depth=1)
        assert "gpus" in neighbors or "cuda" in neighbors

    def test_validate_relationship(self, kg):
        conf = kg.validate_relationship("python", "created_by", "guido van rossum")
        assert conf > 0.9

    def test_validate_relationship_missing(self, kg):
        conf = kg.validate_relationship("python", "created_by", "elon musk")
        assert conf == 0.0

    def test_stats(self, kg):
        stats = kg.stats()
        assert stats["nodes"] >= 4
        assert stats["edges"] >= 4

    def test_save_load(self, kg, tmp_path):
        save_path = str(tmp_path / "test_graph.json")
        kg.save(save_path)
        kg2 = KnowledgeGraph()
        kg2.load(save_path)
        assert kg2.entity_exists("python")

    def test_graph_path_string(self):
        path = GraphPath(path=[("python", "created_by", "guido")], confidence=0.9)
        s = path.to_string()
        assert "python" in s
        assert "created_by" in s


# ─── QueryUnderstanding Tests ─────────────────────────────────────────────────

class TestQueryUnderstanding:

    def setup_method(self):
        self.qu = QueryUnderstanding()

    def test_parse_basic(self):
        parsed = self.qu.parse("Who created Python?")
        assert parsed.intent in ("entity_info", "general", "definition")
        assert "Python" in parsed.entities or "python" in [e.lower() for e in parsed.entities]

    def test_intent_definition(self):
        parsed = self.qu.parse("What is machine learning?")
        assert parsed.intent == "definition"

    def test_intent_comparison(self):
        parsed = self.qu.parse("Compare PyTorch vs TensorFlow")
        assert parsed.intent == "comparison"

    def test_intent_causal(self):
        parsed = self.qu.parse("Why is Python popular?")
        assert parsed.intent == "causal"

    def test_constraints_year(self):
        parsed = self.qu.parse("What happened in AI in 2023?")
        assert parsed.constraints.get("year") == 2023

    def test_constraints_recency(self):
        parsed = self.qu.parse("What are the latest GPU models?")
        assert parsed.constraints.get("recency") == "recent"

    def test_ambiguity_low(self):
        parsed = self.qu.parse("Who created Python and what does Guido van Rossum do at Google?")
        assert parsed.ambiguity_score < 0.7

    def test_ambiguity_high(self):
        parsed = self.qu.parse("stuff?")
        assert parsed.ambiguity_score > 0.3

    def test_graph_query_generated(self):
        parsed = self.qu.parse("What did NVIDIA create?")
        assert isinstance(parsed.graph_query, str)

    def test_semantic_query_enriched(self):
        parsed = self.qu.parse("What is deep learning?")
        assert len(parsed.semantic_query) >= len("What is deep learning?")


# ─── VectorStore Tests ────────────────────────────────────────────────────────

class TestVectorStore:

    def test_add_and_search(self):
        vs = VectorStore()
        vs.add_documents([
            {"content": "Python is a programming language.", "source": "wiki"},
            {"content": "NVIDIA makes GPUs.", "source": "wiki"},
        ])
        results = vs.search("programming language", top_k=2)
        assert len(results) > 0
        assert results[0].content != ""

    def test_empty_search(self):
        vs = VectorStore()
        results = vs.search("anything", top_k=3)
        assert results == []

    def test_score_range(self):
        vs = VectorStore()
        vs.add_documents([{"content": "Deep learning uses neural networks.", "source": "a"}])
        results = vs.search("neural networks", top_k=1)
        assert 0.0 <= results[0].score <= 1.0


# ─── HybridRetriever Tests ────────────────────────────────────────────────────

class TestHybridRetriever:

    def test_retrieve_returns_docs(self, config, kg):
        retriever = HybridRetriever(config, kg)
        retriever.add_documents([
            {"content": "Python is a language by Guido van Rossum.", "source": "wiki"},
        ])
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("Who created Python?")
        docs = retriever.retrieve(parsed)
        assert isinstance(docs, list)

    def test_graph_boost_works(self, config, kg):
        retriever = HybridRetriever(config, kg)
        retriever.add_documents([
            {"content": "nvidia cuda gpu computing parallel", "source": "nvidia"},
            {"content": "unrelated content about cooking pasta.", "source": "food"},
        ])
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("What does NVIDIA produce?")
        docs = retriever.retrieve(parsed)
        # nvidia-related doc should score higher
        assert any("nvidia" in d.content.lower() or "cuda" in d.content.lower() for d in docs[:2])


# ─── GraphReasoner Tests ──────────────────────────────────────────────────────

class TestGraphReasoner:

    def test_reason_returns_context(self, config, kg):
        reasoner = GraphReasoner(config, kg)
        qu = QueryUnderstanding()
        parsed = qu.parse("Who created Python?")
        docs = [RetrievedDocument("d1", "Python by Guido.", 0.9, "wiki")]
        ctx = reasoner.reason(parsed, docs)
        assert isinstance(ctx, FusedContext)
        assert 0.0 <= ctx.confidence <= 1.0

    def test_get_best_path(self, config, kg):
        reasoner = GraphReasoner(config, kg)
        qu = QueryUnderstanding()
        parsed = qu.parse("Where does Guido van Rossum work?")
        path = reasoner.get_best_path(parsed)
        # May or may not find a path depending on entity linking
        assert path is None or isinstance(path, GraphPath)

    def test_contradiction_detection(self, config, kg):
        reasoner = GraphReasoner(config, kg)
        qu = QueryUnderstanding()
        parsed = qu.parse("Who created CUDA?")
        docs = [
            RetrievedDocument("d1", "nvidia does not produce GPUs.", 0.5, "test"),
        ]
        ctx = reasoner.reason(parsed, docs)
        assert isinstance(ctx.contradictions, list)


# ─── CriticModule Tests ───────────────────────────────────────────────────────

class TestCriticModule:

    def test_faithfulness_high(self, config):
        critic = CriticModule(config)
        answer = GRAGAnswer(
            answer="Python was created by Guido van Rossum.",
            graph_path=None,
            document_summary="",
            confidence=0.9,
            entities_used=["Python", "Guido van Rossum"],
        )
        ctx = FusedContext(
            graph_facts=["python --[created_by]--> guido van rossum"],
            document_chunks=[
                RetrievedDocument("d1", "Python was created by Guido van Rossum.", 0.9, "wiki")
            ],
            contradictions=[],
            confidence=0.9,
        )
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("Who created Python?")
        result = critic.evaluate(answer, parsed, ctx)
        assert result.faithfulness >= 0.5

    def test_eval_passed(self, config):
        critic = CriticModule(config)
        answer = GRAGAnswer(
            answer="NVIDIA produces GPUs and developed CUDA for parallel computing.",
            graph_path=None,
            document_summary="",
            confidence=0.85,
            entities_used=["NVIDIA", "GPUs", "CUDA"],
        )
        ctx = FusedContext(
            graph_facts=["nvidia --[produces]--> gpus", "cuda --[developed_by]--> nvidia"],
            document_chunks=[
                RetrievedDocument("d1", "NVIDIA designs GPUs and CUDA.", 0.9, "wiki")
            ],
            contradictions=[],
            confidence=0.85,
        )
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("What does NVIDIA produce?")
        result = critic.evaluate(answer, parsed, ctx)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.overall_score <= 1.0


# ─── RewardEngine Tests ───────────────────────────────────────────────────────

class TestRewardEngine:

    def test_record_reward(self, config):
        engine = RewardEngine(config)
        answer = GRAGAnswer("Test answer.", None, "", 0.8, ["entity"])
        eval_result = EvaluationResult(0.9, 0.8, 0.85, 0.9, 0.86)
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("Test query?")
        reward = engine.record(answer, eval_result, parsed)
        assert isinstance(reward, float)

    def test_adapted_config_returned(self, config):
        engine = RewardEngine(config)
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("What is Python?")
        adapted = engine.get_adapted_config(parsed)
        assert "top_k" in adapted
        assert "max_hops" in adapted

    def test_hallucination_penalized(self, config):
        engine = RewardEngine(config)
        answer = GRAGAnswer("Fabricated answer.", None, "", 0.3, [])
        eval_result = EvaluationResult(
            0.1, 0.2, 0.1, 0.1, 0.13,
            failure_type=FailureType.HALLUCINATION
        )
        from grag.retrieval.query_understanding import QueryUnderstanding
        parsed = QueryUnderstanding().parse("What is X?")
        reward = engine.record(answer, eval_result, parsed)
        assert reward < 0.5

    def test_stats(self, config):
        engine = RewardEngine(config)
        stats = engine.stats()
        assert "total_queries" in stats
        assert "average_reward" in stats


# ─── MemoryStore Tests ────────────────────────────────────────────────────────

class TestMemoryStore:

    def test_store_and_retrieve(self):
        mem = MemoryStore()
        mem.store("Who created Python?", "entity_info", "Guido van Rossum", "python->guido", 0.9, 0.8)
        results = mem.retrieve_similar("Who created Python?")
        assert len(results) > 0
        assert "Guido" in results[0].answer_summary

    def test_failure_pattern(self):
        mem = MemoryStore()
        mem.store("Who made XYZ?", "entity_info", "Unknown", "", 0.1, -0.5)
        assert mem.is_failure_pattern("Who made XYZ entity?")

    def test_stats(self):
        mem = MemoryStore()
        stats = mem.get_stats()
        assert "total_memories" in stats

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = MemoryStore(path=path)
        mem.store("Query 1", "general", "Answer 1", "A->B", 0.9, 0.8)
        mem2 = MemoryStore(path=path)
        assert len(mem2._entries) == 1


# ─── Full Pipeline Integration Tests ─────────────────────────────────────────

class TestGRAGPipeline:

    def test_basic_query(self, pipeline):
        result = pipeline.query("Who created Python?")
        assert isinstance(result, GRAGAnswer)
        assert len(result.answer) > 10
        assert 0.0 <= result.confidence <= 1.0

    def test_answer_contains_entities(self, pipeline):
        result = pipeline.query("What does NVIDIA produce?")
        answer_lower = result.answer.lower()
        assert "nvidia" in answer_lower or "gpu" in answer_lower or "cuda" in answer_lower

    def test_insufficient_evidence(self, pipeline):
        result = pipeline.query("What is the secret formula of Coca-Cola?")
        # Should not hallucinate — may return low confidence or insufficient evidence
        assert result.confidence <= 1.0
        assert isinstance(result.answer, str)

    def test_iterations_tracked(self, pipeline):
        result = pipeline.query("Who created PyTorch?")
        assert result.iterations >= 1

    def test_stats_populated(self, pipeline):
        pipeline.query("Test query")
        stats = pipeline.stats()
        assert stats["rl"]["total_queries"] >= 1
        assert stats["kg"]["nodes"] > 0

    def test_graph_path_explainability(self, pipeline):
        result = pipeline.query("Where does Guido van Rossum work?")
        # graph_path can be None or GraphPath
        assert result.graph_path is None or isinstance(result.graph_path, GraphPath)

    def test_document_summary_populated(self, pipeline):
        result = pipeline.query("What is CUDA?")
        assert isinstance(result.document_summary, str)

    def test_config_fast(self):
        config = GRAGConfig.fast()
        assert config.top_k == 3
        assert config.max_refinement_iterations == 1

    def test_config_production(self):
        config = GRAGConfig.production()
        assert config.top_k == 10
        assert config.confidence_threshold >= 0.8


# ─── RelationExtractor Tests ──────────────────────────────────────────────────

from grag.extraction.relation_extractor import RelationExtractor


class TestRelationExtractor:

    @pytest.fixture
    def rx(self):
        # Force the deterministic regex backend so tests don't depend on spaCy.
        return RelationExtractor(use_spacy=False)

    def _triples(self, rx, text):
        return {(t.subject, t.predicate, t.obj) for t in rx.extract(text)}

    def test_passive_authorship(self, rx):
        triples = self._triples(rx, "Python is a language created by Guido van Rossum.")
        assert ("python", "created_by", "guido van rossum") in triples

    def test_multiple_agents_split(self, rx):
        triples = self._triples(rx, "Google was founded by Larry Page and Sergey Brin.")
        assert ("google", "founded_by", "larry page") in triples
        assert ("google", "founded_by", "sergey brin") in triples

    def test_irregular_participle(self, rx):
        triples = self._triples(rx, "PyTorch is a framework developed by Meta AI.")
        assert ("pytorch", "developed_by", "meta ai") in triples

    def test_direct_relation(self, rx):
        triples = self._triples(rx, "NVIDIA produces GPUs and CUDA.")
        assert ("nvidia", "produces", "gpus") in triples
        assert ("nvidia", "produces", "cuda") in triples

    def test_works_at(self, rx):
        triples = self._triples(rx, "Guido van Rossum works at Google.")
        assert ("guido van rossum", "works_at", "google") in triples

    def test_founded_in_year(self, rx):
        triples = self._triples(rx, "NVIDIA was founded in 1993.")
        assert ("nvidia", "founded_in", "1993") in triples

    def test_is_a_object_trimmed_before_clause(self, rx):
        # The "created by ..." clause must not leak into the is_a object.
        triples = self._triples(rx, "Python is a programming language created by Guido.")
        is_a = {o for s, p, o in triples if p == "is_a"}
        assert "programming language" in is_a

    def test_empty_text(self, rx):
        assert rx.extract("") == []
        assert rx.extract("   ") == []

    def test_min_confidence_filter(self):
        rx = RelationExtractor(use_spacy=False, min_confidence=0.99)
        assert rx.extract("Python was created by Guido.") == []


class TestPipelineExtraction:

    def test_add_documents_extracts_triples(self, config):
        p = GRAGPipeline(config=config, relation_extractor=RelationExtractor(use_spacy=False))
        added = p.add_documents([
            {"content": "Python was created by Guido van Rossum.", "source": "chat"},
        ])
        assert added >= 1
        assert p.kg.validate_relationship("python", "created_by", "guido van rossum") > 0

    def test_extraction_can_be_disabled(self, config):
        p = GRAGPipeline(config=config, relation_extractor=RelationExtractor(use_spacy=False))
        before = p.kg.stats()["edges"]
        added = p.add_documents(
            [{"content": "Python was created by Guido van Rossum.", "source": "chat"}],
            extract_relations=False,
        )
        assert added == 0
        assert p.kg.stats()["edges"] == before
