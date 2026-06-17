# GRAG — Graph Retrieval-Augmented Generation System

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" />
  <img src="https://img.shields.io/badge/version-1.1.0-orange?style=flat-square" alt="Version 1.1.0" />
  <img src="https://img.shields.io/badge/pip%20install-grag--system-purple?style=flat-square" alt="pip install grag-system" />
</p>

A production-grade Graph RAG system that combines knowledge-graph reasoning, vector similarity search, reinforcement-learning self-improvement, and explainable outputs — all in a single `pip install`.

## Overview

GRAG is a self-improving Retrieval-Augmented Generation (RAG) system that goes beyond naive vector search. It integrates the following components:

| Component | Responsibility |
|---|---|
| Knowledge Graph | Multi-hop reasoning over entities and relationships, with persistence |
| Relation Extractor | Turns ingested text into knowledge-graph triples automatically |
| Hybrid Retrieval | Combines vector similarity with graph-neighbor expansion |
| Graph Reasoner | Entity linking, relationship validation, contradiction detection |
| Critic Module | Self-evaluates faithfulness, relevance, completeness, and consistency |
| Refinement Loop | Iteratively improves answers when confidence is low |
| RL Reward Engine | Learns from every query to improve future retrieval strategies |
| Memory Store | Remembers successful patterns and avoids repeated failures |
| Safety Guardrails | Returns "Insufficient evidence" rather than hallucinating when unsure |

## Installation

Minimal install (no machine-learning dependencies):

```bash
pip install grag-system
```

With semantic embeddings (recommended):

```bash
pip install grag-system[ml]
```

With NLP entity recognition and higher-quality relation extraction:

```bash
pip install grag-system[nlp]
python -m spacy download en_core_web_sm
```

Full installation:

```bash
pip install grag-system[all]
```

## Quick Start

```python
from grag import GRAGPipeline, GRAGConfig

# Initialize pipeline
pipeline = GRAGPipeline(config=GRAGConfig())

# Add your knowledge graph
pipeline.kg.add_triple("python",          "created_by",   "guido van rossum", confidence=0.99)
pipeline.kg.add_triple("guido van rossum", "works_at",    "google",           confidence=0.95)
pipeline.kg.add_triple("nvidia",          "produces",     "gpus",             confidence=0.99)
pipeline.kg.add_triple("cuda",            "developed_by", "nvidia",           confidence=0.99)

# Index your documents
pipeline.add_documents([
    {"content": "Python is a high-level language created by Guido van Rossum in 1991.", "source": "wiki"},
    {"content": "NVIDIA designs GPUs and CUDA for AI training and parallel computing.",  "source": "nvidia"},
    {"content": "Guido van Rossum works at Google as a software engineer.",               "source": "linkedin"},
])

# Query
result = pipeline.query("Who created Python and where do they work now?")
print(result)
```

Example output (minimal install, built-in TF-IDF fallback):

```
============================================================
ANSWER:
Based on the knowledge graph: python --[created_by]--> guido van rossum;
python --[created_by]--> guido van rossum | guido van rossum --[works_at]--> google.
Supporting evidence: Python is a high-level language created by Guido van Rossum
in 1991. NVIDIA designs GPUs and CUDA for AI training and parallel computing.

GRAPH PATH:
python --[created_by]--> guido van rossum

DOCUMENT SUPPORT:
[wiki]   (score=0.38): Python is a high-level language created by Guido van Rossum in 1991....
[nvidia] (score=0.32): NVIDIA designs GPUs and CUDA for AI training and parallel computing....

CONFIDENCE: 0.93
ITERATIONS: 1
============================================================
```

Exact similarity scores and document ranking depend on the embedding backend. The
output above uses the built-in TF-IDF fallback that ships with the minimal install;
install `grag-system[ml]` for sentence-transformer embeddings and higher-quality
ranking. All entity names are normalized to lowercase inside the knowledge graph.

## Command-Line Usage

After installation, the `grag` command is available directly:

```bash
# Ask a question (uses built-in demo knowledge)
grag query "Who created Python?"

# Interactive REPL
grag interactive

# Ingest documents from a JSON file
grag ingest --file my_documents.json

# Show system statistics
grag stats
```

Interactive session example:

```
============================================================
  GRAG — Graph Retrieval-Augmented Generation System
  Type 'exit' or 'quit' to exit | 'stats' for stats
============================================================

You: Who created Python?
...
You: What does NVIDIA produce?
...
You: stats
{"kg": {"nodes": 21, "edges": 15, ...}, "vector_store_size": 8, "rl": {...}, "memory": {...}}
```

The `query`, `stats`, and `interactive` commands run against a built-in demo
knowledge base. Use `grag ingest --file documents.json` to load your own documents,
where the JSON file is a list of objects each containing a `content` field and an
optional `source` field. Ingestion also extracts knowledge-graph triples from the
document text.

## Architecture

```
User Query
    |
    v
[1] QueryUnderstanding          Intent detection, entity extraction, constraint parsing
    |
    v
[2] HybridRetriever             Vector search + graph-neighbor boosting, adaptive k
    |         |
    |    VectorStore (embeddings)
    |         |
    v         v
[3] GraphReasoner               Entity linking, multi-hop traversal, path ranking
    |
    v
[4] Context Fusion              Dedup, contradiction detection, confidence weighting
    |
    v
[5] Answer Generation           Graph facts > high-confidence docs > weak signals
    |
    v
[6] Explainability              Graph path, document summary, confidence score
    |
    v
[7] CriticModule                Faithfulness, relevance, completeness, consistency
    |
    +---- PASS --------------------------------------------------> Return GRAGAnswer
    |
    +---- FAIL --> [8] Refinement Loop (up to N iterations)
                        |
                        v
                   [9] RewardEngine    Update retrieval weights, k, max_hops
                        |
                        v
                  [10] MemoryStore     Cache patterns, avoid failures
```

## Configuration

```python
from grag import GRAGConfig

# Default
config = GRAGConfig()

# Fast prototyping
config = GRAGConfig.fast()

# Production
config = GRAGConfig.production()

# Custom
config = GRAGConfig(
    top_k=8,                      # Documents retrieved per query
    max_hops=3,                   # Maximum knowledge-graph traversal depth
    confidence_threshold=0.75,    # Minimum score to accept an answer without refinement
    max_refinement_iterations=4,  # Maximum self-improvement loops
    embedding_model="all-MiniLM-L6-v2",
    graph_weight=0.6,             # Graph-facts weight in fusion
    vector_weight=0.4,            # Vector-docs weight in fusion
    verbose=True,
)
```

Note: `graph_weight` and `vector_weight` must sum to `1.0`, and `confidence_threshold` must fall within the range `(0, 1]`. Both constraints are validated on construction.

## Core API

```python
# Add facts to the knowledge graph (subject, predicate, object).
# Entities and predicates are normalized to lowercase. Re-adding the same fact
# updates it in place rather than creating a duplicate edge.
pipeline.kg.add_triple(subject, predicate, obj, confidence=1.0, source="manual")

# Index documents and auto-extract knowledge-graph triples from their text.
# Each dict requires a "content" key; "source" and "metadata" are optional.
# Returns the number of triples added; pass extract_relations=False to skip extraction.
n_triples = pipeline.add_documents([{"content": "...", "source": "wiki", "metadata": {}}])

# Run the pipeline. Returns a GRAGAnswer.
answer = pipeline.query(
    question,
    user_feedback=None,      # optional explicit signal in [-1, 1] for the RL update
    metadata_filter=None,    # optional dict to filter retrieved documents by metadata
)

# System-wide statistics: knowledge graph, vector store size, RL engine, memory.
pipeline.stats()
```

## Persistence

A pipeline created with the default configuration persists state across sessions
under `memory_path` (default `.grag_memory`):

- The knowledge graph is saved to `{memory_path}/knowledge_graph.json` and reloaded
  automatically on the next run, so ingested facts survive restarts.
- The memory store is saved to `{memory_path}/memory.json`.

The knowledge graph is serialized as a flat list of triples — a format that is
independent of the installed NetworkX version. Supplying your own
`KnowledgeGraph(...)` to `GRAGPipeline` uses it as-is and disables the default
auto-persistence path.

## Automatic Knowledge-Graph Construction

By default, `add_documents` does not only index text for vector search — it also
mines each document for `(subject, predicate, object)` triples and adds them to the
knowledge graph. This means the graph, and therefore multi-hop reasoning, grows
automatically as documents or chat messages are ingested; manual `add_triple` calls
are optional rather than required.

```python
from grag import GRAGPipeline

pipeline = GRAGPipeline()

# No manual add_triple calls — the graph is built from the text itself.
pipeline.add_documents([
    {"content": "Python is a programming language created by Guido van Rossum.", "source": "chat"},
    {"content": "Guido van Rossum works at Google.", "source": "chat"},
])

pipeline.kg.find_paths("python", "google", max_hops=3)
# -> ["python --[created_by]--> guido van rossum | guido van rossum --[works_at]--> google"]
```

The extractor can also be used directly:

```python
from grag import RelationExtractor

rx = RelationExtractor()
rx.extract("PyTorch is a framework developed by Meta AI.")
# -> [Triple(subject="pytorch", predicate="developed_by", obj="meta ai", confidence=...)]
```

Behavior and limitations:

- Backends: spaCy dependency parsing when the `nlp` extra is installed (more
  accurate, recognizes multi-word entities), otherwise a dependency-free regular
  expression fallback. Extracted triples carry a confidence score and record their
  source, so they are weighted below manually curated facts.
- The regex fallback reliably handles patterns such as "X was created/founded/
  developed by Y", "X produces Y", "X works at Y", and "X is a Y", but it can
  mis-segment lowercase multi-word concepts (for example, "deep learning"). Install
  `grag-system[nlp]` for higher-quality extraction.
- Pass `extract_relations=False` to `add_documents` to index for vector search only.
- Supply a custom extractor (for example, an LLM-backed one) by subclassing
  `RelationExtractor` and overriding `extract`, then passing it to
  `GRAGPipeline(relation_extractor=...)`.

## Pipeline Stages (Detailed)

### Stage 1 — Query Understanding

Parses queries into structured objects with intent, entities, relationships, and constraints.

```python
from grag.retrieval.query_understanding import QueryUnderstanding

qu = QueryUnderstanding()
parsed = qu.parse("List the frameworks created by Google")
# parsed.intent         -> "listing"
# parsed.entities       -> ["Google"]
# parsed.relationships  -> ["created_by"]
# parsed.constraints    -> {"domain": "software"}
```

Entity extraction uses spaCy named-entity recognition when the `nlp` extra is
installed, and falls back to a capitalized-token heuristic otherwise. Multi-word
entities such as "Guido van Rossum" are recognized as a single entity only with
spaCy; the fallback splits them into separate tokens.

### Stage 2 — Hybrid Retrieval

Combines vector similarity with knowledge-graph-neighbor boosting.

```python
from grag.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(config, kg)
retriever.add_documents(docs)
docs = retriever.retrieve(parsed_query)
```

### Stage 3 — Graph Reasoning

Multi-hop entity linking and relationship validation.

```python
from grag.reasoning.graph_reasoner import GraphReasoner

reasoner = GraphReasoner(config, kg)
fused_context = reasoner.reason(parsed_query, retrieved_docs)
# fused_context.graph_facts      -> ["python --[created_by]--> guido van rossum"]
# fused_context.document_chunks  -> [RetrievedDocument(...), ...]
# fused_context.contradictions   -> []
# fused_context.confidence       -> 0.87
```

### Stage 6 — Explainability

Every answer includes a full reasoning trace:

```python
result = pipeline.query("Who created Python?")

print(result.answer)                    # The generated answer text (str)
print(result.graph_path.to_string())    # Entity --[relation]--> Entity chain (GraphPath, may be None)
print(result.document_summary)          # Supporting document excerpts (str)
print(result.confidence)                # 0-1 confidence score (float)
print(result.iterations)                # How many refinement loops ran (int)
print(result.entities_used)             # Key entities in the reasoning (list[str])
print(result.failure_type)              # FailureType enum (NONE when successful)
```

`graph_path` is a `GraphPath` object (or `None` when no supporting path exists), so
guard for `None` before calling `to_string()`.

### Stage 7 — Self-Evaluation (Critic)

```python
from grag.evaluation.critic import CriticModule

critic = CriticModule(config)
eval_result = critic.evaluate(answer, parsed_query, fused_context)

# eval_result.faithfulness  -> 0.92  (no hallucination)
# eval_result.relevance     -> 0.88  (answers user intent)
# eval_result.completeness  -> 0.85  (key entities covered)
# eval_result.consistency   -> 1.0   (no contradictions)
# eval_result.overall_score -> 0.90
# eval_result.passed        -> True
```

### Stage 9 — Reinforcement Learning

```python
from grag.rl.reward_engine import RewardEngine

engine = RewardEngine(config)
reward = engine.record(answer, eval_result, parsed_query, user_feedback=1.0)
# Returns the computed reward (float) and updates the per-pattern strategy:
# graph_weight, vector_weight, top_k, max_hops

print(engine.stats())
# {
#   "total_queries": 42,
#   "average_reward": 0.73,
#   "failure_breakdown": {"none": 38, "retrieval_failure": 4},
#   "patterns_learned": 18,
# }
```

## Running Tests

```bash
# Install development dependencies
pip install grag-system[dev]

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=grag --cov-report=html

# Run a specific test class
pytest tests/test_grag.py::TestKnowledgeGraph -v
pytest tests/test_grag.py::TestRelationExtractor -v
```

## Project Structure

```
grag-system/
|-- grag/
|   |-- __init__.py               # Public API
|   |-- cli.py                    # CLI entry point (grag command)
|   |-- core/
|   |   |-- config.py             # GRAGConfig dataclass
|   |   |-- models.py             # Typed data models
|   |   |-- pipeline.py           # Master orchestrator (multi-stage pipeline)
|   |-- graph/
|   |   |-- knowledge_graph.py    # NetworkX-backed knowledge graph, with persistence
|   |-- extraction/
|   |   |-- relation_extractor.py # Text -> (subject, predicate, object) triples
|   |-- retrieval/
|   |   |-- hybrid_retriever.py   # Vector + graph hybrid search
|   |   |-- query_understanding.py# Query parsing, intent, entity extraction
|   |-- reasoning/
|   |   |-- graph_reasoner.py     # Multi-hop reasoning, contradiction detection
|   |-- evaluation/
|   |   |-- critic.py             # Self-evaluation metrics
|   |-- rl/
|   |   |-- reward_engine.py      # RL reward/penalty engine
|   |-- memory/
|       |-- memory_store.py       # Episodic memory store
|-- tests/
|   |-- test_grag.py              # 57 unit and integration tests
|-- examples/
|   |-- quickstart.py             # End-to-end demo
|-- setup.py
|-- pyproject.toml
|-- README.md
```

## Extending GRAG

### Swap in a real vector database (FAISS, Pinecone, Weaviate)

```python
from grag.retrieval.hybrid_retriever import HybridRetriever

class FAISSRetriever(HybridRetriever):
    def __init__(self, config, kg):
        super().__init__(config, kg)
        import faiss
        self.vector_store = MyFAISSVectorStore()   # plug in your own
```

### Use Neo4j instead of the in-memory graph

```python
from grag.graph.knowledge_graph import KnowledgeGraph

class Neo4jKnowledgeGraph(KnowledgeGraph):
    def __init__(self, uri, user, password):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    # Override add_triple, find_paths, etc.
```

### Plug in an LLM-backed relation extractor

```python
from grag import GRAGPipeline, RelationExtractor, Triple

class LLMRelationExtractor(RelationExtractor):
    def extract(self, text):
        # Call your LLM, parse its output into Triple objects.
        return [Triple(subject="...", predicate="...", obj="...", confidence=0.9)]

pipeline = GRAGPipeline(relation_extractor=LLMRelationExtractor())
```

### Add an LLM-backed answer generator

```python
from grag.core.pipeline import GRAGPipeline

class LLMGRAGPipeline(GRAGPipeline):
    def _compose_answer(self, parsed, facts_part, doc_part, context):
        import openai
        prompt = f"Facts: {facts_part}\nEvidence: {doc_part}\nQuestion: {parsed.raw_query}"
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
```

## Design Goals

GRAG does not yet ship a published benchmark suite. The architecture is designed to
improve on naive vector-only RAG along the following dimensions:

- Faithfulness: answers are grounded in retrieved graph facts and documents, and the
  pipeline abstains with "Insufficient evidence" rather than fabricating a response.
- Multi-hop reasoning: graph traversal connects entities across multiple relationships
  that single-shot vector retrieval cannot reach.
- Explainability: every answer returns its supporting graph path, document excerpts,
  and a calibrated confidence score.
- Self-improvement: the reinforcement-learning reward engine adapts retrieval weights
  and traversal depth per query pattern over time.

Quantitative evaluation against baselines is planned. Contributions of a reproducible
benchmark harness are welcome.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Run the tests: `pytest tests/ -v`.
4. Submit a pull request.

## License

Released under the MIT License. Free for commercial and research use. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{grag2025,
  title  = {GRAG: Graph Retrieval-Augmented Generation with RL Self-Improvement},
  author = {Nandigam, Bobby},
  year   = {2025},
  url    = {https://github.com/bobby-nandigam/grag_system}
}
```

## Links

- Repository: https://github.com/bobby-nandigam/grag_system
- PyPI: https://pypi.org/project/grag-system
</content>
