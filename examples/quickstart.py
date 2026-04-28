"""
GRAG Quickstart — End-to-end demonstration.

Run with: python examples/quickstart.py
"""

from grag import GRAGPipeline, GRAGConfig


def build_pipeline() -> GRAGPipeline:
    """Build a demo pipeline with AI/tech knowledge."""
    pipeline = GRAGPipeline(config=GRAGConfig(verbose=False))

    # ── Knowledge Graph Triples ────────────────────────────────────────────
    triples = [
        ("python",           "created_by",    "guido van rossum", 0.99),
        ("guido van rossum", "works_at",       "google",           0.95),
        ("python",           "type",           "programming language", 0.99),
        ("python",           "first_released", "1991",             0.99),
        ("tensorflow",       "created_by",     "google",           0.99),
        ("pytorch",          "created_by",     "meta ai",          0.99),
        ("nvidia",           "produces",       "gpus",             0.99),
        ("nvidia",           "founded_in",     "1993",             0.99),
        ("cuda",             "developed_by",   "nvidia",           0.99),
        ("cuda",             "used_for",       "gpu computing",    0.99),
        ("transformer",      "introduced_by",  "google brain",     0.97),
        ("bert",             "created_by",     "google",           0.99),
        ("gpt",              "created_by",     "openai",           0.99),
        ("deep learning",    "pioneered_by",   "geoffrey hinton",  0.97),
        ("google",           "founded_by",     "larry page",       0.99),
        ("google",           "founded_by",     "sergey brin",      0.99),
    ]
    for s, p, o, c in triples:
        pipeline.kg.add_triple(s, p, o, confidence=c)

    # ── Documents ─────────────────────────────────────────────────────────
    docs = [
        {
            "content": "Python is a high-level, interpreted programming language created by Guido van Rossum. "
                       "First released in 1991, Python is known for its clean syntax and wide use in AI and data science.",
            "source": "wiki",
        },
        {
            "content": "NVIDIA Corporation is an American technology company founded in 1993. "
                       "NVIDIA designs and sells GPUs used for gaming, professional visualization, data centers, and AI training.",
            "source": "wiki",
        },
        {
            "content": "CUDA (Compute Unified Device Architecture) is a parallel computing platform developed by NVIDIA. "
                       "It enables developers to use NVIDIA GPUs for general-purpose processing, critical for deep learning.",
            "source": "nvidia",
        },
        {
            "content": "TensorFlow is an open-source machine learning framework created by Google. "
                       "PyTorch is a deep learning framework developed by Meta AI (formerly Facebook AI Research).",
            "source": "wiki",
        },
        {
            "content": "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Google Brain researchers. "
                       "It revolutionized NLP and is the foundation for models like BERT and GPT.",
            "source": "arxiv",
        },
        {
            "content": "Guido van Rossum, the creator of Python, joined Google in 2005 as a software engineer. "
                       "He later worked at Dropbox and eventually returned to Google.",
            "source": "techcrunch",
        },
        {
            "content": "Deep learning was pioneered by Geoffrey Hinton, often called the 'Godfather of AI'. "
                       "His work on neural networks laid the groundwork for modern AI.",
            "source": "wiki",
        },
    ]
    pipeline.add_documents(docs)
    return pipeline


def run_demo():
    print("\n" + "=" * 65)
    print("  GRAG — Graph Retrieval-Augmented Generation Demo")
    print("=" * 65)

    pipeline = build_pipeline()
    print(f"\n✅ Pipeline ready | KG stats: {pipeline.kg.stats()}\n")

    queries = [
        "Who created Python and where do they work?",
        "What did NVIDIA develop for GPU computing?",
        "Which company created both TensorFlow and the Transformer architecture?",
        "Who pioneered deep learning?",
        "What is the relationship between CUDA and NVIDIA?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 65}")
        print(f"Query {i}: {query}")
        result = pipeline.query(query)
        print(result)

    # System stats after all queries
    print("\n" + "=" * 65)
    print("SYSTEM STATS AFTER DEMO")
    print("=" * 65)
    import json
    print(json.dumps(pipeline.stats(), indent=2))


if __name__ == "__main__":
    run_demo()
