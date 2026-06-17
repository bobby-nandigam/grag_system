"""
GRAG CLI — Command-line interface for the GRAG system.

Usage
-----
  grag query "Who created Python?"
  grag ingest --file documents.json
  grag stats
  grag interactive
"""

import argparse
import json
import sys
import logging

logger = logging.getLogger(__name__)


def _get_demo_pipeline():
    """Create a demo pipeline pre-loaded with example knowledge."""
    from grag import GRAGPipeline, GRAGConfig

    pipeline = GRAGPipeline(config=GRAGConfig(verbose=False))

    # Demo knowledge graph triples
    triples = [
        ("python", "created_by", "guido van rossum", 0.99),
        ("guido van rossum", "works_at", "google", 0.95),
        ("python", "type", "programming language", 0.99),
        ("python", "released_in", "1991", 0.99),
        ("tensorflow", "created_by", "google", 0.99),
        ("pytorch", "created_by", "meta ai", 0.99),
        ("nvidia", "produces", "gpus", 0.99),
        ("nvidia", "founded_in", "1993", 0.99),
        ("cuda", "developed_by", "nvidia", 0.99),
        ("transformer", "introduced_by", "google brain", 0.97),
        ("bert", "created_by", "google", 0.99),
        ("gpt", "created_by", "openai", 0.99),
        ("openai", "founded_by", "elon musk", 0.8),
        ("openai", "founded_by", "sam altman", 0.99),
        ("deep learning", "pioneered_by", "geoffrey hinton", 0.97),
    ]
    for s, p, o, c in triples:
        pipeline.kg.add_triple(s, p, o, confidence=c)

    # Demo documents
    docs = [
        {"content": "Python is a high-level programming language created by Guido van Rossum in 1991. It is widely used in data science, web development, and AI.", "source": "wiki"},
        {"content": "NVIDIA Corporation is an American multinational technology company founded in 1993. It designs GPUs and CUDA, widely used for AI training.", "source": "wiki"},
        {"content": "Google was founded by Larry Page and Sergey Brin. Google Brain created TensorFlow and the Transformer architecture.", "source": "wiki"},
        {"content": "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Google Brain researchers in 2017.", "source": "arxiv"},
        {"content": "PyTorch is an open-source machine learning framework developed by Meta AI (formerly Facebook AI Research).", "source": "wiki"},
        {"content": "BERT (Bidirectional Encoder Representations from Transformers) is a language model created by Google.", "source": "wiki"},
        {"content": "Deep learning was pioneered by Geoffrey Hinton, often called the 'Godfather of AI'.", "source": "wiki"},
        {"content": "CUDA is a parallel computing platform developed by NVIDIA that enables GPU-accelerated computations.", "source": "nvidia"},
    ]
    pipeline.add_documents(docs)
    return pipeline


def cmd_query(args):
    pipeline = _get_demo_pipeline()
    result = pipeline.query(args.question)
    print(result)


def cmd_ingest(args):
    from grag import GRAGPipeline, GRAGConfig
    pipeline = GRAGPipeline(config=GRAGConfig())

    with open(args.file) as f:
        docs = json.load(f)

    triples = pipeline.add_documents(docs)
    print(f"Ingested {len(docs)} documents | extracted {triples} knowledge-graph triples.")


def cmd_stats(args):
    pipeline = _get_demo_pipeline()
    stats = pipeline.stats()
    print(json.dumps(stats, indent=2))


def cmd_interactive(args):
    """Interactive REPL session."""
    print("\n" + "="*60)
    print("  GRAG — Graph Retrieval-Augmented Generation System")
    print("  Type 'exit' or 'quit' to exit | 'stats' for stats")
    print("="*60 + "\n")

    pipeline = _get_demo_pipeline()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "stats":
            print(json.dumps(pipeline.stats(), indent=2))
            continue

        result = pipeline.query(question)
        print(result)


def main():
    parser = argparse.ArgumentParser(
        prog="grag",
        description="GRAG — Graph Retrieval-Augmented Generation System",
    )
    subparsers = parser.add_subparsers(dest="command")

    # query
    p_query = subparsers.add_parser("query", help="Answer a single question")
    p_query.add_argument("question", type=str, help="The question to ask")
    p_query.set_defaults(func=cmd_query)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents from JSON file")
    p_ingest.add_argument("--file", required=True, help="Path to JSON file")
    p_ingest.set_defaults(func=cmd_ingest)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show system statistics")
    p_stats.set_defaults(func=cmd_stats)

    # interactive
    p_inter = subparsers.add_parser("interactive", help="Launch interactive REPL")
    p_inter.set_defaults(func=cmd_interactive)

    args = parser.parse_args()

    if not args.command:
        # Default: launch interactive mode
        cmd_interactive(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
