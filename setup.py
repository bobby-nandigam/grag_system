"""
GRAG — Graph Retrieval-Augmented Generation System
pip install grag-system
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="grag-system",
    version="1.0.0",
    author="Bobby Nandigam",
    author_email="bobbynandigam.official@gmail.com",
    description=(
        "Production-grade Graph RAG system with RL self-improvement, "
        "multi-hop knowledge graph reasoning, and explainable AI outputs."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bobby-nandigam/grag_system",
    project_urls={
        "Bug Tracker": "https://github.com/bobby-nandigam/grag_system/issues",
        "Documentation": "https://github.com/bobby-nandigam/grag_system#readme",
    },
    license="MIT",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.8",

    install_requires=[
        "networkx>=3.0",
        "numpy>=1.24",
    ],

    extras_require={
        # Semantic embeddings (recommended)
        "ml": [
            "sentence-transformers>=2.2",
            "torch>=2.0",
        ],
        # NLP entity recognition
        "nlp": [
            "spacy>=3.5",
        ],
        # OpenAI-backed generation
        "openai": [
            "openai>=1.0",
        ],
        # Full install
        "all": [
            "sentence-transformers>=2.2",
            "torch>=2.0",
            "spacy>=3.5",
            "openai>=1.0",
            "faiss-cpu>=1.7",
        ],
        # Developer tools
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
            "ruff>=0.1",
        ],
    },

    entry_points={
        "console_scripts": [
            "grag=grag.cli:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    keywords=[
        "rag", "graph-rag", "knowledge-graph", "retrieval-augmented-generation",
        "nlp", "llm", "ai", "reinforcement-learning", "explainable-ai",
        "networkx", "vector-search", "grag",
    ],
)
