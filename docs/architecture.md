# Architecture

GRAG-System consists of several modular components.

## Query Layer

Responsible for:

- Entity extraction
- Intent classification
- Query decomposition

---

## Retrieval Layer

Supports:

Graph traversal

Semantic retrieval

Hybrid retrieval


---

## Context Builder

Aggregates retrieved evidence.

Ranks relevant nodes.

Builds prompts.

---

## LLM Layer

Compatible with:

OpenAI

Anthropic

Ollama

Local models


---

## Reinforcement Layer

Collects feedback.

Updates rewards.

Optimizes retrieval behavior.
