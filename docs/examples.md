# Examples

This section demonstrates common usage patterns for GRAG-System.

---

# Basic Query

The simplest way to interact with GRAG-System.

```python
from grag import GraphRAG

rag = GraphRAG()

response = rag.query(
    "What is Graph Retrieval Augmented Generation?"
)

print(response)
```

---

# Multi-Hop Reasoning

Traverse relationships between entities.

```python
response = rag.query(

"""
Find researchers working on Graph RAG
who later joined NVIDIA.
"""

)

print(response)
```

GRAG attempts to perform:

Entity Extraction

↓

Relationship Discovery

↓

Multi-hop Traversal

↓

Evidence Aggregation

↓

Answer Generation

---

# Query Understanding

Analyze query structure before retrieval.

```python
from grag.retrieval.query_understanding import QueryParser


parser = QueryParser()

result = parser.parse(

"Find papers written by DeepMind researchers on Graph Neural Networks"

)

print(result)

```

Example output

```python
{
 "entities":[
      "DeepMind",
      "Graph Neural Networks"
 ],

 "intent":"search",

 "constraints":[],


 "reasoning_depth":2

}
```

---

# Reward Learning

Update retrieval preferences using feedback.

```python
from grag.rl.reward_engine import RewardEngine


reward = RewardEngine()


reward.update(

query="What is RAG?",


response="...",


reward=1.0

)

```

Negative feedback

```python
reward.update(

query=q,


response=r,


reward=-1

)

```

---

# Knowledge Graph Construction

Build graphs from relationships.

```python
import networkx as nx


graph = nx.Graph()



graph.add_edge(

"GraphRAG",

"NVIDIA"

)


graph.add_edge(

"NVIDIA",

"Research"

)

```

---

# Hybrid Retrieval

Combine semantic retrieval with graph traversal.

```python
results = rag.retrieve(


query="Graph Transformers",


strategy="hybrid"

)


```

Supported strategies

```python
vector


graph


hybrid

```

---

# OpenAI Integration

```python
from openai import OpenAI


client = OpenAI()



response = rag.query(


question="Explain Graph Attention Networks",



llm=client

)

```

---

# Batch Processing

Evaluate multiple queries.

```python
queries=[

"What is GraphRAG?",


"Explain RLHF",


"Describe Neo4j"

]



for q in queries:


    print(


        rag.query(q)

    )

```

---

# Local Models

Example with Ollama.

```python
response = rag.query(


question="What is GraphRAG?",



model="mistral"

)

```

---

# End-to-End Workflow

```python
from grag import GraphRAG


rag = GraphRAG()



question = """


Find Graph RAG researchers who later joined NVIDIA.

"""



answer = rag.query(question)



print(answer)



reward_engine.update(


query=question,


response=answer,


reward=1.0

)

```

Pipeline

```text
User Query
↓

Query Understanding
↓

Knowledge Graph
↓

Multi-hop Retrieval
↓

Context Builder
↓

LLM
↓

Answer
↓

User Feedback
↓

Reward Engine
```

---

# Future Examples

Upcoming tutorials

• Neo4j Integration

• DSPy Support

• LangChain Connectors

• Agent Memory

• Distributed Retrieval

• Benchmarks

• Graph Embeddings

• Human Preference Optimization

---

Thank you for using GRAG-System.

Contributions are welcome.

GitHub

https://github.com/bobby-nandigam/grag_system

PyPI

https://pypi.org/project/grag-system/
