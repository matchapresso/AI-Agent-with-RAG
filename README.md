# AI Agent with RAG — Portfolio

> **Author:** Areta Vahtsa Nur Kirana  
> **Notebooks:** `1. End-to-end RAG.ipynb` · `2. AI Agent with Tool Use.ipynb`

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Part 1 — End-to-End RAG System](#part-1--end-to-end-retrieval-augmented-generation-rag-system)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Retrieval Strategy](#2-retrieval-strategy)
  - [3. Model Selection](#3-model-selection)
  - [4. RAG Pipeline](#4-rag-pipeline)
  - [5. Evaluation Dataset](#5-evaluation-dataset)
  - [6. Evaluation Metrics](#6-evaluation-metrics)
  - [7. Recommendations](#7-recommendations)
- [Part 2 — AI Agent with Tool Use](#part-2--ai-agent-with-tool-use)
  - [1. Agent Architecture](#1-agent-architecture)
  - [2. Tool Registry](#2-tool-registry)
  - [3. Agent Behaviors](#3-agent-behaviors)
  - [4. Agent Evaluation](#4-agent-evaluation)
  - [5. Documentation](#5-documentation)
- [Setup & Installation](#setup--installation)
- [Running the Notebooks](#running-the-notebooks)
- [License](#license)

---

## Overview

This repository contains two Jupyter Notebooks. Together, they form a complete on-premise AI system for internal document intelligence:

- **Part 1** builds a full **Retrieval-Augmented Generation (RAG)** pipeline from raw document preprocessing through to graded evaluation.
- **Part 2** wraps that RAG system into an **autonomous AI Agent** equipped with multiple tools that can automate multi-step tasks via natural language.

Both parts use open-source, Google Colab-friendly components throughout.

---

## Repository Structure

```
AI-Agent-with-RAG/
│
├── 1. End-to-end RAG.ipynb          # Part 1: RAG pipeline
├── 2. AI Agent with Tool Use.ipynb  # Part 2: AI Agent
├── README.md
└── LICENSE
```

---

## Part 1: End-to-End Retrieval Augmented Generation (RAG) System

The goal of Part 1 is to prototype an on-premise RAG system that helps engineers retrieve accurate, grounded answers from a private collection of internal documents (technical guidelines, system architecture notes, product specifications, compliance documents, and engineering handbooks).

---

### 1. Data Preprocessing

Raw documents are cleaned and normalized before being split into retrievable chunks.

**Steps:**
- **Text Cleaning:** Remove noise (excessive whitespace, special characters, encoding artifacts) and normalize casing where appropriate.
- **Document Chunking:** Documents are split into overlapping fixed-size chunks to balance context completeness with retrieval granularity.
- **Metadata Enrichment:** Each chunk is tagged with:
  - `source_doc` — the filename or document identifier
  - `chunk_id` — sequential chunk index within the document
  - `offset` — character offset of the chunk within the original document

**Chunking Strategy Justification:**

A sliding-window approach (e.g., 512 tokens with ~10% overlap) is used because:
- Technical documents often contain dense, context-dependent passages, overlap prevents answers from being split across chunk boundaries.
- Fixed-size chunks produce consistent embedding dimensions across all indexed vectors.
- Semantic chunking was considered but adds latency; fixed-size is a practical baseline that performs well across document types.

---

### 2. Retrieval Strategy

Three retrieval strategies are implemented and compared:

| Strategy | Description |
|---|---|
| **Dense Retrieval** | Embeddings stored in a vector database (ChromaDB / FAISS); cosine similarity search |
| **Sparse Retrieval** | BM25 keyword-based ranking using `rank_bm25` |
| **Hybrid Retrieval** | Reciprocal Rank Fusion (RRF) to merge dense and sparse ranked lists |

**Justification:**
- *Dense retrieval* captures semantic similarity and handles paraphrased queries well but can miss exact keyword matches.
- *Sparse retrieval (BM25)* excels at precise keyword and entity lookups (e.g., version numbers, product names) but fails on synonymous queries.
- *Hybrid retrieval* combines the strengths of both, producing more robust recall across query types, hence particularly valuable for engineering documents that mix technical jargon with natural language.

---

### 3. Model Selection

All models are chosen to be open-source and runnable within Google Colab's free-tier constraints (≤15 GB RAM, no persistent GPU guaranteed).

| Role | Model | Justification |
|---|---|---|
| **Embedding** | `all-MiniLM-L6-v2` (Sentence Transformers) | Small (80 MB), fast, strong semantic quality for English technical text |
| **LLM (Generation)** | `google/flan-t5-base` or `TinyLlama-1.1B-Chat` | Fits in Colab RAM; instruction-tuned; acceptable latency |

**Tradeoff Summary:**

| Dimension | Small Model (chosen) | Larger Model |
|---|---|---|
| Latency | Fast (< 2s) | Slow (10–30s) |
| Accuracy | Moderate | Higher |
| RAM | 1–3 GB | 8–40 GB |
| Colab compatibility | ✅ | ⚠️ / ❌ |

---

### 4. RAG Pipeline

The full pipeline follows four sequential stages:

```
[Raw Documents]
      │
      ▼
[Preprocessing & Chunking]
      │
      ▼
[Embedding + Indexing → Vector DB]
      │
      ▼  ◄── Query
[Retrieval (Dense / Sparse / Hybrid)]
      │
      ▼
[Context Assembly]
      │
      ▼
[Prompt Construction]
      │
      ▼
[Local LLM Inference]
      │
      ▼
[Grounded Answer]
```

**Prompt Design (Best Practices Applied):**
- System context instructs the model to answer *only* from provided context and say "I don't know" if the answer isn't present.
- Retrieved chunks are clearly delimited with `<context>` tags.
- The user question is placed at the end to leverage recency bias in attention.
- Few-shot examples are included where applicable to guide output format.

---

### 5. Evaluation Dataset

A set of **20+ question–answer pairs** is constructed manually from the source document(s), covering a range of question types:

- Factual lookups (e.g., "What is the retention period defined in the compliance document?")
- Reasoning questions (e.g., "Why is BM25 preferred over dense retrieval for exact entity matching?")
- Multi-hop questions that require synthesizing across chunks

Each entry in the evaluation dataset contains:

| Field | Description |
|---|---|
| `question` | Natural language query |
| `expected_answer` | Ground-truth reference answer |
| `reference_doc` | Source document / chunk identifier |

---

### 6. Evaluation Metrics

**Retrieval Metrics:**

| Metric | Description |
|---|---|
| `Recall@k` | Fraction of relevant documents retrieved in top-k results |
| `MRR` (Mean Reciprocal Rank) | Average of reciprocal rank of first relevant result |
| `Hit Rate@k` | Binary: whether a relevant chunk appears in top-k |

**Generation Metrics:**

| Metric | Description |
|---|---|
| `Faithfulness` | Does the answer stay grounded in the retrieved context? (no hallucination) |
| `Answer Relevance` | Is the generated answer responsive to the question? |

Faithfulness and relevance are evaluated using an LLM-as-judge pattern (e.g., prompting the model to score 1–5) or via RAGAS-compatible scoring where available.

---

### 7. Recommendations

The following improvements are proposed for a production-grade system:

1. **Semantic Chunking**: Use sentence boundaries or section headings (e.g., via `spaCy` or `NLTK`) instead of fixed-size splits to preserve coherent context units.
2. **Better Embedding Models**: Upgrade to `BAAI/bge-base-en-v1.5` or `intfloat/e5-base-v2` for higher retrieval precision on technical corpora.
3. **Re-ranking**: Add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) after initial retrieval to improve precision@1.
4. **Query Expansion**: Generate query variations using an LLM to reduce vocabulary mismatch in sparse retrieval.
5. **Larger LLM**: Use `Mistral-7B-Instruct` or `Llama-3-8B-Instruct` (quantized via `bitsandbytes`) for higher generation quality within GPU limitation (Colab's free A100/T4 tier).
6. **Automated Evaluation (RAGAS)**: Integrate the `ragas` library for systematic, reproducible generation evaluation without manual scoring.
7. **Persistent Vector Store**: Replace in-memory indexing with a persistent ChromaDB or Qdrant instance for production use.
8. **Incremental Indexing**: Support adding new documents without full re-embedding by using an append-friendly index structure.
9. **Metadata Filtering**: Enable pre-filtering by `source_doc` type (e.g., only query compliance documents) to improve precision for domain-specific queries.
10. **Scalability via Async Batch Processing**: Parallelize embedding and indexing using `asyncio` or `concurrent.futures` for large document collections.

---

## Part 2: AI Agent with Tool Use

Part 2 builds an autonomous AI Agent that can perform multi-step tasks over the RAG knowledge base and beyond, using a **tool-based architecture**.

---

### 1. Agent Architecture

The agent uses a **ReAct (Reasoning + Acting)** loop:

```
User Query
    │
    ▼
[THINK] — Analyze intent, form a plan
    │
    ▼
[ACT]   — Select and invoke the appropriate tool
    │
    ▼
[OBSERVE] — Parse tool output
    │
    ▼
[REPEAT if multi-step] ──────────┐
    │                            │
    ▼                         (loop)
[ANSWER] — Produce final response
```

The controller maintains a running scratchpad of thought–action–observation triples, enabling it to chain multiple tool calls, handle partial results, and recover from errors before producing a final structured output.

---

### 2. Tool Registry

The agent has access to the following tools:

| # | Tool | Description |
|---|---|---|
| 1 | **RAG Search Tool** | Queries the vector index from Part 1; returns top-k relevant chunks with metadata |
| 2 | **Calculator Tool** | Evaluates arithmetic and mathematical expressions safely using Python's `eval` with a sandboxed context |
| 3 | **Python Sandbox** | Executes arbitrary Python snippets in a restricted namespace for data transformations and computations |
| 4 | **Summarizer Tool** | Condenses long retrieved passages or documents into concise bullet-point summaries |
| 5 | **Text Transformation Tool** | Applies text operations: translation, format conversion, tone adjustment, keyword extraction |

Each tool is registered in a central **Tool Registry** dictionary with: `name`, `description`, `input_schema`, and `callable`.

---

### 3. Agent Behaviors

The agent satisfies the following behavioral requirements:

- **Intent-based Tool Selection**: The LLM reasons about the user's intent and maps it to the correct tool before invoking it (e.g., arithmetic intent → Calculator, document lookup intent → RAG Search).
- **Multi-step Execution**:Tasks requiring more than one tool are broken down into sequential sub-steps (e.g., retrieve a document → summarize it → calculate a metric from it).
- **Structured Output**: Final responses follow a consistent format with `answer`, `sources`, and `tool_trace` fields.
- **Graceful Error Handling**: If a tool fails (invalid expression, no retrieval results), the agent catches the exception, logs the failure, and either retries with a reformulated query or returns a safe fallback message.

**5 Test Run Transcripts** are provided within the notebook, covering:

| Run | Task Type |
|---|---|
| 1 | Single-hop document retrieval question |
| 2 | Arithmetic calculation from retrieved data |
| 3 | Multi-step: retrieve → summarize → transform |
| 4 | Edge case: out-of-scope question (graceful fallback) |
| 5 | Multi-step: retrieve → calculate → synthesize answer |

---

### 4. Agent Evaluation

Qualitative evaluation across four dimensions:

| Dimension | Description | Assessment |
|---|---|---|
| **Tool Selection Accuracy** | Did the agent choose the correct tool for the intent? | Evaluated across all 5 test runs |
| **Execution Reliability** | Did the agent complete the task without unexpected crashes? | Checked for stability across run types |
| **Error Handling** | Did the agent recover gracefully from failures? | Tested via the edge-case run (Run 4) |
| **Multi-step Task Stability** | Did the agent maintain coherent state across tool calls? | Evaluated in Runs 3 and 5 |

---

### 5. Documentation

#### Agent Loop

1. Receive user query.
2. Feed query + tool descriptions into the LLM controller prompt.
3. LLM produces a `Thought` and selects an `Action` (tool name + input).
4. Tool is invoked; output is appended as `Observation`.
5. Steps 3–4 repeat until the LLM produces a `Final Answer`.
6. Structured response is returned to the user.

#### Tool Registry

Tools are stored as a Python dict:
```python
TOOL_REGISTRY = {
    "rag_search": {
        "description": "Retrieve relevant document chunks from the internal knowledge base.",
        "callable": rag_search_fn,
    },
    "calculator": {
        "description": "Evaluate a mathematical expression and return the result.",
        "callable": calculator_fn,
    },
    ...
}
```

#### Execution Flow

```
User → Agent Controller → [Tool Selection] → Tool Execution → Observation → [Loop / Final Answer]
```

#### RAG Integration

The `rag_search` tool directly calls the indexed vector store built in Part 1. It accepts a natural language query, runs hybrid retrieval (dense + BM25), and returns the top-k chunks with `source_doc`, `chunk_id`, and relevance score.

#### Logging & Observability

- Every tool invocation is logged with: timestamp, tool name, input, output, and latency.
- The full thought–action–observation chain is stored per-run as a `transcript` list.
- Errors are caught and logged with exception type and message for post-hoc debugging.

---

## Setup & Installation

### Option A: Google Colab (Recommended)

1. Open the notebooks directly in Google Colab using the links at the top of each notebook.
2. Run the first cell (`!pip install ...`) to install all dependencies.
3. No additional configuration is needed for Part 1. Part 2 imports the RAG components built in Part 1, so run Notebook 1 first (or mount the same runtime).

### Option B: Local Environment

```bash
# Clone the repository
git clone https://github.com/matchapresso/AI-Agent-with-RAG.git
cd AI-Agent-with-RAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies include:**

```
sentence-transformers
chromadb
rank-bm25
transformers
torch
langchain
ragas          # optional, for evaluation
jupyter
```

> ⚠️ A GPU is recommended for local inference but not required. CPU inference is supported at reduced speed.

---

## Running the Notebooks

Run the notebooks **in order**:

1. **`1. End-to-end RAG.ipynb`** — Preprocesses documents, builds the vector index, runs the RAG pipeline, and evaluates retrieval and generation quality.

2. **`2. AI Agent with Tool Use.ipynb`** — Imports the RAG components from Notebook 1, registers all tools, initializes the ReAct agent, and runs the 5 test transcripts.

Each notebook is self-contained with markdown cells explaining every step.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

> *This repository is a submission for the GDP Labs SDE GenAI Practical Test. All contents are confidential per GDP Labs guidelines.*
