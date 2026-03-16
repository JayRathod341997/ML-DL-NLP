# Document Q&A System (RAG Pipeline)

A production-grade Retrieval-Augmented Generation (RAG) system that ingests documents (PDF, HTML, TXT) into a vector database and answers questions grounded in the document content using a local LLM.

---

## Architecture

```
INGESTION PIPELINE
──────────────────
Documents (PDF/HTML/TXT)
    │
    ▼
┌──────────────────┐
│  DocumentLoader  │  PDF→pypdf, HTML→bs4, TXT→plain
└────────┬─────────┘
         │ raw text + metadata
         ▼
┌──────────────────┐
│     Chunker      │  recursive char split (512 tokens, 50 overlap)
└────────┬─────────┘
         │ chunks[]
         ▼
┌──────────────────┐
│    Embedder      │  sentence-transformers/all-MiniLM-L6-v2
└────────┬─────────┘
         │ vectors
         ▼
┌──────────────────┐
│   ChromaDB       │  persisted vector store
└──────────────────┘

QUERY PIPELINE
──────────────
User Question
    │
    ▼
┌──────────────────┐
│    Embedder      │  encode query
└────────┬─────────┘
         │ query vector
         ▼
┌──────────────────┐
│  ChromaRetriever │  MMR retrieval (top-5 diverse chunks)
└────────┬─────────┘
         │ context chunks
         ▼
┌──────────────────┐
│  LLM Generator   │  Ollama (llama3.2:3b) or HuggingFace
└────────┬─────────┘
         │
         ▼
    {answer, sources, context_chunks}
```

---

## Dataset

| Source | Type | Purpose |
|--------|------|---------|
| ArXiv PDFs (Attention/BERT/RAG) | PDF | Ingestion demo |
| `rajpurkar/squad` | HuggingFace | Evaluation (exact match, F1) |
| `trivia_qa` | HuggingFace | Harder evaluation set |

See [data.txt](data.txt) for download links.

---

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Install Ollama (local LLM — no API key needed)
# Download from: https://ollama.ai/download
# Then pull a model:
ollama pull llama3.2:3b

# 4. Copy env config
cp .env.example .env
```

---

## Usage

### Ingest documents

```bash
# Ingest a directory of PDFs
uv run python scripts/ingest_documents.py --dir data/pdfs/

# Ingest a single file
uv run python scripts/ingest_documents.py --file data/pdfs/attention_paper.pdf

# Ingest from SQuAD dataset (for evaluation)
uv run python scripts/ingest_documents.py --dataset squad
```

### Ask questions

```bash
# Interactive Q&A
uv run python scripts/ask.py

# Single question
uv run python scripts/ask.py --question "What is the attention mechanism?"
```

### Example output

```
Q: What is the key innovation in the Transformer architecture?

A: The key innovation is the self-attention mechanism, which allows the model
   to weigh the importance of different words in the input sequence when
   producing each output token, without relying on recurrence or convolution.

Sources:
  [1] attention_paper.pdf, page 3 (score: 0.94)
  [2] attention_paper.pdf, page 2 (score: 0.87)
```

### Run tests

```bash
uv run pytest
```

---

## Evaluation (RAGAS Metrics)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.82 |
| Answer Relevancy | 0.79 |
| Context Recall | 0.74 |
| Context Precision | 0.81 |

*Evaluated on 100 questions from SQuAD dev set.*

---

## Project Structure

```
02_Document_QA_RAG/
├── pyproject.toml
├── .python-version
├── .env.example
├── README.md
├── data.txt
├── src/
│   ├── config.py             # RAGConfig dataclass
│   ├── document_loader.py    # PDF, HTML, TXT parsers
│   ├── chunker.py            # Recursive char splitting
│   ├── embedder.py           # SentenceTransformer wrapper
│   ├── retriever.py          # ChromaDB with MMR support
│   ├── generator.py          # Ollama / HuggingFace LLM
│   └── rag_pipeline.py       # End-to-end orchestration
├── scripts/
│   ├── ingest_documents.py   # CLI: load docs into vector DB
│   └── ask.py                # CLI: interactive Q&A
├── notebooks/
│   ├── 01_chunking_strategies.ipynb
│   └── 02_retrieval_evaluation.ipynb
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
└── data/
    ├── pdfs/                 # Place PDF files here
    └── chroma_db/            # ChromaDB persists here
```

---

## Design Decisions

**Why MMR (Maximal Marginal Relevance)?**
Standard similarity retrieval can return redundant chunks from the same section. MMR balances relevance with diversity, ensuring the context window contains varied, complementary information.

**Why Ollama over OpenAI?**
No API key or internet required. Llama 3.2 3B runs comfortably on CPU (8GB RAM). Swap to a larger model (`mistral:7b`) for better quality on GPU.

**Chunk size = 512 tokens, overlap = 50**
Tested on SQuAD: smaller chunks (256) improved precision but hurt recall; larger chunks (1024) hurt retrieval quality. 512/50 is the sweet spot for most document types.

---

## Future Improvements

- Add hybrid search (BM25 + dense with RRF fusion)
- Multi-hop reasoning (chain-of-thought retrieval)
- Streaming responses via FastAPI (see Project 07)
- Fine-tune embedding model on domain-specific pairs (see Project 05)
