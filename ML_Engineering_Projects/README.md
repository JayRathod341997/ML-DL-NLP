# ML Engineering Projects

A portfolio of 9 production-quality ML engineering projects covering the core skills for ML Engineer roles in search, NLP, ranking, and MLOps.

---

## Project Map

### Area 1 — ML System Design & Development

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 01 | [Semantic Search Engine](Area_1_ML_System_Design/01_Semantic_Search_Engine/) | Dense vector search over large corpora | sentence-transformers, FAISS, ChromaDB |
| 02 | [Document Q&A RAG](Area_1_ML_System_Design/02_Document_QA_RAG/) | Retrieval-Augmented Generation pipeline | LangChain, ChromaDB, Ollama |
| 03 | [NLP Entity Extraction](Area_1_ML_System_Design/03_NLP_Entity_Extraction/) | NER + document parsing pipeline | HuggingFace Transformers, spaCy |

### Area 2 — Model Training & Optimization

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 04 | [Text Classification BERT](Area_2_Model_Training_Optimization/04_Text_Classification_BERT/) | Fine-tune DistilBERT for text classification | transformers, torch, accelerate |
| 05 | [Sentence Embedding Training](Area_2_Model_Training_Optimization/05_Sentence_Embedding_Training/) | Fine-tune sentence embeddings with MNRL loss | sentence-transformers, NLI data |
| 06 | [Learning to Rank](Area_2_Model_Training_Optimization/06_Learning_To_Rank/) | LambdaMART + neural cross-encoder reranker | LightGBM, FAISS, BM25 |

### Area 3 — MLOps & Productionisation

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 07 | [ML Model REST API](Area_3_MLOps_Productionisation/07_ML_Model_REST_API/) | Production FastAPI inference server | FastAPI, Docker, Prometheus |
| 08 | [Batch Inference Pipeline](Area_3_MLOps_Productionisation/08_Batch_Inference_Pipeline/) | Scalable batch processing workflow | PyTorch DataLoader, Parquet |
| 09 | [Model Monitoring & Drift](Area_3_MLOps_Productionisation/09_Model_Monitoring_Drift/) | Drift detection + Streamlit dashboard | Evidently, Streamlit, Plotly |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Git
- 8 GB RAM minimum (GPU optional but recommended for training projects)

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Quick Start (any project)

```bash
# Navigate to any project
cd Area_1_ML_System_Design/01_Semantic_Search_Engine

# Install all dependencies into an isolated virtualenv
uv sync

# Run tests
uv run pytest

# Launch notebooks
uv run jupyter notebook
```

---

## Dependency Graph

Projects build on each other — recommended order:

```
Phase 1 (Foundation)
├── 04_Text_Classification_BERT   ← produces trained model artifacts
└── 01_Semantic_Search_Engine     ← establishes embedding + retrieval patterns

Phase 2 (Extend)
├── 05_Sentence_Embedding_Training  ← extends training patterns from 04
├── 02_Document_QA_RAG              ← extends retrieval patterns from 01
└── 03_NLP_Entity_Extraction        ← standalone

Phase 3 (Advanced)
└── 06_Learning_To_Rank             ← builds on embedding knowledge

Phase 4 (MLOps — uses models from Phases 1-3)
├── 07_ML_Model_REST_API            ← serves model from 04
├── 08_Batch_Inference_Pipeline     ← extends inference from 07
└── 09_Model_Monitoring_Drift       ← monitors output of 07 + 08
```

---

## JD Coverage Map

| JD Requirement | Projects |
|----------------|----------|
| Semantic retrieval & vector databases | 01, 02 |
| Question-answering systems | 02 |
| NLP pipelines, NER, document parsing | 03 |
| Text understanding & entity extraction | 03 |
| Classical ML + deep learning training | 04, 05 |
| Embedding models & transformers | 04, 05 |
| Ranking models | 06 |
| Hyperparameter tuning & optimization | 04, 05, 06 |
| Model deployment & scalable inference | 07, 08 |
| MLOps, model monitoring, drift detection | 09 |
| APIs & batch processing workflows | 07, 08 |

---

## Tech Stack Summary

| Library | Projects | Purpose |
|---------|----------|---------|
| `transformers` | 03, 04, 06, 07, 08 | Model loading, fine-tuning, inference |
| `sentence-transformers` | 01, 02, 05, 06 | Embedding models |
| `torch` | 04, 05, 06, 07, 08 | Deep learning framework |
| `faiss-cpu` | 01, 06 | Fast vector similarity search |
| `chromadb` | 01, 02 | Managed vector database |
| `langchain` | 02 | RAG orchestration |
| `spacy` | 03 | Industrial NLP |
| `lightgbm` | 06 | Gradient boosting / LambdaMART |
| `fastapi` | 07 | REST API framework |
| `evidently` | 09 | ML monitoring & drift detection |
| `streamlit` | 09 | Dashboard UI |
