# NLP Entity Extraction Pipeline

A production NER pipeline that extracts named entities from PDF, DOCX, and TXT documents using HuggingFace Transformers and spaCy, with entity aggregation, relation extraction, and structured JSON/CSV output.

---

## Architecture

```
Input Documents (PDF / DOCX / TXT)
    │
    ▼
┌──────────────────┐
│ DocumentParser   │  pypdf (PDF), python-docx (DOCX), plain text
└────────┬─────────┘
         │ pages as strings + metadata
         ▼
┌──────────────────┐
│   NER Model      │  HuggingFace dslim/bert-base-NER (primary)
│                  │  spaCy en_core_web_sm (fallback / extended types)
└────────┬─────────┘
         │ raw entities [{text, label, start, end, score}]
         ▼
┌──────────────────┐
│ EntityAggregator │  dedup, majority-vote labelling, rank by freq+conf
└────────┬─────────┘
         │ canonical entities with counts
         ▼
┌──────────────────┐
│RelationExtractor │  co-occurrence matrix + optional RE model
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ OutputFormatter  │  JSON / flat CSV / spaCy displacy HTML
└──────────────────┘
```

---

## Entity Types Supported

| Label | Meaning | Example |
|-------|---------|---------|
| PER | Person | Elon Musk |
| ORG | Organisation | Google DeepMind |
| LOC | Location | San Francisco |
| MISC | Miscellaneous | Python (language) |
| DATE | Date/time | January 2024 |
| MONEY | Monetary value | $1.5 billion |
| GPE | Geopolitical entity | United States |
| PRODUCT | Product name | iPhone 15 |

*PER/ORG/LOC/MISC from BERT-NER; extended types from spaCy OntoNotes.*

---

## Dataset

| Dataset | Entity Types | Size | Use |
|---------|-------------|------|-----|
| CoNLL-2003 | 4 types | 14k sentences | Evaluation |
| OntoNotes 5.0 | 18 types | 76k sentences | Extended evaluation |
| Reuters-21578 | - | 11k articles | Demo corpus |

See [data.txt](data.txt) for download links.

---

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Python dependencies
uv sync

# 3. Install spaCy English model
uv run python -m spacy download en_core_web_sm
```

---

## Usage

### Extract entities from a single file

```bash
uv run python scripts/extract_entities.py --file data/documents/report.pdf --output results.json
```

### Batch process a directory

```bash
uv run python scripts/batch_process.py --dir data/documents/ --output-dir results/
```

### Example JSON output

```json
{
  "source": "annual_report.pdf",
  "entities": [
    {"text": "Apple Inc.", "label": "ORG", "count": 23, "confidence": 0.97},
    {"text": "Tim Cook", "label": "PER", "count": 8, "confidence": 0.99},
    {"text": "Cupertino", "label": "LOC", "count": 5, "confidence": 0.94}
  ],
  "relations": [
    {"entity1": "Tim Cook", "entity2": "Apple Inc.", "relation": "co-occurrence", "count": 7}
  ]
}
```

### Run tests & evaluation

```bash
uv run pytest
# F1 evaluation on CoNLL-2003:
uv run python scripts/extract_entities.py --eval-conll
```

---

## Results (CoNLL-2003 Test Set)

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| dslim/bert-base-NER | 91.3 | 90.8 | 91.8 |
| spaCy en_core_web_sm | 85.1 | 84.2 | 86.0 |
| DistilBERT-NER | 89.6 | 89.1 | 90.1 |

---

## Project Structure

```
03_NLP_Entity_Extraction/
├── pyproject.toml
├── .python-version
├── README.md
├── data.txt
├── src/
│   ├── ner_model.py           # HF + spaCy NER, unified Entity output
│   ├── document_parser.py     # PDF, DOCX, TXT loading
│   ├── entity_aggregator.py   # Dedup + canonical entity ranking
│   ├── relation_extractor.py  # Co-occurrence + RE
│   └── output_formatter.py    # JSON, CSV, HTML export
├── scripts/
│   ├── extract_entities.py    # CLI: single file
│   └── batch_process.py       # CLI: directory batch
├── notebooks/
│   ├── 01_ner_exploration.ipynb
│   └── 02_entity_analysis.ipynb
├── tests/
│   ├── test_ner_model.py
│   ├── test_document_parser.py
│   └── test_entity_aggregator.py
└── data/
    └── documents/             # Place input files here
```

---

## Design Decisions

**HuggingFace primary, spaCy fallback**
BERT-NER scores ~91 F1 on CoNLL-2003. spaCy is used for additional entity types (DATE, MONEY, CARDINAL) not in the HF model's label set.

**Entity aggregation with majority vote**
The same surface form (e.g., "Apple") may be tagged differently across contexts. Aggregation groups by normalised text, picks the label with the highest frequency × confidence product.

---

## Future Improvements

- Add coreference resolution (link "he" → "Tim Cook")
- Fine-tune on domain-specific data (medical, legal, financial)
- Expose as REST API endpoint (see Project 07)
