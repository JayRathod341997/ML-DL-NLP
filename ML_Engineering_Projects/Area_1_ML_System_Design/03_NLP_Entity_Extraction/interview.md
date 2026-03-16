# NLP Entity Extraction - Interview Preparation Guide

> Comprehensive interview guide for ML Engineers covering production NER systems, transformer-based entity extraction, and MLOps considerations.

---

## Table of Contents

1. [Production-Level Interview Questions](#1-production-level-interview-questions)
2. [System Design Discussions](#2-system-design-discussions)
3. [Common Bugs and Solutions](#3-common-bugs-and-possible-solutions)
4. [Deployment Strategies](#4-deployment-strategies)
   - [Azure Deployment](#azure-deployment)
   - [AWS Deployment](#aws-deployment)
5. [Post-Production Issues](#5-post-production-issues)
6. [Visualizations](#6-visualizations)
7. [General ML/MLOps Topics](#7-general-mlmlops-topics)

---

## 1. Production-Level Interview Questions

### 1.1 NER Fundamentals

**Q1: What is Named Entity Recognition (NER) and why is it important?**

Named Entity Recognition is a subtask of Information Extraction that identifies and classifies named entities in unstructured text into predefined categories such as persons, organizations, locations, dates, monetary values, etc.

**Why important:**
- Powers search engines for entity-centric retrieval
- Enables knowledge graph construction
- Critical for question answering systems
- Foundation for relation extraction
- Used in document summarization and indexing

**Example categories:**
```
┌─────────────────────────────────────────────────────────┐
│  ENTITY TYPES                                          │
├─────────────┬───────────────────────────────────────────┤
│ PER         │ Person names (Elon Musk, Barack Obama)   │
│ ORG         │ Organizations (Google, NASA)             │
│ LOC         │ Locations (San Francisco, Mount Everest) │
│ DATE        │ Dates (January 2024, 2 weeks ago)        │
│ TIME        │ Time expressions (2:30 PM, three hours)  │
│ MONEY       │ Monetary values ($1.5B, €500)            │
│ PERCENT     │ Percentages (20%, fifty percent)         │
│ GPE         │ Geopolitical entities (United States)    │
│ PRODUCT     │ Products (iPhone 15, Tesla Model 3)     │
│ EVENT       │ Events (World War II, Olympics 2024)     │
│ WORK_OF_ART │ Artworks (Mona Lisa, Hamlet)            │
│ LAW         │ Laws (GDPR, First Amendment)            │
└─────────────┴───────────────────────────────────────────┘
```

---

**Q2: Explain the difference between rule-based, statistical, and neural NER approaches.**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    NER APPROACH COMPARISON                             │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│ Approach         │ Examples         │ Characteristics                  │
├──────────────────┼──────────────────┼──────────────────────────────────┤
│ Rule-Based       │ regex, gazetteers│ High precision, low recall,     │
│                  │                  │ requires domain expertise,       │
│                  │                  │ hard to scale to new domains     │
├──────────────────┼──────────────────┼──────────────────────────────────┤
│ Statistical      │ CRF, HMM,        │ Good balance, requires          │
│                  │ Maximum Entropy  │ feature engineering, moderate   │
│                  │                  │ training data                    │
├──────────────────┼──────────────────┼──────────────────────────────────┤
│ Neural          │ BiLSTM-CRF,      │ State-of-the-art, learns         │
│                 │ Transformer-BERT │ features automatically, needs    │
│                 │                  │ large datasets, end-to-end       │
└──────────────────┴──────────────────┴──────────────────────────────────┘
```

**Key insight:** Modern production systems often combine approaches (hybrid) for best results.

---

**Q3: How does a BiLSTM-CRF model work for NER?**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BiLSTM-CRF ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────┘

Input: "Elon Musk founded SpaceX"
       │      │    │      │
       ▼      ▼    ▼      ▼
   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   │ E   │ │ l   │ │ o   │ │ n   │  ← Character Embeddings
   │ 5   │ │ 5   │ │ 5   │ │ 5   │
   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
      │       │       │       │
      └───────┴───────┴───────┘
              │
              ▼
   ┌─────────────────────┐
   │   Word Embeddings   │  ← Pre-trained (BERT, GloVe, FastText)
   │   (512-dim)         │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   BiLSTM Forward    │  ← Captures context from left
   └──────────┬──────────┘
              │
   ┌──────────┴──────────┐
   │                      │
   ▼                      ▼
┌──────────┐         ┌──────────┐
│ BiLSTM   │         │ BiLSTM   │
│ Backward │         │ Forward  │
└────┬─────┘         └────┬─────┘
     │                    │
     └────────┬───────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Concatenated       │  ← Full context representation
   │  Context Vectors   │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   CRF Layer         │  ← Models label dependencies
   │                     │  • B-PER → I-PER (valid)
   │   -2.3  0.1  0.0    │  • O   → B-PER (valid)
   │    0.0 -1.5  0.2    │  • O   → I-PER (invalid!)
   │    ...              │
   └──────────┬──────────┘
              │
              ▼
Output: B-PER I-PER O O B-ORG I-ORG
        (Elon) (Musk) (founded) (SpaceX)
```

**Why CRF matters:**
- Prevents invalid label sequences (e.g., I-PER without B-PER)
- Models transition probabilities between labels
- Global optimization instead of local classification

---

**Q4: What are the advantages of using transformer-based models (BERT) for NER?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    BERT vs BiLSTM-CRF                                  │
├─────────────────────────────┬──────────────────────────────────────────┤
│ Aspect                       │ BERT Advantage                          │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Context Understanding         │ True bidirectional context            │
│                               │ "Apple" (fruit) vs "Apple" (company)  │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Pre-training                 │ Leverages massive unlabeled data       │
│                               │ MLM + NSP pre-training objectives     │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Transfer Learning            │ Fine-tune with small labeled datasets  │
│                               │ Domain adaptation made easy           │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Parallelization              │ Attention is parallel O(1) vs LSTM O(n)│
│                               │ Much faster training/inference         │
├─────────────────────────────┼──────────────────────────────────────────┤
│ State-of-the-art             │ 2-5% F1 improvement over BiLSTM-CRF    │
│                               │ CoNLL-2003: 92.4% BERT vs 90.2% BiLSTM  │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Model Size                   │ Larger (110M params) but more capable   │
└─────────────────────────────┴──────────────────────────────────────────┘
```

---

**Q5: How do you handle entity overlap and nested entities?**

**Problem:** Standard NER assumes non-overlapping, flat entities. But real text has nested entities.

```
Example with nested entities:
"Apple Inc. CEO Tim Cook visited the Apple Store in Cupertino"

Flat NER output (incorrect):
ORG: Apple Inc.
ORG: Apple Store
PER: Tim Cook
LOC: Cupertino

Nested NER output (correct):
ORG: Apple Inc.
  └─TITLE: CEO
PER: Tim Cook
ORG: Apple Store
LOC: Cupertino

Solution approaches:
┌────────────────────────────────────────────────────────────────────────┐
│ APPROACH                    │ METHOD                                   │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Token Classification        │ Use BIO tagging with hierarchical labels │
│ (flat)                     │ e.g., B-ORG-LEVEL1, I-ORG-LEVEL1         │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Multi-layer Classification  │ Predict entity type at each token       │
│                            │ Layer 1: ORG/LOC, Layer 2: CEO/Store    │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Span-based                 │ Enumerate all spans, classify each      │
│                            │ More flexible, but expensive            │
├─────────────────────────────┼──────────────────────────────────────────┤
│ Architecture: BranchNER     │ Main NER + auxiliary classifiers        │
│                            │ for nested entities                      │
└─────────────────────────────┴──────────────────────────────────────────┘
```

---

### 1.2 Model Training & Optimization

**Q6: How would you fine-tune a BERT model for NER on a domain-specific dataset?**

```python
# Complete fine-tuning pipeline for NER

from transformers import (
    BertForTokenClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset

# 1. Load pre-trained model and tokenizer
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),  # e.g., 9 for BIO scheme
    id2label=id2label,
    label2id=label2id
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. Tokenize with alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
        return_offsets_mapping=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First token
            else:
                label_ids.append(label[word_idx])  # Sub-tokens
            previous_word_idx = word_idx
        
        label_ids.append(-100)  # Padding
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 3. Training arguments
training_args = TrainingArguments(
    output_dir="./ner_bert",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    fp16=True,
    dataloader_num_workers=4,
)

# 4. Data collator handles padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

**Q7: What are the key hyperparameters to tune when fine-tuning BERT for NER?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    BERT NER HYPERPARAMETERS                           │
├─────────────────────────┬─────────────────────────────────────────────┤
│ Parameter               │ Recommended Range         │ Impact           │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Learning Rate          │ 1e-5 to 5e-5              │ Lower = stable  │
│                        │ (2e-5 typical)            │ Higher = fast   │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Batch Size             │ 8-32                      │ Memory vs       │
│                        │ (16 typical)              │ speed tradeoff  │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Epochs                │ 3-10                      │ Overfitting     │
│                        │ (3-5 typical)             │ risk            │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Warmup Ratio           │ 0.1-0.2                   │ Stabilizes      │
│                        │                           │ early training  │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Weight Decay           │ 0.01-0.1                  │ Regularization  │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Max Sequence Length   │ 128-512                   │ Memory &        │
│                        │ (128 typical for NER)    │ context         │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ AdamW Epsilon          │ 1e-8                      │ Numerical       │
│                        │                           │ stability       │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Gradient Clipping      │ 1.0                       │ Prevents        │
│                        │                           │ exploding       │
├─────────────────────────┼───────────────────────────┼─────────────────┤
│ Mixed Precision (fp16)│ True if GPU               │ 2x speed,       │
│                        │ supports it               │ less memory     │
└─────────────────────────┴───────────────────────────┴─────────────────┘
```

---

**Q8: How do you handle class imbalance in NER datasets?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    CLASS IMBALANCE IN NER                              │
└────────────────────────────────────────────────────────────────────────┘

Typical distribution in NER datasets:
┌────────────────────────────────────────────────────────────────────────┐
│  Label   │  CoNLL-2003    │  OntoNotes 5.0    │  Impact               │
├──────────┼────────────────┼───────────────────┼───────────────────────┤
│  O       │  ~88%          │  ~85%             │  Majority class       │
│  B-LOC   │  ~3%           │  ~2%              │  Minority             │
│  I-LOC   │  ~2%           │  ~1.5%            │  Minority             │
│  B-PER   │  ~2%           │  ~3%              │  Minority             │
│  B-ORG   │  ~1.5%         │  ~2%              │  Minority             │
│  ...     │  ...           │  ...              │  ...                  │
└────────────────────────────────────────────────────────────────────────┘

SOLUTION STRATEGIES:

1. WEIGHTED LOSS
   from sklearn.utils.class_weight import compute_class_weight
   
   weights = compute_class_weight(
       class_weight='balanced',
       classes=np.unique(all_labels),
       y=all_labels
   )
   loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))

2. FOCAL LOSS
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma
       
       def forward(self, inputs, targets):
           ce_loss = F.cross_entropy(inputs, targets)
           pt = torch.exp(-ce_loss)
           focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
           return focal_loss

3. DATA AUGMENTATION
   - Back-translation (EN→FR→EN)
   - Synonym replacement
   - Random token masking
   - Entity replacement
```

---

### 1.3 Evaluation Metrics

**Q9: What metrics do you use to evaluate NER models?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    NER EVALUATION METRICS                              │
└────────────────────────────────────────────────────────────────────────┘

1. STRICT (EXACT MATCH)
   - Entity must match exactly (start, end, type)
   - Most strict, penalizes partial matches

2. PARTIAL MATCH
   - Partial overlap counts as correct

3. EXACT + TYPE
   - Type must match, boundaries can be partial
   Most common in production

4. TOKEN-LEVEL
   - Evaluate each token independently

Python Implementation:
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

print(classification_report(y_true, y_pred))
```

---

### 1.4 Advanced NER Questions

**Q10: What is the difference between NER and Named Entity Linking (NEL)?**

NER identifies and classifies entities, while NEL disambiguates entities to a knowledge base.

```
Text: "Apple announced new features"

NER Output:
- "Apple" → ORG (Named Entity Recognition)

NEL Output:
- "Apple" → ORG → Knowledge Base ID: Apple Inc. (NASDAQ: AAPL)
- Links to: https://en.wikipedia.org/wiki/Apple_Inc.
```

**Q11: How would you handle multi-lingual NER?**

- Use multilingual models (mBERT, XLM-RoBERTa)
- Train language-specific models for high-resource languages
- Use language detection as preprocessing

**Q12: What is the role of gazetteers in NER?**

Gazetteers are curated lists of entities. They provide high precision for known entities but poor recall for new entities. They're often used as features in hybrid systems.

**Q13: Explain the BIO tagging scheme.**

- B-XXX: Beginning of an entity
- I-XXX: Inside/continuation of an entity
- O: Outside (no entity)

```
"Elon Musk founded SpaceX"
B-PER I-PER O O B-ORG I-ORG
```

---

## 2. System Design Discussions

### 2.1 High-Level Architecture

**Q14: Design a production NER system architecture.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION NER SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Client Apps    │
                              │ (Web, Mobile)   │
                              └────────┬────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │   Load Balancer / API   │
                         │        Gateway           │
                         └────────┬────────────────┘
                                  │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          ┌────────────┐   ┌────────────┐   ┌────────────┐
          │ API Pod 1  │   │ API Pod 2  │   │ API Pod N  │
          │ (FastAPI)  │   │ (FastAPI)  │   │ (FastAPI)  │
          └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────────┐
                 │         INFERENCE LAYER               │
                 │  ┌─────────────────────────────────┐  │
                 │  │  Model Server (TorchServe/      │  │
                 │  │  Triton/ONNX Runtime)           │  │
                 │  │  - GPU acceleration             │  │
                 │  │  - Batch processing             │  │
                 │  │  - Model versioning             │  │
                 │  └─────────────────────────────────┘  │
                 └───────────────────────────────────────┘
                                   │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │ Blob/S3     │         │ Redis Cache │         │ Database    │
   │ Storage     │         │ (Embeddings)│         │ (Entities)  │
   └─────────────┘         └─────────────┘         └─────────────┘
```

---

### 2.2 Document Processing Pipeline

**Q15: How would you design a document processing pipeline?**

```
Document Ingestion Flow:
────────────────────────

Input → Validation → Text Extraction → NER → Entity Post-processing → Storage
             │              │               │            │                │
             ▼              ▼               ▼            ▼                ▼
         [Type/MD5]     [PDF/DOCX]    [BERT-NER]   [Dedupe]       [DB/Elasticsearch]
```

**Key Components:**
- Document parsers for PDF, DOCX, TXT
- Text cleaning and normalization
- Batch processing for large volumes
- Error handling and retry logic

---

### 2.3 Hybrid NER Architecture

**Q16: How would you combine multiple NER models?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    HYBRID NER ROUTING                                   │
└────────────────────────────────────────────────────────────────────────┘

        Input Text
            │
            ▼
    ┌───────────────────┐
    │  Routing Layer     │
    │  - Entity type    │
    │    detection      │
    └─────────┬──────────┘
              │
    ┌─────────┼─────────┬─────────┬─────────┐
    ▼         ▼         ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌────────┐
│ BERT- │ │ spaCy │ │ Custom│ │ Rules │ │Gazetteer│
│ NER   │ │ NER   │ │Domain │ │       │ │Lookup  │
│ (PER, │ │(DATE, │ │(Medical)│ │(NUM) │ │(ORG)   │
│ ORG)  │ │ MONEY)│ │       │ │       │ │        │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬────┘
    └─────────┴────────┴────────┴─────────┘
              │
              ▼
      Entity Merger
      - Merge overlapping
      - Resolve conflicts
      - Deduplicate
              │
              ▼
        Final Entities
```

---

## 3. Common Bugs and Solutions

### 3.1 Tokenization Issues

**Bug: Subword Tokenization Causing Label Mismatch**

```
Problem:
Original: "Elon Musk"
Tokens:   [Elon, Musk]
Labels:   [B-PER, I-PER] ✓

But BERT tokenizer: [El, ##on, Mus, ##k]
Wrong labels would be applied to subwords!

Solution: Use word_ids mapping

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True)
    labels = []
    word_ids = tokenized.word_ids()
    
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != previous_word_id:
            labels.append(examples["labels"][word_id])
        else:
            labels.append(-100)  # Ignore subwords
        previous_word_id = word_id
    
    return {"labels": labels}

PROS: ✓ Correct alignment, ✓ CRF compatible
CONS: ✗ More complex code
```

---

### 3.2 Memory Issues

**Bug: GPU Out-of-Memory**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    OOM SOLUTIONS                                        │
├────────────────────────────────────────────────────────────────────────┤
│ Solution             │ Memory Impact │ Speed Impact │ Complexity      │
├──────────────────────┼───────────────┼──────────────┼────────────────┤
│ Reduce batch size    │ High          │ Medium        │ Low            │
│ Gradient accumulation│ Medium        │ Low           │ Low            │
│ Mixed precision      │ High          │ High          │ Low            │
│ Gradient checkpoint  │ High          │ Medium        │ Medium         │
│ Reduce seq length    │ High          │ High          │ Low            │
│ Use DistilBERT       │ High          │ High          │ Low            │
└──────────────────────┴───────────────┴──────────────┴────────────────┘
```

---

### 3.3 Quality Issues

**Bug: Entity Overlap in Results**

```python
# Solution: Confidence-based resolution

def resolve_entity_overlap(entities):
    sorted_entities = sorted(entities, key=lambda x: x["score"], reverse=True)
    resolved = []
    
    for entity in sorted_entities:
        is_overlapping = False
        for existing in resolved:
            if entities_overlap(entity, existing):
                is_overlapping = True
                break
        if not is_overlapping:
            resolved.append(entity)
    
    return resolved
```

---

## 4. Deployment Strategies

### 4.1 Azure Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AZURE DEPLOYMENT ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Azure Front    │
                    │  Door (CDN)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ Azure API        │
                    │ Management (APIM) │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │ AKS East US│     │ AKS West EU│     │ AKS Asia   │
   │            │     │            │     │            │
   └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────────────┐
         │          INFERENCE LAYER                     │
         │  Azure ML Endpoints (GPU: NCasT4_v3)      │
         └─────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │Blob Store│      │Cosmos DB │      │Redis     │
   └──────────┘      └──────────┘      └──────────┘
```

**Azure Services:**
- Compute: AKS (Azure Kubernetes Service)
- Model Serving: Azure ML Endpoints
- API Gateway: Azure API Management
- Storage: Azure Blob Storage
- Database: Azure Cosmos DB
- Cache: Azure Cache for Redis

---

### 4.2 AWS Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AWS DEPLOYMENT ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  CloudFront     │
                    │  (CDN)          │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ API Gateway     │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │ ECS Fargate│     │ ECS Fargate│     │ ECS Fargate│
   │ us-east-1  │     │ eu-west-1  │     │ ap-south-1 │
   └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────────────┐
         │          INFERENCE LAYER                     │
         │  SageMaker Endpoints (GPU: g4dn, p3)        │
         └─────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │ S3        │      │ DynamoDB  │      │ElastiCache│
   └──────────┘      └──────────┘      └──────────┘
```

**AWS Services:**
- Compute: ECS Fargate / EKS
- Model Serving: SageMaker Endpoints
- API Gateway: API Gateway
- Storage: S3
- Database: DynamoDB
- Cache: ElastiCache for Redis

---

### 4.3 Azure vs AWS Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│                    AZURE vs AWS COMPARISON                               │
├────────────────────────┬────────────────────────┬──────────────────────┤
│ Aspect                │ Azure                  │ AWS                  │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ Kubernetes            │ AKS                    │ EKS                  │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ Model Serving          │ Azure ML Endpoints     │ SageMaker Endpoints  │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ GPU Instances         │ NC series (T4, V100)  │ p3, g4dn (T4, A100)  │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ Serverless Inference  │ Not available          │ SageMaker Serverless │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ Vector Database       │ Azure AI Search        │ OpenSearch (k-NN)    │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ Pricing (GPU)         │ ~$3.06/hr (NC4as_v3)   │ ~$3.06/hr (g4dn.xl)  │
├────────────────────────┼────────────────────────┼──────────────────────┤
│ Best For              │ Microsoft ecosystem    │ AWS ecosystem        │
│                       │ Enterprise workloads   │ ML-native startups   │
└────────────────────────┴────────────────────────┴──────────────────────┘
```

---

## 5. Post-Production Issues

### 5.1 Data Drift

```
┌────────────────────────────────────────────────────────────────────────┐
│                    DATA DRIFT TYPES                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ 1. VOCABULARY DRIFT                                                    │
│    Training: "Apple", "Google" → ORG                                   │
│    Production: "Tesla", "Stripe" → Poor recognition                   │
│                                                                        │
│ 2. CONTEXT DRIFT                                                       │
│    Training: "Apple" = fruit (food blogs)                              │
│    Production: "Apple" = company (news) → Misclassification            │
│                                                                        │
│ 3. LABEL DRIFT                                                         │
│    Old: "COVID" → MISC                                                 │
│    New: "COVID" → DISEASE → Inconsistent labels                       │
│                                                                        │
│ 4. POPULATION DRIFT                                                   │
│    Original: English documents only                                    │
│    Production: Multi-lingual → Lower accuracy                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Detection Methods:**
- KL Divergence on embeddings
- Chi-Square on entity distributions
- Jaccard Similarity on vocabulary

---

### 5.2 Model Degradation

```
┌────────────────────────────────────────────────────────────────────────┐
│                    DEGRADATION PATTERNS                                  │
├────────────────────────────────────────────────────────────────────────┤
│ Metric              │ Detection            │ Solution                 │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Precision Drop      │ Monitor per-type     │ Retrain with            │
│                     │ precision            │ hard negatives          │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Recall Drop         │ Monitor false        │ Add more training       │
│                     │ negatives            │ data                    │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Latency Spike       │ P99 > 500ms          │ Scale up, optimize     │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Error Rate Increase │ 5xx errors           │ Circuit breaker,       │
│                     │                      │ fallback model          │
└─────────────────────┴──────────────────────┴─────────────────────────┘
```

---

### 5.3 Common Issues Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│                    COMMON POST-PRODUCTION ISSUES                       │
├────────────────────────────────────────────────────────────────────────┤
│ Issue              │ Cause              │ Detection     │ Solution    │
├────────────────────┼────────────────────┼───────────────┼─────────────┤
│ Latency Spike      │ GPU throttling     │ P99 > 500ms  │ Scale up    │
│ Memory Leak        │ Unbounded cache    │ Memory growth │ Limit cache │
│ False Positives    │ Overfitting        │ User reports │ Retrain     │
│ Model Downtime     │ Health check fail  │ 503 errors   │ Multi-region│
│ Cost Overrun       │ Idle GPUs          │ Bill increase│ Auto-scale  │
│ Privacy Violation │ PII in entities    │ Audit failure│ PII filter  │
└────────────────────┴────────────────────┴───────────────┴─────────────┘
```

---

## 6. Visualizations

### 6.1 NER Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NER PIPELINE VISUALIZATION                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  INPUT: "Tim Cook announced that Apple will launch iPhone 15 in September"│
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Text Preprocessing                                               │
│  - Tokenization                                                           │
│  - Lowercasing                                                            │
│  - Sentence splitting                                                    │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Model Inference                                                 │
│                                                                           │
│  Tokens:   [Tim] [Cook] [announced] [that] [Apple] [will] ...          │
│  Preds:    [B-PER] [I-PER] [O] [O] [B-ORG] [O] ...                     │
│  Scores:   [0.99] [0.97] [N/A] [N/A] [0.95] [N/A] ...                   │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Post-processing                                                 │
│  - Merge subword predictions                                             │
│  - Filter by confidence threshold                                        │
│  - Resolve overlaps                                                      │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Output                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ {                                                                    │ │
│  │   "entities": [                                                     │ │
│  │     {"text": "Tim Cook", "type": "PER", "score": 0.98},            │ │
│  │     {"text": "Apple", "type": "ORG", "score": 0.95},                 │ │
│  │     {"text": "iPhone 15", "type": "PRODUCT", "score": 0.92},        │ │
│  │     {"text": "September", "type": "DATE", "score": 0.89}            │ │
│  │   ]                                                                  │ │
│  │ }                                                                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### 6.2 Error Analysis Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│                    ERROR ANALYSIS VISUALIZATION                          │
└────────────────────────────────────────────────────────────────────────┘

                    PREDICTED
                  PER   ORG   LOC   DATE   O
              ┌─────┬─────┬─────┬─────┬─────┐
         PER  │ 45  │  3  │  2  │  0  │ 10  │  ← 80% recall for PER
       T ┌─────┼─────┼─────┼─────┼─────┤
       R │ ORG │  2  │ 38  │  1  │  0  │ 19  │  ← 65% recall for ORG
       U ┌─────┼─────┼─────┼─────┼─────┤
       E │ LOC │  1  │  2  │ 32  │  3  │ 22  │  ← 53% recall for LOC
          ├─────┼─────┼─────┼─────┼─────┤
          │DATE │  0  │  1  │  2  │ 28  │ 19  │  ← 56% recall for DATE
          ├─────┼─────┼─────┼─────┼─────┤
          │ O   │  5  │  8  │  4  │  6  │ 850 │
              └─────┴─────┴─────┴─────┴─────┘
                 85%     73%    78%   70%   90%

Analysis:
- PER: High precision, good recall
- ORG: Needs improvement (confused with LOC)
- LOC: Lower recall (missed entities)
- DATE: Misses many date expressions
```

---

## 7. General ML/MLOps Topics

### 7.1 MLOps Best Practices

**Q17: What MLOps practices do you follow for NER models?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MLOPS PIPELINE FOR NER                               │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Data      │   │   Train     │   │  Validate   │   │  Deploy     │
│   Pipeline  │──▶│   Model     │──▶│   Model     │──▶│   Model     │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
      │                 │                 │                 │
      ▼                 ▼                 ▼                 ▼
┌───────────┐     ┌───────────┐    ┌───────────┐     ┌───────────┐
│Data       │     │Hyperparam │    │Metrics    │     │A/B        │
│Versioning │     │Tracking   │    │Comparison │     │Testing    │
│(DVC)      │     │(MLflow)   │    │(Evidently)│     │           │
└───────────┘     └───────────┘    └───────────┘     └───────────┘

Key Components:
- DVC: Data and model versioning
- MLflow: Experiment tracking
- Evidently: Data/model monitoring
- CI/CD: Automated testing
- Feature store: Reusable features
```

---

**Q18: How do you version NER models?**

```
Model Version Strategy:
──────────────────────

v1.0.0-YYYYMMDD
 │ │ └─ Patch (bug fixes)
 │ │   
 │ └─── Minor (new entity types, improvements)
 │
 └────── Major (architecture changes)

Example:
ner-bert-base-v1.0.0.tar.gz
ner-bert-base-v1.1.0.tar.gz  (added DATE entity type)
ner-bert-base-v2.0.0.tar.gz  (switched to DistilBERT)
```

---

### 7.2 Testing ML Systems

**Q19: How do you test NER systems?**

```
┌────────────────────────────────────────────────────────────────────────�│
│                    TESTING STRATEGY                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. UNIT TESTS                                                          │
│     - Individual function tests                                         │
│     - Tokenizer alignment tests                                        │
│     - Entity resolution logic                                          │
│                                                                         │
│  2. INTEGRATION TESTS                                                   │
│     - End-to-end pipeline tests                                        │
│     - API endpoint tests                                               │
│     - Database connection tests                                        │
│                                                                         │
│  3. PERFORMANCE TESTS                                                  │
│     - Latency benchmarks                                               │
│     - Throughput tests                                                 │
│     - Memory usage tests                                               │
│                                                                         │
│  4. SHADOW DEPLOYMENT                                                  │
│     - Run new model in parallel                                        │
│     - Compare outputs                                                  │
│     - No customer impact                                               │
│                                                                         │
│  5. CANARY DEPLOYMENT                                                  │
│     - Route 1% traffic to new model                                    │
│     - Monitor metrics                                                  │
│     - Gradual rollout                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 7.3 Monitoring & Observability

**Q20: What metrics do you monitor for production NER?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    KEY METRICS TO MONITOR                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  BUSINESS METRICS:                                                     │
│  ─────────────────                                                     │
│  - Entities extracted per request                                     │
│  - Unique entities discovered                                          │
│  - Entity type distribution                                            │
│                                                                        │
│  OPERATIONAL METRICS:                                                  │
│  ───────────────────                                                   │
│  - Request latency (P50, P95, P99)                                     │
│  - Error rate (5xx, 4xx)                                               │
│  - Throughput (requests/second)                                        │
│  - GPU utilization                                                    │
│  - Memory usage                                                        │
│                                                                        │
│  MODEL METRICS:                                                        │
│  ──────────────                                                        │
│  - Prediction confidence distribution                                  │
│  - Entity type confidence                                              │
│  - Known vs unknown entities                                           │
│  - Drift detection scores                                             │
│                                                                        │
│  INFRASTRUCTURE METRICS:                                               │
│  ─────────────────────                                                 │
│  - Pod health                                                          │
│  - Database connections                                                │
│  - Cache hit rate                                                      │
│  - Queue depth                                                         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This guide covers the essential topics for ML Engineer interviews focusing on NER and entity extraction systems:

1. **NER Fundamentals**: Understanding of token classification, transformer-based NER, and evaluation metrics
2. **System Design**: Production architecture, scalability, and hybrid approaches
3. **Common Bugs**: Tokenization issues, memory problems, and quality challenges
4. **Deployment**: Azure and AWS cloud deployment strategies
5. **Post-Production**: Drift detection, performance monitoring, and issue resolution
6. **MLOps**: CI/CD, testing, and observability best practices

**Key Takeaways:**
- BERT-based models provide state-of-the-art accuracy
- Hybrid architectures combine multiple NER approaches
- Production systems require careful latency and scaling design
- Continuous monitoring is essential for maintaining model quality
- Both Azure and AWS provide robust ML serving platforms

---

*Interview preparation complete. Good luck!*
