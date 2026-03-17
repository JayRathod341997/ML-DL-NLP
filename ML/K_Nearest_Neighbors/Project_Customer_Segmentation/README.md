# K-Nearest Neighbors Production Project

## Project Overview
**Algorithm:** K-Nearest Neighbors (KNN)  
**Objective:** Build a production-ready customer classification system

---

## Dataset
- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Alternative:** Custom customer data
- **Features:** Customer demographics, transaction history
- **Target:** Customer classification (e.g., high-value, medium-value, low-value)

---

## Project Structure

```
Project_Customer_Segmentation/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/        # KNN model implementation
│   ├── pipelines/     # Training pipeline
│   └── utils/        # Logger & config
├── tests/            # Unit tests
├── config/           # YAML configurations
├── docs/             # Production issues & system design
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python -m src.pipelines.training_pipeline

# Run tests
pytest tests/
```

---

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
model:
  name: "KNeighborsClassifier"
  params:
    n_neighbors: 5
    weights: "distance"
    metric: "minkowski"
```

---

## Logging

Logs are saved to `logs/` directory with:
- Training progress
- Model metrics
- Errors and warnings

---

## Production Issues

See `docs/production_issues.md` for:
- Data drift detection
- Performance optimization
- Error handling

---

## Interview Questions

See `docs/system_design.md` for:
- Architecture diagrams
- Follow-up patterns
- Q&A from interview perspective
