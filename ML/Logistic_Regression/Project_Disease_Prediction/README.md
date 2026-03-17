# Logistic Regression Production Project

## Project Overview
**Algorithm:** Logistic Regression  
**Objective:** Build a production-ready disease prediction system

---

## Dataset
- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Features:** Patient medical history, vital signs
- **Target:** Binary classification (Disease/No Disease)

---

## Project Structure

```
Project_Disease_Prediction/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/        # Logistic Regression model
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

## Interview Questions

See `docs/system_design.md` for comprehensive interview preparation with:
- Architecture diagrams
- Follow-up questions
- Short Q&A format

---

## Production Considerations

- Handle class imbalance with SMOTE
- Feature scaling is essential
- Monitor prediction probabilities
- Regular calibration needed
