# Random Forest Production Project

## Project Overview
**Algorithm:** Random Forest  
**Objective:** Build a production-ready fraud detection system

---

## Dataset
- **Source:** [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features:** Transaction amount, time, V1-V28 (PCA features)
- **Target:** Binary classification (Fraud/Not Fraud)

---

## Project Structure

```
Project_Fraud_Detection/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/        # Random Forest model
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
pip install -r requirements.txt
python -m src.pipelines.training_pipeline
pytest tests/
```

---

## Key Considerations

- Handle class imbalance
- Feature importance analysis
- Model interpretability with SHAP
- Real-time prediction requirements
