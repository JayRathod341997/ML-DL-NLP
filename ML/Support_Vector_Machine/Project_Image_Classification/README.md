# Support Vector Machine - Project Image Classification

## Project Overview

This project implements a Support Vector Machine (SVM) classifier for image classification tasks. The model is designed to classify images into different categories using kernel-based SVM algorithms.

## Project Structure

```
Project_Image_Classification/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── training_pipeline.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── docs/
│   └── production_issues.md
├── azure/
│   └── azure_deployment.md
└── tests/
```

## Features

- Support Vector Machine classification with multiple kernel options (linear, RBF, polynomial)
- Hyperparameter tuning using GridSearchCV
- Image preprocessing and feature extraction
- Model evaluation with comprehensive metrics
- Cross-validation for robust performance estimation
- Azure ML integration for cloud deployment

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```python
from src.pipelines.training_pipeline import train_model
from src.utils.config import load_config

config = load_config('config/settings.yaml')
model = train_model(config)
```

### Making Predictions

```python
from src.models.model import SVMClassifier

classifier = SVMClassifier.load('models/svm_model.pkl')
predictions = classifier.predict(test_images)
```

## Configuration

Edit `config/settings.yaml` to customize:
- Kernel type and parameters
- Training data paths
- Model hyperparameters
- Evaluation metrics

## Azure Deployment

See `azure/azure_deployment.md` for instructions on deploying the model to Azure.

## License

MIT License
