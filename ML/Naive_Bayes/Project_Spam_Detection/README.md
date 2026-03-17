# Naive Bayes - Project Spam Detection

## Project Overview

This project implements a Naive Bayes classifier for spam email detection. The model classifies emails as spam or ham (not spam) using text classification techniques.

## Project Structure

```
Project_Spam_Detection/
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

- Naive Bayes classification (Multinomial, Bernoulli, Gaussian)
- Text preprocessing (tokenization, stopwords removal, stemming)
- TF-IDF vectorization
- Model evaluation with comprehensive metrics
- Azure ML integration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.pipelines.training_pipeline import train_model
config = load_config('config/settings.yaml')
model = train_model(config)
```

## Configuration

Edit `config/settings.yaml` to customize model parameters and data paths.

## License

MIT License
