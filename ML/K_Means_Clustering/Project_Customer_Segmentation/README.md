# K-Means Clustering - Project Customer Segmentation

## Project Overview

This project implements K-Means clustering algorithm for customer segmentation. The model clusters customers based on their purchasing behavior and demographic features to enable targeted marketing strategies.

## Project Structure

```
Project_Customer_Segmentation/
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

- K-Means clustering implementation
- Optimal K selection using Elbow method and Silhouette score
- Feature preprocessing and scaling
- Cluster visualization
- Customer profiling based on clusters
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
from src.models.model import KMeansClustering

clustering = KMeansClustering.load('models/kmeans_model.pkl')
clusters = clustering.predict(customer_data)
```

## Configuration

Edit `config/settings.yaml` to customize:
- Number of clusters (K)
- Initialization method
- Maximum iterations
- Feature columns for clustering

## Azure Deployment

See `azure/azure_deployment.md` for instructions on deploying the model to Azure.

## License

MIT License
