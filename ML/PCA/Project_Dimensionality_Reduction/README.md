# PCA - Project Dimensionality Reduction

## Project Overview

This project implements Principal Component Analysis (PCA) for dimensionality reduction on high-dimensional datasets.

## Project Structure

```
Project_Dimensionality_Reduction/
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

- PCA for dimensionality reduction
- Optimal component selection using explained variance
- Data visualization
- Integration with other ML models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.pipelines.training_pipeline import reduce_dimensions
reduce_dimensions()
```

## License

MIT License
