# Linear Regression Production Project

## Project Overview

**Project Title:** House Price Prediction Model  
**Algorithm:** Linear Regression  
**Objective:** Build a production-ready ML system to predict house prices based on various features

---

## 📁 Project Structure

```
Linear_Regression/
├── Project_House_Prices/
│   ├── src/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   └── data_loader.py         # Data loading and preprocessing
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── model.py               # Linear Regression model
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   └── training_pipeline.py   # End-to-end training pipeline
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   └── metrics.py             # Model evaluation metrics
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logger.py              # Logging configuration
│   │       └── config.py              # Configuration management
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_model.py              # Unit tests
│   ├── notebooks/
│   │   └── EDA.ipynb                  # Exploratory Data Analysis
│   ├── config/
│   │   └── settings.yaml              # Configuration parameters
│   ├── docs/
│   │   ├── production_issues.md       # Production issues & solutions
│   │   └── system_design.md           # System design documentation
│   ├── data/
│   │   └── raw/                       # Raw data storage
│   ├── logs/                          # Log files
│   ├── models/                        # Saved models
│   ├── requirements.txt               # Python dependencies
│   ├── setup.py                       # Package setup
│   ├── Makefile                       # Build automation
│   └── README.md                      # Setup instructions
├── README.md
├── interview_questions.md
├── quiz.md
└── toy_example.md
```

---

## 📊 Dataset

### Primary Dataset: House Prices

| Attribute | Description |
|-----------|-------------|
| **Source** | [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) |
| **Alternative** | [scikit-learn datasets](https://scikit-learn.org/stable/datasets/real_world.html#boston-dataset) |
| **Samples** | ~1,460 training samples |
| **Features** | 80+ features (numerical & categorical) |
| **Target** | SalePrice (continuous) |

### Dataset Variables

- **Numerical Features:** LotArea, YearBuilt, GrLivArea, TotalBsmtSF, GarageArea, etc.
- **Categorical Features:** MSZoning, Street, Neighborhood, SaleCondition, etc.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Installation

```bash
# Navigate to project directory
cd Linear_Regression/Project_House_Prices

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Run data preprocessing
python -m src.data.data_loader

# Train the model
python -m src.pipelines.training_pipeline

# Run tests
python -m pytest tests/

# View logs
tail -f logs/training_$(date +%Y%m%d).log
```

---

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

```yaml
model:
  name: "LinearRegression"
  params:
    fit_intercept: true
    normalize: false
    alpha: 1.0

data:
  train_path: "data/raw/train.csv"
  test_path: "data/raw/test.csv"
  target_column: "SalePrice"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 📝 Project Components

### 1. Data Loading (`src/data/data_loader.py`)

- Automated data download from URL
- Missing value handling
- Feature engineering
- Train/test splitting

### 2. Model Training (`src/models/model.py`)

- Linear Regression implementation
- Regularization options (Ridge, Lasso)
- Model persistence

### 3. Training Pipeline (`src/pipelines/training_pipeline.py`)

- End-to-end training workflow
- Cross-validation
- Model evaluation

### 4. Logging (`src/utils/logger.py`)

- File and console logging
- Debug information tracking
- Error reporting

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py -v

# Generate coverage report
pytest --cov=src tests/
```

---

## 📈 Model Performance

Expected metrics on test set:

| Metric | Target Value |
|--------|--------------|
| R² Score | > 0.7 |
| RMSE | < $30,000 |
| MAE | < $20,000 |

---

## 🔄 CI/CD Pipeline

The project includes a Makefile with common targets:

```bash
make install    # Install dependencies
make train      # Train model
make test       # Run tests
make clean      # Clean temporary files
make deploy     # Deploy model (requires cloud setup)
```

---

## 📄 License

MIT License

---

## 👤 Author

ML Learning Project Series
