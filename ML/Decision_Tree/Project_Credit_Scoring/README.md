# Credit Scoring Model - Decision Tree Project

## Project Overview

**Project Title:** Credit Scoring Prediction Model  
**Algorithm:** Decision Tree Classifier  
**Objective:** Build a production-ready credit scoring system to predict loan approval/rejection based on applicant financial data

---

## 📁 Project Structure

```
Project_Credit_Scoring/
├── src/
│   ├── data/
│   │   └── data_loader.py         # Data loading and preprocessing
│   ├── models/
│   │   └── model.py               # Decision Tree model definition
│   ├── pipelines/
│   │   └── training_pipeline.py   # End-to-end training pipeline
│   └── utils/
│       └── logger.py              # Logging configuration
├── tests/
│   └── test_model.py              # Unit tests
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis
├── config/
│   └── settings.yaml              # Configuration parameters
├── azure/
│   └── azure_deployment.md        # Azure deployment guide
├── docs/
│   └── production_issues.md       # Production issues & solutions
└── requirements.txt               # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure Account (for cloud deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Project_Credit_Scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Run the training pipeline
python -m src.pipelines.training_pipeline

# Run tests
pytest tests/
```

---

## 📊 Dataset

**Source:** [Kaggle - Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

**Features:**
- `age`: Applicant age
- `income`: Annual income
- `credit_score`: Credit score
- `employment_years`: Years of employment
- `loan_amount`: Requested loan amount

**Target:** `default` (1 = Default, 0 = No Default)

---

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

```yaml
model:
  max_depth: 5
  min_samples_split: 10
  min_samples_leaf: 5
  criterion: "gini"

data:
  test_size: 0.2
  random_state: 42
  target_column: "default"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 💡 Code Explanation (High Level)

### 1. Data Loading (`src/data/data_loader.py`)
- Loads data from CSV/Excel
- Handles missing values
- Performs feature engineering
- Splits data into train/test sets

### 2. Model (`src/models/model.py`)
- Defines Decision Tree Classifier
- Hyperparameter tuning
- Model serialization

### 3. Training Pipeline (`src/pipelines/training_pipeline.py`)
- Orchestrates the entire ML workflow:
  1. Load data
  2. Preprocess features
  3. Train model
  4. Evaluate performance
  5. Save model
  6. Generate reports

### 4. Logger (`src/utils/logger.py`)
- Centralized logging
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- File and console output

---

## 🔨 Azure Integration

### Step-by-Step Azure Deployment

1. **Create Azure ML Workspace**
   ```bash
   az ml workspace create -w <workspace-name> -g <resource-group>
   ```

2. **Create Compute Target**
   ```bash
   az ml compute create -n cpu-cluster --type cpu --size STANDARD_DS2_V2
   ```

3. **Submit Training Job**
   ```bash
   az ml job create -f azure/pipeline.yml --workspace-name <workspace>
   ```

4. **Deploy Model**
   ```bash
   az ml model deploy -n credit-scoring-model -m model:1 --ic inference_config.yml
   ```

5. **Create Visualizations**
   - Use Azure ML Studio for experiment tracking
   - View metrics in Azure Dashboard
   - Create Power BI reports from predictions

**Detailed Guide:** See [`azure/azure_deployment.md`](azure/azure_deployment.md)

---

## ⚠️ Production Issues & Solutions

See [`docs/production_issues.md`](docs/production_issues.md) for:
- Data drift handling
- Model retraining strategies
- API latency optimization
- Logging best practices

---

## 📝 Logging

The project uses structured logging:

```python
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.debug(f"Training data shape: {X_train.shape}")
logger.warning("Missing values found in column 'income'")
logger.error("Model failed to save")
```

**Log Levels:**
- DEBUG: Detailed diagnostic information
- INFO: Confirmation that things are working as expected
- WARNING: Something unexpected happened, but program still works
- ERROR: Serious problem, program failed to perform
- CRITICAL: Very serious error, program may crash

---

## 📄 License

MIT License

---

## 👤 Author

Your Name - [your-email@example.com]

---

## 🙏 Acknowledgments

- Kaggle for the dataset
- Azure for cloud infrastructure
- scikit-learn for ML algorithms