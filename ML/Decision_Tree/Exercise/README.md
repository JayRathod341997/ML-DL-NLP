# Decision Tree Exercise - Credit Risk Assessment

---

## 📚 Exercise Overview

This exercise will help you implement a Decision Tree classifier for a real-world credit risk assessment problem.

---

## 🎯 Learning Objectives

By the end of this exercise, you will:
- Understand how Decision Trees work for classification
- Learn to handle imbalanced datasets
- Apply hyperparameter tuning
- Evaluate model performance with appropriate metrics

---

## 📊 Dataset

**Source:** [Kaggle - Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

**Alternative Sources:**
- [UCI Credit Approval Dataset](https://archive.ics.uci.edu/ml/datasets/credit+approval)
- [Lend Club Loan Data](https://www.lendingclub.com/info/download-data.action)

---

## 🗄️ Database Setup

You can use any of the following databases:

### Option 1: SQLite (Recommended for Beginners)
```python
import sqlite3
import pandas as pd

# Create database
conn = sqlite3.connect('credit_risk.db')

# Load data
df = pd.read_csv('credit_risk.csv')

# Save to SQLite
df.to_sql('credit_data', conn, if_exists='replace')

# Query data
query = "SELECT * FROM credit_data LIMIT 100"
df = pd.read_sql(query, conn)
```

### Option 2: PostgreSQL
```sql
-- Create table
CREATE TABLE credit_data (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    income DECIMAL,
    credit_score INTEGER,
    employment_years INTEGER,
    loan_amount DECIMAL,
    default INTEGER
);

-- Load data
COPY credit_data(age, income, credit_score, employment_years, loan_amount, default)
FROM 'credit_risk.csv' DELIMITER ',' HEADER CSV;
```

---

## 📝 Exercise Questions

### Part 1: Data Understanding
1. Load the dataset and explore its structure
2. What is the distribution of the target variable (default)?
3. Are there any missing values? How would you handle them?
4. What are the key features that might predict credit default?

### Part 2: Data Preprocessing
1. Split the data into training and testing sets (80/20)
2. Handle any categorical variables
3. Scale features if necessary (explain why Decision Trees don't require scaling)
4. Handle class imbalance using SMOTE or class weights

### Part 3: Model Building
1. Build a basic Decision Tree classifier
2. Explain the Gini impurity and entropy criteria
3. How does the decision tree make splits?
4. What is the difference between max_depth, min_samples_split, and min_samples_leaf?

### Part 4: Model Evaluation
1. Evaluate the model using accuracy, precision, recall, and F1-score
2. Why is accuracy not enough for imbalanced datasets?
3. Create a confusion matrix
4. Plot the ROC curve and calculate AUC

### Part 5: Hyperparameter Tuning
1. Use GridSearchCV to find optimal hyperparameters
2. What is the best max_depth for this dataset?
3. How does pruning help prevent overfitting?

### Part 6: Feature Importance
1. Which features are most important for prediction?
2. Visualize the decision tree
3. Explain how the tree makes decisions

### Part 7: Production Considerations
1. How would you handle new data with missing values?
2. What monitoring would you put in place?
3. How would you detect data drift?

---

## 💻 Implementation Template

```python
# Your task: Complete this template

# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Add more imports as needed

# 2. Load data
# TODO: Load data from database or CSV

# 3. Explore data
# TODO: Check data types, missing values, distributions

# 4. Preprocess data
# TODO: Handle missing values, encode categorical variables

# 5. Split data
# TODO: Split into train/test sets

# 6. Train model
# TODO: Build and train Decision Tree

# 7. Tune hyperparameters
# TODO: Use GridSearchCV

# 8. Evaluate
# TODO: Calculate metrics

# 9. Analyze results
# TODO: Feature importance, visualizations
```

---

## 📤 Submission

Complete the solution file: `solution.py`

Your solution should include:
1. Data loading and preprocessing
2. Model training with hyperparameter tuning
3. Evaluation metrics
4. Feature importance analysis
5. Documentation of your approach

---

## ✅ Solution

See `solution.py` for the complete implementation.

### Key Points in Solution:
- **Data Preprocessing**: Handle missing values using median imputation
- **Class Imbalance**: Use class_weight='balanced'
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Evaluation**: Multiple metrics including ROC-AUC
- **Interpretation**: Feature importance and tree visualization