"""
Decision Tree Exercise Solution - Credit Risk Assessment
========================================================
This is a complete solution for the Credit Risk Assessment exercise.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_and_explore_data():
    """Load and explore the dataset."""
    
    # For demonstration, create sample data
    # In production, load from: pd.read_csv('credit_risk.csv')
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.randint(1000, 100000, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business'], n_samples),
    }
    
    # Create target (more non-default than default)
    default_prob = 0.15  # 15% default rate
    data['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nTarget Distribution:\n{df['default'].value_counts()}")
    print(f"\nTarget Distribution (%):\n{df['default'].value_counts(normalize=True)*100}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    return df


# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Preprocess the data."""
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Make a copy
    df_processed = df.copy()
    
    # Handle categorical variables
    le = LabelEncoder()
    df_processed['employment_type'] = le.fit_transform(df_processed['employment_type'])
    
    print("\nEncoded 'employment_type':")
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    
    # Note: Decision Trees don't require feature scaling
    # because they make decisions based on threshold values
    
    return df_processed


# ============================================================================
# PART 3: MODEL BUILDING
# ============================================================================

def build_model(X_train, y_train):
    """Build and train the Decision Tree model."""
    
    print("\n" + "=" * 60)
    print("MODEL BUILDING")
    print("=" * 60)
    
    # Basic model
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle imbalance
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("\nModel Parameters:")
    print(f"  max_depth: {model.max_depth}")
    print(f"  min_samples_split: {model.min_samples_split}")
    print(f"  min_samples_leaf: {model.min_samples_leaf}")
    print(f"  criterion: {model.criterion}")
    
    return model


# ============================================================================
# PART 4: HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparameters(X_train, y_train):
    """Tune hyperparameters using GridSearchCV."""
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    base_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# ============================================================================
# PART 5: MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return y_pred, y_pred_proba


# ============================================================================
# PART 6: FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance."""
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print("\nFeature Importances:")
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
    plt.xlabel('Importance')
    plt.title('Feature Importance - Decision Tree')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return feature_importance


def visualize_tree(model, feature_names):
    """Visualize the decision tree."""
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['No Default', 'Default'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Decision Tree Visualization')
    plt.tight_layout()
    plt.savefig('decision_tree.png')
    plt.show()


