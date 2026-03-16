#!/usr/bin/env python3
"""
Artificial Neural Network - CSV Example
This example demonstrates using a real dataset from a CSV file
to train an ANN for classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleANN:
    """A simple Artificial Neural Network implementation"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation function for multi-class classification"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data (batch_size, input_size)
        
        Returns:
            Output predictions
        """
        # Hidden layer computation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer computation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        if self.output_size == 1:
            self.a2 = self.sigmoid(self.z2)
        else:
            self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward pass (backpropagation)
        
        Args:
            X: Input data
            y: True labels
            output: Network predictions
        """
        m = X.shape[0]
        
        if self.output_size == 1:
            # Binary classification
            dz2 = output - y
        else:
            # Multi-class classification
            dz2 = output - y
        
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000, print_every=100):
        """
        Train the neural network
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            print_every: Print loss every N epochs
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            if self.output_size == 1:
                # Binary cross-entropy
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            else:
                # Categorical cross-entropy
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if epoch % print_every == 0:
                if self.output_size == 1:
                    predictions = (output > 0.5).astype(int)
                    accuracy = np.mean(predictions == y)
                else:
                    predictions = np.argmax(output, axis=1)
                    true_labels = np.argmax(y, axis=1)
                    accuracy = np.mean(predictions == true_labels)
                
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input data
        
        Returns:
            Predicted class labels
        """
        output = self.forward(X)
        if self.output_size == 1:
            return (output > 0.5).astype(int)
        else:
            return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input data
        
        Returns:
            Prediction probabilities
        """
        return self.forward(X)


def load_iris_data():
    """Load and prepare the Iris dataset"""
    print("Loading Iris dataset...")
    
    # Create synthetic Iris-like data since we don't have the actual CSV
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Target classes: {list(iris.target_names)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df, iris.feature_names, 'species'


def load_wine_data():
    """Load and prepare the Wine dataset"""
    print("Loading Wine dataset...")
    
    # Create synthetic Wine-like data since we don't have the actual CSV
    from sklearn.datasets import load_wine
    wine = load_wine()
    
    # Create DataFrame
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target_names[wine.target]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(wine.feature_names)}")
    print(f"Target classes: {list(wine.target_names)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df, wine.feature_names, 'target'


def prepare_data(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """Prepare data for training"""
    print(f"\nPreparing data...")
    
    # Separate features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Encode target labels if they are strings
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Encoded target labels: {le.classes_}")
    else:
        le = None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to appropriate format for neural network
    if len(np.unique(y)) == 2:
        # Binary classification
        y_train_nn = y_train.reshape(-1, 1)
        y_test_nn = y_test.reshape(-1, 1)
        output_size = 1
    else:
        # Multi-class classification
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse_output=False)
        y_train_nn = ohe.fit_transform(y_train.reshape(-1, 1))
        y_test_nn = ohe.transform(y_test.reshape(-1, 1))
        output_size = len(np.unique(y))
    
    print(f"Training set shape: X={X_train_scaled.shape}, y={y_train_nn.shape}")
    print(f"Test set shape: X={X_test_scaled.shape}, y={y_test_nn.shape}")
    
    return (X_train_scaled, X_test_scaled, y_train_nn, y_test_nn, 
            scaler, le, output_size, np.unique(y))


def main():
    """Main function to demonstrate ANN with CSV data"""
    print("Artificial Neural Network - CSV Example")
    print("=" * 50)
    
    # Choose dataset (Iris for multi-class, Wine for another multi-class example)
    print("1. Iris Dataset (3 classes)")
    print("2. Wine Dataset (3 classes)")
    
    choice = input("Choose dataset (1 or 2): ").strip()
    
    if choice == "1":
        df, feature_cols, target_col = load_iris_data()
    elif choice == "2":
        df, feature_cols, target_col = load_wine_data()
    else:
        print("Invalid choice, using Iris dataset")
        df, feature_cols, target_col = load_iris_data()
    
    # Prepare data
    (X_train, X_test, y_train, y_test, scaler, label_encoder, 
     output_size, unique_labels) = prepare_data(df, feature_cols, target_col)
    
    # Create and train the neural network
    print(f"\nCreating neural network...")
    print(f"Input size: {X_train.shape[1]}")
    print(f"Hidden size: 10")
    print(f"Output size: {output_size}")
    
    ann = SimpleANN(
        input_size=X_train.shape[1], 
        hidden_size=10, 
        output_size=output_size, 
        learning_rate=0.01
    )
    
    print("Training the network...")
    losses = ann.train(X_train, y_train, epochs=2000, print_every=200)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = ann.predict(X_test)
    
    if output_size == 1:
        # Binary classification
        test_accuracy = np.mean(test_predictions.flatten() == y_test.flatten())
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=label_encoder.classes_ if label_encoder else None))
        
    else:
        # Multi-class classification
        test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(np.argmax(y_test, axis=1), test_predictions, 
                                  target_names=label_encoder.classes_ if label_encoder else None))
    
    # Confusion Matrix
    if output_size == 1:
        cm = confusion_matrix(y_test.flatten(), test_predictions.flatten())
    else:
        cm = confusion_matrix(np.argmax(y_test, axis=1), test_predictions)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize results
    visualize_results(losses, cm, ann, X_test, y_test, output_size, label_encoder)
    
    print("\nTraining completed successfully!")


def visualize_results(losses, confusion_mat, model, X_test, y_test, output_size, label_encoder):
    """Visualize the training results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training loss
    axes[0].plot(losses)
    axes[0].set_title('Training Loss Over Time')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Plot 2: Confusion Matrix
    if label_encoder:
        labels = label_encoder.classes_
    else:
        labels = [f"Class {i}" for i in range(confusion_mat.shape[0])]
    
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # Plot 3: Feature importance (simplified)
    if X_test.shape[1] <= 10:  # Only plot if not too many features
        # Calculate feature importance as sum of absolute weights from input to hidden layer
        feature_importance = np.sum(np.abs(model.W1), axis=1)
        
        axes[2].bar(range(len(feature_importance)), feature_importance)
        axes[2].set_title('Feature Importance')
        axes[2].set_xlabel('Feature Index')
        axes[2].set_ylabel('Importance')
        axes[2].set_xticks(range(len(feature_importance)))
        axes[2].set_xticklabels([f'F{i}' for i in range(len(feature_importance))], rotation=45)
    else:
        axes[2].text(0.5, 0.5, 'Too many features to display', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('ann_csv_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()