#!/usr/bin/env python3
"""
Artificial Neural Network - In Memory Example
This example demonstrates a simple ANN implementation using NumPy
for binary classification on synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        
        # Initialize weights and biases with random values
        # Using He initialization for better convergence
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
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
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward pass (backpropagation)
        
        Args:
            X: Input data
            y: True labels
            output: Network predictions
        """
        m = X.shape[0]  # Number of examples
        
        # Calculate output layer error
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Calculate hidden layer error
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
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if epoch % print_every == 0:
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
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
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input data
        
        Returns:
            Prediction probabilities
        """
        return self.forward(X)


def main():
    """Main function to demonstrate the ANN"""
    print("Artificial Neural Network - In Memory Example")
    print("=" * 50)
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Reshape y to be a column vector
    y = y.reshape(-1, 1)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the neural network
    print("\nCreating neural network...")
    ann = SimpleANN(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    
    print("Training the network...")
    losses = ann.train(X_train_scaled, y_train, epochs=2000, print_every=200)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = ann.predict(X_test_scaled)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    visualize_results(X_test_scaled, y_test, test_predictions, losses, ann)
    
    print("\nTraining completed successfully!")


def visualize_results(X, y, predictions, losses, model):
    """Visualize the training results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Decision boundary
    axes[0].scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', alpha=0.6)
    
    # Create a mesh for decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    axes[0].contour(xx, yy, Z, levels=[0.5], colors='red', linestyles='--')
    axes[0].set_title('Decision Boundary')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Plot 2: Training loss
    axes[1].plot(losses)
    axes[1].set_title('Training Loss Over Time')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Plot 3: Predictions vs True labels
    axes[2].scatter(X[:, 0], X[:, 1], c=predictions.flatten(), cmap='coolwarm', alpha=0.6, label='Predictions')
    axes[2].scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', alpha=0.1, marker='x', s=20, label='True labels')
    axes[2].set_title('Predictions vs True Labels')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('ann_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()