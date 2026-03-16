#!/usr/bin/env python3
"""
Artificial Neural Networks - Solutions
This file contains solutions to the exercises in exercises.md
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import seaborn as sns


class AdvancedANN:
    """An advanced Artificial Neural Network implementation with multiple features"""
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, 
                 activation='relu', output_activation='sigmoid', 
                 weight_init='he', optimizer='sgd', dropout_rate=0.0):
        """
        Initialize the advanced neural network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh', 'swish')
            output_activation: Activation function for output layer
            weight_init: Weight initialization method ('random', 'xavier', 'he')
            optimizer: Optimization algorithm ('sgd', 'adam', 'momentum')
            dropout_rate: Dropout rate for regularization
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        
        # Initialize network architecture
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self.activations = []
        
        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            if weight_init == 'he':
                w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            elif weight_init == 'xavier':
                w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1.0 / self.layers[i])
            else:  # random
                w = np.random.randn(self.layers[i], self.layers[i+1]) * 0.1
            
            b = np.zeros((1, self.layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Determine activation function for this layer
            if i == len(self.layers) - 2:  # Output layer
                self.activations.append(output_activation)
            else:  # Hidden layers
                self.activations.append(activation)
        
        # Initialize optimizer parameters
        if optimizer == 'adam':
            self.m = [np.zeros_like(w) for w in self.weights]
            self.v = [np.zeros_like(w) for w in self.weights]
            self.t = 0
            self.beta1, self.beta2 = 0.9, 0.999
            self.epsilon = 1e-8
        elif optimizer == 'momentum':
            self.v = [np.zeros_like(w) for w in self.weights]
            self.beta = 0.9
    
    def activation_function(self, x, activation_type):
        """Apply activation function"""
        if activation_type == 'relu':
            return np.maximum(0, x)
        elif activation_type == 'sigmoid':
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif activation_type == 'tanh':
            return np.tanh(x)
        elif activation_type == 'swish':
            x = np.clip(x, -500, 500)
            return x * (1 / (1 + np.exp(-x)))
        elif activation_type == 'linear':
            return x
        elif activation_type == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def activation_derivative(self, x, activation_type):
        """Compute derivative of activation function"""
        if activation_type == 'relu':
            return (x > 0).astype(float)
        elif activation_type == 'sigmoid':
            return x * (1 - x)
        elif activation_type == 'tanh':
            return 1 - x**2
        elif activation_type == 'swish':
            sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        elif activation_type == 'linear':
            return np.ones_like(x)
        elif activation_type == 'softmax':
            # Derivative of softmax is handled differently in backprop
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def forward(self, X, training=True):
        """Forward pass through the network"""
        self.layer_inputs = [X]
        self.layer_outputs = [X]
        self.dropout_masks = []
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            
            # Activation function
            a = self.activation_function(z, self.activations[i])
            
            # Apply dropout if training and not output layer
            if training and self.dropout_rate > 0 and i < len(self.weights) - 1:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
                a = a * dropout_mask
                self.dropout_masks.append(dropout_mask)
            else:
                self.dropout_masks.append(np.ones_like(a))
            
            self.layer_outputs.append(a)
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y, output):
        """Backward pass (backpropagation)"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Compute output layer error
        if self.output_size == 1:
            # Binary classification
            dz = output - y
        elif self.output_activation == 'softmax':
            # Multi-class classification with softmax
            dz = output - y
        else:
            # Regression or other output activation
            dz = output - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dw = (1/m) * np.dot(self.layer_outputs[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            # Compute error for next layer
            if i > 0:  # Not input layer
                da = np.dot(dz, self.weights[i].T)
                # Apply dropout mask
                da = da * self.dropout_masks[i-1]
                # Apply activation derivative
                dz = da * self.activation_derivative(self.layer_outputs[i], self.activations[i-1])
    
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        """Update parameters using the chosen optimizer"""
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        
        elif self.optimizer == 'momentum':
            for i in range(len(self.weights)):
                self.v[i] = self.beta * self.v[i] + self.learning_rate * gradients_w[i]
                self.weights[i] -= self.v[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        
        elif self.optimizer == 'adam':
            self.t += 1
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients_w[i]
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients_w[i]**2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                
                # Update parameters
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs=1000, print_every=100, validation_data=None):
        """Train the neural network"""
        losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X, training=True)
            
            # Calculate loss
            if self.output_size == 1:
                # Binary cross-entropy
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            elif self.output_activation == 'softmax':
                # Categorical cross-entropy
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            else:
                # Mean squared error for regression
                loss = np.mean((output - y)**2)
            
            losses.append(loss)
            
            # Validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.forward(X_val, training=False)
                if self.output_size == 1:
                    val_loss = -np.mean(y_val * np.log(val_output + 1e-8) + (1 - y_val) * np.log(1 - val_output + 1e-8))
                elif self.output_activation == 'softmax':
                    val_loss = -np.mean(np.sum(y_val * np.log(val_output + 1e-8), axis=1))
                else:
                    val_loss = np.mean((val_output - y_val)**2)
                val_losses.append(val_loss)
            
            # Backward pass
            gradients_w, gradients_b = self.backward(X, y, output)
            self.update_parameters(gradients_w, gradients_b)
            
            # Print progress
            if epoch % print_every == 0:
                if self.output_size == 1:
                    predictions = (output > 0.5).astype(int)
                    accuracy = np.mean(predictions == y)
                elif self.output_activation == 'softmax':
                    predictions = np.argmax(output, axis=1)
                    true_labels = np.argmax(y, axis=1)
                    accuracy = np.mean(predictions == true_labels)
                else:
                    accuracy = r2_score(y, output)
                
                val_accuracy = None
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_output = self.forward(X_val, training=False)
                    if self.output_size == 1:
                        val_predictions = (val_output > 0.5).astype(int)
                        val_accuracy = np.mean(val_predictions == y_val)
                    elif self.output_activation == 'softmax':
                        val_predictions = np.argmax(val_output, axis=1)
                        val_true_labels = np.argmax(y_val, axis=1)
                        val_accuracy = np.mean(val_predictions == val_true_labels)
                    else:
                        val_accuracy = r2_score(y_val, val_output)
                
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}" + 
                      (f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}" if validation_data else ""))
        
        return losses, val_losses
    
    def predict(self, X):
        """Make predictions on new data"""
        output = self.forward(X, training=False)
        if self.output_size == 1:
            return (output > 0.5).astype(int)
        elif self.output_activation == 'softmax':
            return np.argmax(output, axis=1)
        else:
            return output
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward(X, training=False)


def exercise_1_binary_classification():
    """Solution for Exercise 1: Binary Classification with Custom Data"""
    print("Exercise 1: Binary Classification with Custom Data")
    print("=" * 50)
    
    # Generate synthetic student performance data
    np.random.seed(42)
    n_samples = 1000
    
    study_hours = np.random.uniform(1, 10, n_samples)
    previous_scores = np.random.uniform(0, 100, n_samples)
    
    # Create pass/fail based on a combination of factors
    pass_threshold = (study_hours * 10) + (previous_scores * 0.5) + np.random.normal(0, 5, n_samples)
    pass_fail = (pass_threshold > 70).astype(int)
    
    # Combine features
    X = np.column_stack([study_hours, previous_scores])
    y = pass_fail.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train network
    ann = AdvancedANN(
        input_size=2, 
        hidden_sizes=[5], 
        output_size=1, 
        learning_rate=0.01,
        activation='sigmoid',
        output_activation='sigmoid',
        weight_init='he'
    )
    
    print("Training network...")
    losses, _ = ann.train(X_train_scaled, y_train, epochs=1000, print_every=200)
    
    # Evaluate
    predictions = ann.predict(X_test_scaled)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test.flatten(), cmap='viridis', alpha=0.6)
    plt.title('True Labels')
    plt.xlabel('Study Hours (scaled)')
    plt.ylabel('Previous Score (scaled)')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=predictions.flatten(), cmap='coolwarm', alpha=0.6)
    plt.title('Predictions')
    plt.xlabel('Study Hours (scaled)')
    plt.ylabel('Previous Score (scaled)')
    
    plt.tight_layout()
    plt.savefig('exercise_1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy


def exercise_2_multi_class():
    """Solution for Exercise 2: Multi-class Classification"""
    print("\nExercise 2: Multi-class Classification")
    print("=" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # One-hot encode labels
    ohe = OneHotEncoder(sparse_output=False)
    y_encoded = ohe.fit_transform(y.reshape(-1, 1))
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train network
    ann = AdvancedANN(
        input_size=4,
        hidden_sizes=[10, 5],
        output_size=3,
        learning_rate=0.01,
        activation='relu',
        output_activation='softmax',
        weight_init='he'
    )
    
    print("Training network...")
    losses, _ = ann.train(X_train_scaled, y_train, epochs=2000, print_every=400)
    
    # Evaluate
    predictions = ann.predict(X_test_scaled)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=iris.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 3, 3)
    # Feature importance
    feature_importance = np.sum(np.abs(ann.weights[0]), axis=1)
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('exercise_2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy


def exercise_3_regression():
    """Solution for Exercise 3: Regression Task"""
    print("\nExercise 3: Regression Task")
    print("=" * 50)
    
    # Generate synthetic housing data
    np.random.seed(42)
    n_samples = 2000
    
    house_size = np.random.uniform(500, 5000, n_samples)
    bedrooms = np.random.randint(1, 7, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    # Create price based on features with some noise
    price = (house_size * 100) + (bedrooms * 5000) + (location_score * 10000) + np.random.normal(0, 20000, n_samples)
    price = np.maximum(price, 50000)  # Minimum price
    
    # Combine features
    X = np.column_stack([house_size, bedrooms, location_score])
    y = price.reshape(-1, 1)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create and train network
    ann = AdvancedANN(
        input_size=3,
        hidden_sizes=[10, 5],
        output_size=1,
        learning_rate=0.01,
        activation='relu',
        output_activation='linear',
        weight_init='he'
    )
    
    print("Training network...")
    losses, _ = ann.train(X_train_scaled, y_train_scaled, epochs=2000, print_every=400)
    
    # Evaluate
    predictions_scaled = ann.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    r2 = r2_score(y_test, predictions)
    mse = np.mean((y_test - predictions)**2)
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Predictions vs True Values')
    
    plt.subplot(1, 3, 3)
    residuals = y_test.flatten() - predictions.flatten()
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig('exercise_3_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return r2


def exercise_4_xor():
    """Solution for Exercise 4: XOR Problem"""
    print("\nExercise 4: XOR Problem")
    print("=" * 50)
    
    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Duplicate each combination 100 times
    X_train = np.repeat(X, 100, axis=0)
    y_train = np.repeat(y, 100, axis=0)
    
    # Create test set (original 4 combinations)
    X_test = X
    y_test = y
    
    # Create and train network
    ann = AdvancedANN(
        input_size=2,
        hidden_sizes=[4],  # At least 2 neurons needed for XOR
        output_size=1,
        learning_rate=0.1,
        activation='tanh',  # tanh works well for XOR
        output_activation='sigmoid',
        weight_init='xavier'  # Xavier initialization helps with XOR
    )
    
    print("Training network...")
    losses, _ = ann.train(X_train, y_train, epochs=5000, print_every=1000)
    
    # Evaluate
    predictions = ann.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Check if all XOR combinations are correct
    print("\nXOR Results:")
    for i in range(len(X_test)):
        print(f"XOR({X_test[i][0]}, {X_test[i][1]}) = {predictions[i][0]} (Expected: {y_test[i][0]})")
    
    # Visualize decision boundary
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    # Create mesh for decision boundary
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = ann.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.flatten(), cmap='RdYlBu', s=200, edgecolors='black')
    plt.title('XOR Decision Boundary')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    
    plt.subplot(1, 3, 3)
    plt.bar(range(4), predictions.flatten(), color=['red' if p != y_test[i] else 'green' for i, p in enumerate(predictions)])
    plt.title('XOR Predictions')
    plt.xlabel('Input Combination')
    plt.ylabel('Output')
    plt.xticks(range(4), ['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
    
    plt.tight_layout()
    plt.savefig('exercise_4_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy


def main():
    """Run all exercise solutions"""
    print("Artificial Neural Networks - Exercise Solutions")
    print("=" * 60)
    
    results = {}
    
    try:
        results['Exercise 1'] = exercise_1_binary_classification()
    except Exception as e:
        print(f"Exercise 1 failed: {e}")
    
    try:
        results['Exercise 2'] = exercise_2_multi_class()
    except Exception as e:
        print(f"Exercise 2 failed: {e}")
    
    try:
        results['Exercise 3'] = exercise_3_regression()
    except Exception as e:
        print(f"Exercise 3 failed: {e}")
    
    try:
        results['Exercise 4'] = exercise_4_xor()
    except Exception as e:
        print(f"Exercise 4 failed: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    for exercise, result in results.items():
        print(f"{exercise}: {result:.4f}")


if __name__ == "__main__":
    main()