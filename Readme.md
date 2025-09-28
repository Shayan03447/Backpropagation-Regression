##  🔙 Backpropagation Neural Network for Classification and Regression

This repository demonstrates a neural network implementation of Backpropagation that powers training of nural network for solving both classification and regression problems from scratch.
This project implements a simple feedforward neural network trained using backpropagation algorithm.
It supports both:
Classification (e.g., binary or multi-class)
Regression (predicting continuous outputs)
The network is built from scratch without relying on high-level machine learning libraries, giving insight into the inner workings of backpropagation.


---
## 📌 What is Backpropagation?

Backpropagation (short for *backward propagation of errors*) is an algorithm used to minimize the loss function in neural networks by efficiently computing gradients.

Key steps:
1. **Forward Pass** → Compute output of the neural network.flow towords forword direction Input data is passed through the network layer by layer to compute outputs
2. **Loss Calculation** → Compare predicted vs. actual values.Calculate the difference between predicted and true values.
3. **Backward Pass** → Apply chain rule to compute gradients.Calculate gradients of the loss w.r.t weights using chain rule.
4. **Weight Update** → Adjust weights using Gradient Descent. Adjust weights using gradient descent to minimize loss.

---

## 🧮 Mathematical Intuition
\[
y = f(Wx + b)
\]
Loss function \( L \):  
\[
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
\]
This project explains how the **chain rule** is applied layer by layer to update weights.

## 🛠️ Implementation Details

- **Language**: Python
- **Libraries**: NumPy, Matplotlib,tensor-flow, keras 
- **Network**: Simple Multi-Layer Perceptron (MLP)

Code includes:
- Forward pass
- Loss calculation (MSE / Cross-Entropy)
- Manual gradient computation
- Weight updates

## 📂 Folder Structure

BACKPROPAGATION-REGRESSION/
│── Backpropagation_classification.ipynb
│── Backpropagation_keras.ipynb
│── Backpropagation_regression.ipynb
│── README.md

📊 Example Results

Loss curve convergence
Sample predictions before vs. after training

🚀 Future Work

Add support for multiple hidden layers
Implement other activation functions
Compare manual vs. PyTorch/TensorFlow automatic differentiation

✅ Key Learnings

Understanding gradients via chain rule
Manual backpropagation strengthens fundamentals
How optimization works under the hood of deep learning frameworks

## Features

Fully connected feedforward neural network
Support for multiple hidden layers
Activation functions: Sigmoid, ReLU, Linear (for regression output)
Loss functions: Cross-Entropy (classification), Mean Squared Error (regression)
Mini-batch Gradient Descent
Early stopping and learning rate scheduling (optional)
Easy to extend and customize

## Installation

1. Clone this repository:
2. 
git clone https://github.com/Shayan03447/Backpropagation-Regression.git
cd backpropagation-classification-regression

2. Install dependencies:
pip install -r requirements.txt

Note: This implementation mainly depends on numpy.

Usage
from backpropagation import NeuralNetwork

# Define network architecture
layers = [input_size, hidden1_size, ..., output_size]

# Initialize the network (for classification)
nn = NeuralNetwork(layers, task='classification')

# Train the network
model.fit(df.iloc[:,0:-1].values,df['lpa'].values,epochs=75,verbose=1,batch_size=1)







