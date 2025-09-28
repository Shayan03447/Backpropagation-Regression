##  🔙 Backpropagation Neural Network for Classification and Regression

This repository showcases a from-scratch implementation of Backpropagation in a simple feedforward neural network.
The project demonstrates how backpropagation powers the training process for solving both classification (binary and multi-class) and regression (continuous prediction) problems.
Built entirely without high-level machine learning libraries, this implementation provides a clear understanding of the inner mechanics of backpropagation and how neural networks learn.


---
## 📌 What is Backpropagation?

Backpropagation (short for backward propagation of errors) is a fundamental algorithm used in training neural networks. It works by efficiently calculating gradients of the loss function with respect to the model’s parameters, enabling the network to learn by minimizing errors during training.

🔑 Key Steps in Backpropagation
1. **Forward Pass** → Input data flows through the network layer by layer to generate predictions.
2. **Loss Calculation** → The predicted outputs are compared with the actual values to measure the error.
3. **Backward Pass** → Using the chain rule, gradients of the loss with respect to each weight are computed.
4. **Weight Update** → The weights are updated (typically via Gradient Descent) to minimize the error and improve accuracy.
---

## 🧮 Mathematical Intuition  

A feedforward neural network performs computations in two main phases: **forward pass** and **backward pass**.  

### 1. Forward Pass  
For a single layer:  
\[
z = W x + b
\]  
\[
a = f(z)
\]  

where:  
- \(x\) = input vector  
- \(W\) = weight matrix  
- \(b\) = bias  
- \(f(\cdot)\) = activation function  
- \(a\) = output (activation)  

For multiple layers, the output of one layer becomes the input of the next.  

---

### 2. Loss Function  
The model’s performance is measured using a **loss function** \(L\):  
\[
L = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, \hat{y}_i)
\]  

where:  
- \(y_i\) = true label  
- \(\hat{y}_i\) = predicted output  
- \(\ell\) = error for one sample (e.g., MSE, cross-entropy)  

---

### 3. Backward Pass (Gradients)  
To optimize, we compute gradients of the loss w.r.t parameters:  

For the output layer:  
\[
\delta^L = \frac{\partial L}{\partial a^L} \cdot f'(z^L)
\]  

For hidden layers (using chain rule):  
\[
\delta^l = \big( (W^{l+1})^T \delta^{l+1} \big) \cdot f'(z^l)
\]  

where:  
- \(\delta^l\) = error term for layer \(l\)  
- \(f'(z^l)\) = derivative of activation function  

Gradients:  
\[
\frac{\partial L}{\partial W^l} = \delta^l (a^{l-1})^T
\]  
\[
\frac{\partial L}{\partial b^l} = \delta^l
\]  

---

### 4. Weight Update (Gradient Descent)  
Finally, parameters are updated as:  
\[
W^l \leftarrow W^l - \eta \frac{\partial L}{\partial W^l}
\]  
\[
b^l \leftarrow b^l - \eta \frac{\partial L}{\partial b^l}
\]  

where \(\eta\) = learning rate.  

---

⚡ **In summary:**  
- **Forward Pass** → compute activations  
- **Loss Calculation** → measure error  
- **Backward Pass** → propagate errors using chain rule  
- **Gradient Update** → adjust weights & biases to reduce loss  



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







