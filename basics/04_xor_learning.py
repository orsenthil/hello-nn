"""
Concept 4: Learning XOR from scratch with backpropagation.
No neural network libraries — just numpy and basic calculus.

Network architecture:
  - 2 inputs
  - 2 hidden neurons (with sigmoid activation)
  - 1 output neuron (with sigmoid activation)

We use sigmoid (a smooth S-curve) instead of the hard step function,
because backprop needs derivatives, and step functions aren't differentiable.
sigmoid(x) = 1 / (1 + e^-x)  -- still squashes to (0, 1).
"""
import numpy as np

# Reproducibility
np.random.seed(42)

# The XOR truth table
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])  # the targets

# ---- Initialize weights randomly ----
# W1: weights from inputs to hidden layer (2 inputs -> 2 hidden neurons)
# b1: biases for the hidden neurons
# W2: weights from hidden layer to output (2 hidden -> 1 output)
# b2: bias for the output neuron
W1 = np.random.randn(2, 2)  # 2x2 matrix
b1 = np.random.randn(1, 2)  # 1x2 vector
W2 = np.random.randn(2, 1)  # 2x1 matrix
b2 = np.random.randn(1, 1)  # 1x1 scalar

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    # Note: this expects s = sigmoid(x), not x itself
    return s * (1 - s)

# ---- Training loop ----
learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    # ===== FORWARD PASS =====
    # Compute the hidden layer's output
    z1 = X @ W1 + b1                # weighted sum at hidden layer
    h = sigmoid(z1)                 # hidden activations
    
    # Compute the output
    z2 = h @ W2 + b2                # weighted sum at output
    y_pred = sigmoid(z2)            # final prediction

    # ===== LOSS =====
    # Mean squared error: how wrong are we, on average?
    loss = np.mean((y_pred - y) ** 2)
    
    # ===== BACKWARD PASS (backpropagation) =====
    # We want to know: how should we adjust each weight to reduce loss?
    # Calculus (chain rule) gives us the answer.
    
    # Error at output:
    d_y_pred = 2 * (y_pred - y) / len(X)         # gradient of loss w.r.t. y_pred
    d_z2 = d_y_pred * sigmoid_derivative(y_pred) # gradient through the sigmoid
    
    # Gradients for the output layer's weights:
    d_W2 = h.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    
    # Push the error back to the hidden layer:
    d_h = d_z2 @ W2.T
    d_z1 = d_h * sigmoid_derivative(h)
    
    # Gradients for the hidden layer's weights:
    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    # ===== UPDATE WEIGHTS =====
    # Step in the direction that reduces loss
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
    # Print progress occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:>5} | loss = {loss:.4f}")

print(f"Epoch {epochs:>5} | loss = {loss:.4f}")
print()

# ---- Final predictions ----
print("Trained predictions:")
print(f"{'input':>10} | {'prediction':>11} | {'rounded':>7} | {'target':>6}")
for i, (x_in, target) in enumerate(zip(X, y)):
    pred = y_pred[i, 0]
    print(f"  {tuple(x_in)} -> | {pred:>11.4f} | {round(pred):>7} | {target[0]:>6}")

print()
print("The network LEARNED the right weights all by itself.")
print("That's what backpropagation does — and that's what unlocked deep learning.")
