"""
Concept 5: Visualize what the hidden neurons learned.
Plot the decision boundary that each hidden neuron carved.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

W1 = np.random.randn(2, 2)
b1 = np.random.randn(1, 2)
W2 = np.random.randn(2, 1)
b2 = np.random.randn(1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Train (same as before)
for _ in range(10000):
    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    y_pred = sigmoid(z2)
    
    d_z2 = 2 * (y_pred - y) / len(X) * y_pred * (1 - y_pred)
    d_W2 = h.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    d_h = d_z2 @ W2.T
    d_z1 = d_h * h * (1 - h)
    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    W1 -= 0.5 * d_W1
    b1 -= 0.5 * d_b1
    W2 -= 0.5 * d_W2
    b2 -= 0.5 * d_b2

print("Final weights:")
print(f"W1 (input -> hidden):\n{W1}")
print(f"b1 (hidden biases): {b1}")
print(f"W2 (hidden -> output):\n{W2}")
print(f"b2 (output bias): {b2}")

# Plot the decision boundaries of each hidden neuron and the final output
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Make a fine grid covering the input space
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                     np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Compute hidden neuron outputs and final output across the grid
h1_out = sigmoid(grid @ W1[:, 0:1] + b1[:, 0:1]).reshape(xx.shape)
h2_out = sigmoid(grid @ W1[:, 1:2] + b1[:, 1:2]).reshape(xx.shape)
final = sigmoid(sigmoid(grid @ W1 + b1) @ W2 + b2).reshape(xx.shape)

# Plot each
for ax, surface, title in zip(axes,
                              [h1_out, h2_out, final],
                              ["Hidden neuron 1\n(its decision surface)",
                               "Hidden neuron 2\n(its decision surface)",
                               "Final output\n(combines both)"]):
    cs = ax.contourf(xx, yy, surface, levels=20, cmap="RdYlGn", alpha=0.6)
    ax.contour(xx, yy, surface, levels=[0.5], colors="black", linewidths=2)
    
    # Plot the 4 XOR points
    for (x1, x2), label in [((0,0),0), ((0,1),1), ((1,0),1), ((1,1),0)]:
        color = "#EF9F27" if label == 1 else "#222222"
        ax.scatter(x1, x2, c=color, s=200, zorder=5, edgecolors="white", linewidths=2)
    
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig("05_hidden.png", dpi=80, bbox_inches="tight")
print("\nSaved plot to 05_hidden.png")
print("Each hidden neuron drew a STRAIGHT line.")
print("The output combined them into a curved region that solves XOR!")
