"""
Concept 2: The neuron's decision is a line in 2D.
We're going to LITERALLY plot the line w1*x1 + w2*x2 + bias = 0.
"""
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np


def plot_gate(name, w1, w2, bias, truth_table, ax):
    """Plot the four input points and the neuron's decision line."""
    
    # Plot the 4 corners. Color = output (orange=1, gray=0)
    for (x1, x2), label in truth_table.items():
        color = "#EF9F27" if label == 1 else "#5F5E5A"
        ax.scatter(x1, x2, c=color, s=200, zorder=3, edgecolors="black", linewidths=0.5)
    
    # Draw the decision line: w1*x1 + w2*x2 + bias = 0
    # Solve for x2: x2 = -(w1*x1 + bias) / w2
    if w2 != 0:
        x1_range = np.linspace(-0.5, 1.5, 100)
        x2_line = -(w1 * x1_range + bias) / w2
        ax.plot(x1_range, x2_line, "g--", linewidth=2, label="decision line")
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{name}\nw1={w1}, w2={w2}, bias={bias}")
    ax.grid(True, alpha=0.3)
    ax.legend()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AND: only (1,1) is True
plot_gate("AND", w1=1, w2=1, bias=-1.5,
         truth_table={(0,0):0, (0,1):0, (1,0):0, (1,1):1},
         ax=axes[0])

# OR: only (0,0) is False
plot_gate("OR", w1=1, w2=1, bias=-0.5,
         truth_table={(0,0):0, (0,1):1, (1,0):1, (1,1):1},
         ax=axes[1])

# XOR: try ANY single line — it's impossible.
# Let's just pick something and see it fail.
plot_gate("XOR (impossible with 1 line!)", w1=1, w2=1, bias=-1.0,
         truth_table={(0,0):0, (0,1):1, (1,0):1, (1,1):0},
         ax=axes[2])

plt.tight_layout()
plt.savefig("02_lines.png", dpi=80, bbox_inches="tight")
print("Saved plot to 02_lines.png")
print()
print("Look at the XOR plot:")
print("  - Two orange dots are at (0,1) and (1,0) — opposite corners")
print("  - Two gray dots are at (0,0) and (1,1) — the OTHER opposite corners")
print("  - No matter where you draw a straight line, it can't separate them.")
