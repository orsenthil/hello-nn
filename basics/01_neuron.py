"""
Concept 1: A single neuron is just weighted sum + threshold.
No magic. No machine learning yet. Just arithmetic.
"""

def neuron(x1, x2, w1, w2, bias):
    """
    x1, x2: inputs (0 or 1)
    w1, w2: weights (how much each input matters)
    bias:   threshold offset
    
    Returns 1 if the weighted sum exceeds 0, else 0.
    """
    weighted_sum = (w1 * x1) + (w2 * x2) + bias
    return 1 if weighted_sum > 0 else 0


# Let's build an AND gate by HAND (no learning yet).
# AND: output 1 only if BOTH inputs are 1.
# We need: 0+0 -> 0, 0+1 -> 0, 1+0 -> 0, 1+1 -> 1
# 
# Try weights w1=1, w2=1, bias=-1.5
# (0,0): 0 + 0 - 1.5 = -1.5  -> 0  ✓
# (0,1): 0 + 1 - 1.5 = -0.5  -> 0  ✓
# (1,0): 1 + 0 - 1.5 = -0.5  -> 0  ✓
# (1,1): 1 + 1 - 1.5 =  0.5  -> 1  ✓

print("AND gate (w1=1, w2=1, bias=-1.5):")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        out = neuron(x1, x2, w1=1, w2=1, bias=-1.5)
        print(f"  ({x1}, {x2}) -> {out}")

print("\nOR gate (w1=1, w2=1, bias=-0.5):")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        out = neuron(x1, x2, w1=1, w2=1, bias=-0.5)
        print(f"  ({x1}, {x2}) -> {out}")
