"""
Concept 3: Solve XOR with TWO LAYERS by hand.
We'll use two hidden neurons (OR and NAND), then combine them with AND.
"""

def neuron(x1, x2, w1, w2, bias):
    return 1 if (w1 * x1 + w2 * x2 + bias) > 0 else 0


def xor_network(x1, x2):
    # Hidden layer:
    # h1 = OR(x1, x2)   -> fires when at least one is 1
    h1 = neuron(x1, x2, w1=1, w2=1, bias=-0.5)
    
    # h2 = NAND(x1, x2) -> fires UNLESS both are 1
    # (NAND = NOT AND. Weights are negated AND, then bias flipped.)
    h2 = neuron(x1, x2, w1=-1, w2=-1, bias=1.5)
    
    # Output layer:
    # AND of h1 and h2 -> "at least one is 1, AND not both are 1"
    # which is exactly XOR!
    out = neuron(h1, h2, w1=1, w2=1, bias=-1.5)
    return h1, h2, out


print(f"{'x1':>3} {'x2':>3} | {'h1(OR)':>6} {'h2(NAND)':>9} | {'XOR':>4}")
print("-" * 35)
for x1 in [0, 1]:
    for x2 in [0, 1]:
        h1, h2, out = xor_network(x1, x2)
        print(f"{x1:>3} {x2:>3} | {h1:>6} {h2:>9} | {out:>4}")

print()
print("Notice: XOR is the four numbers (0, 1, 1, 0) — exactly the truth table!")
print("Two layers of neurons solved what one layer cannot.")
