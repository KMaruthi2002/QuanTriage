"""Generate the demo thumbnail: a stylized quantum circuit feeding a 3-way triage
decision (benign / malignant / refer-to-doctor). Reproducible, no external assets.

    python scripts/make_thumbnail.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
INK = "#0b1f3a"      # deep navy text
TEAL = "#17a2a2"     # encode gates
PURPLE = "#6f42c1"   # entangle gates
GREEN = "#27ae60"
RED = "#c0392b"
AMBER = "#e67e22"
PAPER = "#f6f8fb"

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# ---- title -------------------------------------------------------------
ax.text(5, 9.25, "Quantum Cancer Triage", ha="center", va="center",
        fontsize=27, fontweight="bold", color=INK)
ax.text(5, 8.5, "a quantum classifier that knows when to defer", ha="center",
        va="center", fontsize=14.5, color="#4a5568", style="italic")

# ---- stylized quantum circuit (left) -----------------------------------
n_wires = 5
y0, y1 = 2.6, 6.6
ys = [y0 + i * (y1 - y0) / (n_wires - 1) for i in range(n_wires)]
x_start, x_end = 0.7, 4.5
for y in ys:
    ax.plot([x_start, x_end], [y, y], color="#9aa5b1", lw=1.6, zorder=1)

def gate(x, y, color):
    ax.add_patch(FancyBboxPatch((x - 0.28, y - 0.28), 0.56, 0.56,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=0, facecolor=color, zorder=3))

# encode column (teal) + two entangling columns (purple) with CNOT-ish links
for y in ys:
    gate(1.5, y, TEAL)
for col_x in (2.7, 3.7):
    for y in ys:
        gate(col_x, y, PURPLE)
    ax.plot([col_x, col_x], [ys[0], ys[-1]], color=PURPLE, lw=1.2, alpha=0.45, zorder=2)

# measurement meter on the top wire
mx, my = 4.5, ys[-1]
ax.add_patch(FancyBboxPatch((mx - 0.28, my - 0.28), 0.56, 0.56,
             boxstyle="round,pad=0.02,rounding_size=0.12",
             linewidth=1.5, edgecolor=INK, facecolor="white", zorder=3))
ax.plot([mx - 0.13, mx + 0.13], [my - 0.1, my + 0.12], color=INK, lw=1.4, zorder=4)
ax.add_patch(plt.matplotlib.patches.Arc((mx, my - 0.12), 0.34, 0.3, theta1=20,
             theta2=160, color=INK, lw=1.4, zorder=4))

ax.text(2.6, 1.75, "encode  →  entangle  →  measure", ha="center", va="center",
        fontsize=12, color="#4a5568")

# ---- arrow to the decision ---------------------------------------------
ax.add_patch(FancyArrowPatch((4.95, 4.6), (5.85, 4.6), arrowstyle="-|>",
             mutation_scale=26, lw=2.4, color=INK))

# ---- 3-way decision pills (right) --------------------------------------
def pill(y, color, label):
    ax.add_patch(FancyBboxPatch((6.1, y - 0.5), 3.3, 1.0,
                 boxstyle="round,pad=0.02,rounding_size=0.25",
                 linewidth=0, facecolor=color, zorder=3))
    ax.scatter(6.6, y, s=120, color="white", zorder=4)
    ax.text(7.05, y, label, ha="left", va="center", fontsize=15,
            color="white", fontweight="bold", zorder=4)

pill(6.2, GREEN, "Benign")
pill(4.6, RED, "Malignant")
pill(3.0, AMBER, "Refer to doctor")

# ---- footer tag --------------------------------------------------------
ax.text(5, 0.7, "Variational quantum classifier  ·  built with PennyLane",
        ha="center", va="center", fontsize=11.5, color="#7b8794")

plt.tight_layout()
out = []
for p in [ROOT / "demo" / "thumbnail.png", ROOT / "assets" / "thumbnail.png"]:
    fig.savefig(p, dpi=125, facecolor=PAPER)
    out.append(str(p))
plt.close(fig)
print("wrote:", *out, sep="\n  ")
