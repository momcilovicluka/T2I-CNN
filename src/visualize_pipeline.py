"""Generate a clean matplotlib version of the pipeline diagram (Slika 5.1).

Mirrors the Mermaid/ASCII flow in Plan/seminar2-rad-nacrt.md (section 5.1)
so it can be inserted directly into Word as a PNG.

Usage:
    python src/visualize_pipeline.py                 # -> results/figures/pipeline_diagram.png
    python src/visualize_pipeline.py --dpi 200 --output out.png

Layout notes:
- Serbian diacritics render with the default DejaVu Sans (check on the
  target machine if text shows as boxes).
- Colors group the phases: data prep (blue), T2I mapping (green),
  model/training (amber), evaluation (orange), saving (violet),
  synthesis (gray). All colors are light fills + darker borders so the
  figure stays readable in grayscale print.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------
# Node definitions: (id, y_center, width, height, fontsize, lines, style_key)
# --------------------------------------------------------------------------
CX = 5.0          # center x of the main single-column nodes
W = 6.6           # main node width
ARCH_L, ARCH_R = 2.7, 7.3
ARCH_W = 4.1

STYLE = {
    "data":   dict(face="#DEEBF7", edge="#2F5597"),
    "t2i":    dict(face="#E2EFDA", edge="#548235"),
    "model":  dict(face="#FFF2CC", edge="#BF8F00"),
    "eval":   dict(face="#FBE5D6", edge="#C55A11"),
    "save":   dict(face="#E4DFEC", edge="#7030A0"),
    "result": dict(face="#F2F2F2", edge="#404040"),
}

NODES = [
    dict(id="data",   y=14.35, x=CX, w=W, h=0.72, fs=8.6,
         text="Skup podataka\n(CSV / sklearn)"),
    dict(id="data",   y=13.27, x=CX, w=W, h=0.72, fs=8.6,
         text="Preprocesiranje\nčišćenje · one-hot · StandardScaler"),
    dict(id="data",   y=12.19, x=CX, w=W, h=0.72, fs=8.6,
         text="Stratifikovana podela 70/10/20\nista za sve metode · seed 42"),
    dict(id="data",   y=11.11, x=CX, w=W, h=0.72, fs=8.6,
         text="StandardScaler: fit samo na treningu\n(primenjeno na sve podskupove)"),
    dict(id="t2i",    y=9.93,  x=CX, w=W, h=0.78, fs=8.6,
         text="T2I fit na X_train\nkoordinate / statistike opsega (min–max)"),
    dict(id="t2i",    y=8.85,  x=CX, w=W, h=0.78, fs=8.6,
         text="transform → slike\n(N, 1, 32×32) ∈ [0, 1] · seed 42"),
    dict(id="model",  y=7.77,  x=CX, w=W, h=0.72, fs=8.6,
         text="DataLoader · batch = 32"),
    dict(id="model",  y=6.45,  x=ARCH_L, w=ARCH_W, h=0.92, fs=8.2,
         text="Od nule: shallow, resnet_scratch\n1 kanal (siva slika)\nlr = 1e-3"),
    dict(id="model",  y=6.45,  x=ARCH_R, w=ARCH_W, h=0.92, fs=8.2,
         text="Pretrenirani: resnet\n3 kanala · ImageNet normalizacija\nlr = 1e-3"),
    dict(id="model",  y=5.05,  x=CX, w=W, h=0.98, fs=8.0,
         text="Trening\nCrossEntropy + klasne težine + label smoothing (0,1)\n"
              "Adam · scheduler · rano zaustavljanje (15) · max 50 epoha"),
    dict(id="eval",   y=3.90,  x=CX, w=W, h=0.72, fs=8.6,
         text="Evaluacija na test skupu\nAcc · Prec · Rec · F1 · ROC/PR-AUC"),
    dict(id="save",   y=2.82,  x=CX, w=W, h=0.72, fs=8.6,
         text="Čuvanje rezultata\nJSON (atomski upis) + model.pt · resume: samo kompletni"),
    dict(id="result", y=1.74,  x=CX, w=W, h=0.72, fs=8.6,
         text="Agregacija i slike\nheatmap · ROC · Grad-CAM · ablacije · OF/OP · gustina"),
]

# Vertical connectors between the single-column nodes (same center x = CX).
# (y_from, y_to) where y is measured on the *outside* of the boxes.
LINKS = [
    (13.99, 13.63),   # 1 -> 2
    (12.91, 12.55),   # 2 -> 3
    (11.83, 11.47),   # 3 -> 4
    (10.75, 10.32),   # 4 -> 5
    (9.54, 9.24),     # 5 -> 6
    (8.46, 8.13),     # 6 -> 7
    (4.56, 4.28),     # 9 (training) -> 10
    (3.54, 3.18),     # 10 -> 11
    (2.46, 2.10),     # 11 -> 12
]

# Left-hand phase captions: (y_center, text, color)
PHASES = [
    (12.7, "PODACI", "#2F5597"),
    (9.4,  "T2I PRESLIKAVANJE", "#548235"),
    (6.1,  "MODEL I TRENING", "#BF8F00"),
    (3.9,  "EVALUACIJA", "#C55A11"),
    (2.8,  "ČUVANJE", "#7030A0"),
    (1.7,  "REZULTATI", "#404040"),
]

ARROW = dict(arrowstyle="-|>", mutation_scale=11, lw=1.3, color="#595959",
             shrinkA=1, shrinkB=1)


def box(ax, node):
    s = STYLE[node["id"]]
    x0 = node["x"] - node["w"] / 2
    y0 = node["y"] - node["h"] / 2
    p = FancyBboxPatch(
        (x0, y0), node["w"], node["h"],
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.2, edgecolor=s["edge"], facecolor=s["face"], zorder=2,
    )
    ax.add_patch(p)
    ax.text(node["x"], node["y"], node["text"], ha="center", va="center",
            fontsize=node["fs"], color="#1F1F1F", zorder=3, linespacing=1.35)


def vline(ax, x, y1, y2, **kw):
    kw = dict(ARROW)
    kw.update(dict(color="#595959", lw=1.3))
    ax.plot([x, x], [y1, y2], color="#595959", lw=1.3, zorder=1)


def arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(ARROW), zorder=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--output", default="results/figures/pipeline_diagram.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 11.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 15.1)
    ax.set_axis_off()
    ax.set_aspect("equal")

    for node in NODES:
        box(ax, node)

    # single-column arrows
    for y1, y2 in LINKS:
        arrow(ax, CX, y1, y2)

    # 7 -> (arch L/R) split: tee above the arch row
    tee = 7.14
    arch_top = 6.45 + 0.92 / 2
    vline(ax, CX, 7.77 - 0.36, tee)
    ax.plot([ARCH_L, ARCH_R], [tee, tee], color="#595959", lw=1.3, zorder=1)
    arrow(ax, ARCH_L, tee, arch_top)
    arrow(ax, ARCH_R, tee, arch_top)

    # (arch L/R) -> 9 merge: tee below the arch row
    arch_bot = 6.45 - 0.92 / 2
    tee2 = 5.80
    ax.plot([ARCH_L, ARCH_R], [tee2, tee2], color="#595959", lw=1.3, zorder=1)
    arrow(ax, ARCH_L, arch_bot, tee2)
    arrow(ax, ARCH_R, arch_bot, tee2)
    arrow(ax, CX, tee2, 5.05 + 0.98 / 2)

    # phase captions on the left margin
    for y, label, color in PHASES:
        ax.text(0.82, y, label, rotation=90, va="center", ha="center",
                fontsize=8.2, color=color, fontweight="bold")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KiB, dpi={args.dpi})")


if __name__ == "__main__":
    main()
