#!/usr/bin/env python3
"""
Figura: distribuição geral de final_sentiment (rotulagem de referência).

Estilo alinhado a scripts_graficos/gerar_graficos_finais_artigo.py
(SBC/IEEE, matplotlib, 300 DPI).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "reference_labels_final_complete.json"
OUT_PATH = REPO_ROOT / "results" / "figures" / "general_sentiment_distribution.png"

# --- Mesma paleta e rótulos que gerar_graficos_finais_artigo.py ---
COLORS = {
    "POSITIVO": "#1f77b4",
    "NEUTRO": "#7f7f7f",
    "NEGATIVO": "#d62728",
}
BAR_ORDER = ["POSITIVO", "NEUTRO", "NEGATIVO"]
LABEL_PT = {"POSITIVO": "Positivo", "NEUTRO": "Neutro", "NEGATIVO": "Negativo"}

GRID_COLOR = "#888888"
GRID_MAJOR_ALPHA = 0.2
GRID_MAJOR_LW = 0.55
GRID_X_ALPHA = 0.16
GRID_X_LW = 0.45


def apply_article_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": "#b8b8b8",
            "axes.linewidth": 0.75,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": False,
        }
    )


def style_axes_spines(ax, *, color: str = "#c0c0c0", linewidth: float = 0.75) -> None:
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def add_article_grid(
    ax,
    *,
    axes: str = "both",
    vertical_softer: bool = True,
) -> None:
    ax.set_axisbelow(True)
    if axes in ("y", "both"):
        ax.grid(
            True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=GRID_MAJOR_LW,
            alpha=GRID_MAJOR_ALPHA,
            color=GRID_COLOR,
        )
    if axes in ("x", "both"):
        alpha_x = GRID_X_ALPHA if vertical_softer and axes == "both" else GRID_MAJOR_ALPHA
        ax.grid(
            True,
            which="major",
            axis="x",
            linestyle="--",
            linewidth=GRID_X_LW,
            alpha=alpha_x,
            color=GRID_COLOR,
        )


def load_counts(path: Path) -> tuple[list[str], list[int], int]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("JSON deve ser uma lista de objetos")
    c = Counter()
    for r in rows:
        s = str(r.get("final_sentiment", "")).strip().upper()
        if s in COLORS:
            c[s] += 1
    total = sum(c[s] for s in BAR_ORDER)
    counts = [int(c[s]) for s in BAR_ORDER]
    return BAR_ORDER, counts, total


def plot_distribution(order: list[str], counts: list[int], total: int, path: Path) -> None:
    apply_article_theme()
    fig, ax = plt.subplots(figsize=(6.8, 4.35))

    x = np.arange(len(order))
    colors = [COLORS[s] for s in order]
    bars = ax.bar(
        x,
        counts,
        color=colors,
        edgecolor="white",
        linewidth=0.55,
        width=0.52,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_PT[s] for s in order], fontsize=10)
    ax.set_ylabel("Número de tweets")
    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax * 1.18)
    ax.set_title(
        f"Distribuição geral dos sentimentos\n(final_sentiment, N = {total:,})",
        fontsize=11,
        fontweight="normal",
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_article_grid(ax, axes="y", vertical_softer=True)
    style_axes_spines(ax)

    for b, cnt in zip(bars, counts):
        pct = (100.0 * cnt / total) if total else 0.0
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ymax * 0.015,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#212121",
            linespacing=1.05,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"ERRO: ficheiro não encontrado: {DATA_PATH}", file=sys.stderr)
        return 1

    order, counts, total = load_counts(DATA_PATH)

    print("=" * 72)
    print("Distribuição geral — final_sentiment")
    print("=" * 72)
    print(f"Fonte: {DATA_PATH.relative_to(REPO_ROOT)}")
    print(f"Total de tweets (rótulos válidos): {total:,}")
    print()
    for s, n in zip(order, counts):
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {LABEL_PT[s]:12s}  {n:6,}  ({pct:5.2f}%)")
    print()

    plot_distribution(order, counts, total, OUT_PATH)

    print(f"Figura salva: {OUT_PATH.relative_to(REPO_ROOT)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
