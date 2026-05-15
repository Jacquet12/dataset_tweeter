#!/usr/bin/env python3
"""
Barras agrupadas: taxa de concordância por classe de sentimento (âncora) e par de modelos.

Dados: data/agreement_analysis.json (stratified_agreement_by_anchor_class).
Estilo alinhado aos gráficos do artigo (gerar_graficos_finais_artigo.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "agreement_analysis.json"
OUT_PATH = REPO_ROOT / "results" / "figures" / "agreement_by_sentiment.png"

PAIR_ORDER = ["llama_vs_phi", "llama_vs_deepseek", "phi_vs_deepseek"]
PAIR_LEGEND = {
    "llama_vs_phi": "LLaMA vs Phi",
    "llama_vs_deepseek": "LLaMA vs DeepSeek",
    "phi_vs_deepseek": "Phi vs DeepSeek",
}
# Cores alinhadas a CONCORDANCIA_BAR_COLORS do artigo (tons distintos)
PAIR_COLORS = ["#5b79a5", "#4a6d94", "#8b6f62"]

CLASS_ORDER = ["POSITIVO", "NEGATIVO", "NEUTRO"]
LABEL_PT = {"POSITIVO": "Positivo", "NEGATIVO": "Negativo", "NEUTRO": "Neutro"}

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


def load_pct_matrix(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    strat = data.get("stratified_agreement_by_anchor_class")
    if not isinstance(strat, dict):
        raise ValueError("stratified_agreement_by_anchor_class em falta ou inválido")

    matrix: dict[str, dict[str, float]] = {}
    for pair in PAIR_ORDER:
        rows = strat.get(pair)
        if not isinstance(rows, list):
            raise ValueError(f"Dados em falta para o par: {pair}")
        by_class: dict[str, float] = {}
        for row in rows:
            sc = str(row.get("sentiment_class", "")).strip().upper()
            pct = float(row.get("pct_agree_within_anchor_class", 0.0))
            by_class[sc] = pct
        for c in CLASS_ORDER:
            if c not in by_class:
                raise ValueError(f"Classe {c} em falta no par {pair}")
        matrix[pair] = by_class
    return matrix


def print_terminal_stats(matrix: dict[str, dict[str, float]]) -> None:
    print("Valores utilizados no gráfico (% concordância por classe, âncora = 1.ª coluna do par):")
    for pair in PAIR_ORDER:
        print(f"\n  [{PAIR_LEGEND[pair]}]")
        for c in CLASS_ORDER:
            print(f"    {LABEL_PT[c]:12s}  {matrix[pair][c]:6.2f}%")

    print("\nMaior e menor concordância por classe (entre os 3 pares):")
    for c in CLASS_ORDER:
        vals = [(PAIR_LEGEND[p], matrix[p][c]) for p in PAIR_ORDER]
        mx = max(vals, key=lambda t: t[1])
        mn = min(vals, key=lambda t: t[1])
        print(
            f"  {LABEL_PT[c]:12s}  max: {mx[1]:5.2f}% ({mx[0]})  |  min: {mn[1]:5.2f}% ({mn[0]})"
        )


def plot_grouped_bars(matrix: dict[str, dict[str, float]], path: Path) -> None:
    apply_article_theme()
    fig, ax = plt.subplots(figsize=(9.5, 4.35))

    x = np.arange(len(CLASS_ORDER))
    n_pairs = len(PAIR_ORDER)
    width = 0.24
    offsets = np.linspace(-(n_pairs - 1) / 2 * width, (n_pairs - 1) / 2 * width, n_pairs)

    all_bars = []
    for i, pair in enumerate(PAIR_ORDER):
        heights = [matrix[pair][c] for c in CLASS_ORDER]
        bars = ax.bar(
            x + offsets[i],
            heights,
            width,
            label=PAIR_LEGEND[pair],
            color=PAIR_COLORS[i],
            edgecolor="white",
            linewidth=0.55,
        )
        all_bars.append((bars, heights))

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_PT[c] for c in CLASS_ORDER], fontsize=10)
    ax.set_ylabel("Porcentagem de concordância (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Concordância entre rotuladores por classe de sentimento\n"
        "(taxa em que o segundo modelo concorda com o rótulo do âncora)",
        fontsize=11,
        fontweight="normal",
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    add_article_grid(ax, axes="y", vertical_softer=True)
    style_axes_spines(ax)

    for bars, heights in all_bars:
        for b, h in zip(bars, heights):
            y_text = min(h + 1.6, 97.5)
            ax.text(
                b.get_x() + b.get_width() / 2,
                y_text,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
                rotation=0,
            )

    leg = ax.legend(
        title="Par de modelos",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.45,
        frameon=True,
        fancybox=False,
        edgecolor="#c0c0c0",
        framealpha=1.0,
        facecolor="white",
    )
    leg.get_frame().set_linewidth(0.7)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"ERRO: {DATA_PATH} não encontrado", file=sys.stderr)
        return 1

    matrix = load_pct_matrix(DATA_PATH)

    print("=" * 72)
    print("Concordância por classe de sentimento (pares de rotuladores)")
    print("=" * 72)
    print(f"Fonte: {DATA_PATH.relative_to(REPO_ROOT)}")
    print_terminal_stats(matrix)
    print()

    plot_grouped_bars(matrix, OUT_PATH)

    abs_path = OUT_PATH.resolve()
    print(f"Figura salva: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Caminho absoluto: {abs_path}")
    if abs_path.is_file():
        print(f"Confirmado no disco: sim ({abs_path.stat().st_size:,} bytes)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
