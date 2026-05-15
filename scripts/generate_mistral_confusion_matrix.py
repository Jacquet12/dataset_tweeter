#!/usr/bin/env python3
"""
Heatmap da matriz de confusão Mistral (predição) vs final_sentiment (referência).

Dados: data/mistral_confusion_matrix.csv
Estilo alinhado aos gráficos do artigo (matplotlib, 300 DPI).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "mistral_confusion_matrix.csv"
OUT_PATH = REPO_ROOT / "results" / "figures" / "mistral_confusion_matrix.png"

CLASS_ORDER = ["POSITIVO", "NEGATIVO", "NEUTRO"]
LABEL_PT = {"POSITIVO": "Positivo", "NEGATIVO": "Negativo", "NEUTRO": "Neutro"}


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


def load_matrix(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows_map: dict[str, dict[str, str]] = {}
        for row in reader:
            tr = str(row.get("final_sentiment_true", "")).strip().upper()
            rows_map[tr] = row
    missing = [c for c in CLASS_ORDER if c not in rows_map]
    if missing:
        raise ValueError(f"Linhas em falta no CSV: {missing}")

    col_keys = [f"mistral_pred_{c}" for c in CLASS_ORDER]
    mat = np.zeros((3, 3), dtype=int)
    for i, tr in enumerate(CLASS_ORDER):
        r = rows_map[tr]
        for j, ck in enumerate(col_keys):
            if ck not in r:
                raise ValueError(f"Coluna em falta: {ck}")
            mat[i, j] = int(r[ck])
    return mat


def off_diagonal_stats(mat: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Devolve ((i,j,val) maior fora da diagonal), ((i,j,val) menor fora da diagonal, valor>0)."""
    off = [(i, j, int(mat[i, j])) for i in range(3) for j in range(3) if i != j]
    if not off:
        raise ValueError("Matriz inválida")
    mx = max(off, key=lambda t: t[2])
    positive = [t for t in off if t[2] > 0]
    mn = min(positive, key=lambda t: t[2]) if positive else mx
    return mx, mn


def print_matrix_terminal(mat: np.ndarray) -> None:
    print("Matriz utilizada (linhas = referência final_sentiment, colunas = Mistral):")
    header = " " * 14 + "".join(f"{LABEL_PT[c]:>12s}" for c in CLASS_ORDER)
    print(header)
    for i, tr in enumerate(CLASS_ORDER):
        row = f"  {LABEL_PT[tr]:12s}"
        for j in range(3):
            row += f"{int(mat[i, j]):12d}"
        print(row)


def plot_heatmap(mat: np.ndarray, path: Path) -> None:
    apply_article_theme()
    fig, ax = plt.subplots(figsize=(6.9, 5.6))

    vmax = float(mat.max()) if mat.size else 1.0
    im = ax.imshow(mat, cmap="Blues", aspect="equal", vmin=0.0, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Número de tweets", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([LABEL_PT[c] for c in CLASS_ORDER], fontsize=10)
    ax.set_yticklabels([LABEL_PT[c] for c in CLASS_ORDER], fontsize=10)
    ax.set_xlabel("Sentimento predito (Mistral 7B Instruct)", fontsize=10)
    ax.set_ylabel("Sentimento de referência (final_sentiment)", fontsize=10)
    ax.set_title(
        "Matriz de confusão — Mistral vs referência\n"
        "(contagens absolutas de tweets)",
        fontsize=11,
        fontweight="normal",
        pad=12,
    )

    for i in range(3):
        for j in range(3):
            v = int(mat[i, j])
            lum = mat[i, j] / vmax if vmax else 0
            txt_color = "white" if lum > 0.55 else "#1a1a1a"
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="medium",
                color=txt_color,
            )

    ax.set_xticks(np.arange(3), minor=True)
    ax.set_yticks(np.arange(3), minor=True)
    ax.grid(which="minor", color="#e8e8e8", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    style_axes_spines(ax)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"ERRO: {DATA_PATH} não encontrado", file=sys.stderr)
        return 1

    mat = load_matrix(DATA_PATH)

    print("=" * 72)
    print("Matriz de confusão — Mistral (heatmap)")
    print("=" * 72)
    print(f"Fonte: {DATA_PATH.relative_to(REPO_ROOT)}")
    print()
    print_matrix_terminal(mat)

    mx, mn = off_diagonal_stats(mat)
    tri, tci, tv = mx
    fri, fci, fv = mn
    ref_mx = LABEL_PT[CLASS_ORDER[tri]]
    pred_mx = LABEL_PT[CLASS_ORDER[tci]]
    ref_mn = LABEL_PT[CLASS_ORDER[fri]]
    pred_mn = LABEL_PT[CLASS_ORDER[fci]]
    print()
    print("Maior confusão (fora da diagonal, maior contagem):")
    print(f"  ref={ref_mx} → Mistral={pred_mx}  n={tv:,}")
    print("Menor confusão (fora da diagonal, menor contagem > 0):")
    print(f"  ref={ref_mn} → Mistral={pred_mn}  n={fv:,}")
    print()

    plot_heatmap(mat, OUT_PATH)

    abs_p = OUT_PATH.resolve()
    print(f"Figura salva: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Caminho absoluto: {abs_p}")
    if abs_p.is_file():
        print(f"Tamanho: {abs_p.stat().st_size:,} bytes")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
