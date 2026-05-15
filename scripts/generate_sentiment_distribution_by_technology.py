#!/usr/bin/env python3
"""
Barras agrupadas: distribuição percentual de final_sentiment por tecnologia e momento.

Dados: data/reference_labels_final_complete.json
Estilo alinhado aos gráficos do artigo (sans-serif, grade discreta, 300 DPI).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "reference_labels_final_complete.json"
OUT_PATH = REPO_ROOT / "results" / "figures" / "sentiment_distribution_by_technology.png"

MODELS: list[str] = ["gpt", "gemini", "copilot"]
MODEL_TITLE = {"gpt": "GPT", "gemini": "Gemini", "copilot": "Copilot"}

EVENT_SLUGS = ["launch_initial", "latest_version"]
EVENT_LABEL = {
    "launch_initial": "Lançamento inicial",
    "latest_version": "Versão recente",
}

# Ordem das barras em cada grupo (pedido): NEG → NEU → POS (mesmas cores dos temporais)
SENTIMENT_ORDER = ["NEGATIVO", "NEUTRO", "POSITIVO"]
LABEL_PT = {"NEGATIVO": "Negativo", "NEUTRO": "Neutro", "POSITIVO": "Positivo"}
COLORS = {
    "NEGATIVO": "#d62728",
    "NEUTRO": "#7f7f7f",
    "POSITIVO": "#1f77b4",
}

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
            "font.size": 11,
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


def load_counters(path: Path) -> dict[tuple[str, str], Counter[str]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("JSON deve ser uma lista de objetos")

    allowed_models = set(MODELS)
    allowed_events = set(EVENT_SLUGS)
    out: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for r in rows:
        if not isinstance(r, dict):
            continue
        model = str(r.get("model") or "").strip().lower()
        event = str(r.get("event") or "").strip().lower()
        if model not in allowed_models or event not in allowed_events:
            continue
        s = str(r.get("final_sentiment") or "").strip().upper()
        if s not in SENTIMENT_ORDER:
            continue
        out[(model, event)][s] += 1

    return dict(out)


def counter_total(c: Counter[str]) -> int:
    return int(sum(c[s] for s in SENTIMENT_ORDER))


def percentages(c: Counter[str]) -> dict[str, float]:
    tot = counter_total(c)
    if tot <= 0:
        return dict.fromkeys(SENTIMENT_ORDER, 0.0)
    return {s: 100.0 * float(c[s]) / float(tot) for s in SENTIMENT_ORDER}


def plot_figure(
    counters: dict[tuple[str, str], Counter[str]],
    path: Path,
) -> None:
    apply_article_theme()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.65), sharey=True)

    x = np.arange(len(EVENT_SLUGS))
    n_sent = len(SENTIMENT_ORDER)
    bar_w = 0.23
    offsets = (np.arange(n_sent) - (n_sent - 1) / 2.0) * bar_w

    for ax, model in zip(axes, MODELS, strict=True):
        vals = np.zeros((len(EVENT_SLUGS), n_sent), dtype=float)
        for j, ev in enumerate(EVENT_SLUGS):
            pct = percentages(counters.get((model, ev), Counter()))
            for i, sent in enumerate(SENTIMENT_ORDER):
                vals[j, i] = pct[sent]

        for i, sent in enumerate(SENTIMENT_ORDER):
            ax.bar(
                x + offsets[i],
                vals[:, i],
                bar_w,
                label=LABEL_PT[sent],
                color=COLORS[sent],
                edgecolor="white",
                linewidth=0.65,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([EVENT_LABEL[e] for e in EVENT_SLUGS], fontsize=10.5)
        ax.set_title(MODEL_TITLE[model], fontsize=12, fontweight="normal", pad=10)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        add_article_grid(ax, axes="both", vertical_softer=True)
        style_axes_spines(ax)

    axes[0].set_ylabel("Porcentagem (%)", fontsize=11.5)

    st = fig.suptitle(
        "Distribuição percentual dos sentimentos por tecnologia e momento de referência",
        fontsize=12.5,
        fontweight="normal",
        y=1.02,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[s], ec="white", linewidth=0.65)
        for s in SENTIMENT_ORDER
    ]
    labels = [LABEL_PT[s] for s in SENTIMENT_ORDER]
    leg = fig.legend(
        handles,
        labels,
        title="Sentimento",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#b8b8b8",
        fontsize=10.5,
        title_fontsize=11,
        columnspacing=1.6,
        handletextpad=0.65,
        handlelength=1.35,
        borderpad=0.55,
    )
    leg.get_title().set_fontweight("normal")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.82])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.28,
        bbox_extra_artists=[st, leg],
    )
    plt.close(fig)


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"ERRO: {DATA_PATH} não encontrado", file=sys.stderr)
        return 1

    print("=" * 72)
    print("Distribuição por tecnologia e momento — final_sentiment")
    print("=" * 72)
    print(f"Fonte: {DATA_PATH.relative_to(REPO_ROOT)}")
    print()

    counters = load_counters(DATA_PATH)

    print("Totais por tecnologia (tweets nos dois momentos):")
    for model in MODELS:
        t = sum(counter_total(counters.get((model, ev), Counter())) for ev in EVENT_SLUGS)
        print(f"  {MODEL_TITLE[model]:8s}  {t:6,} tweets")
    print()

    print("Totais por evento (todas as tecnologias):")
    for ev in EVENT_SLUGS:
        t = sum(counter_total(counters.get((m, ev), Counter())) for m in MODELS)
        print(f"  {EVENT_LABEL[ev]:22s}  {t:6,} tweets")
    print()

    print("Percentuais calculados (por tecnologia × momento):")
    for model in MODELS:
        print(f"  [{MODEL_TITLE[model]}]")
        for ev in EVENT_SLUGS:
            c = counters.get((model, ev), Counter())
            n = counter_total(c)
            pct = percentages(c)
            print(
                f"    {EVENT_LABEL[ev]:22s}  N = {n:5,}  |  "
                f"Neg {pct['NEGATIVO']:5.2f}%  Neu {pct['NEUTRO']:5.2f}%  Pos {pct['POSITIVO']:5.2f}%"
            )
    print()

    plot_figure(counters, OUT_PATH)

    abs_p = OUT_PATH.resolve()
    print("Figura gerada:")
    print(f"  Relativo: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  Absoluto: {abs_p}")
    if abs_p.is_file():
        print(f"  Tamanho: {abs_p.stat().st_size:,} bytes")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
