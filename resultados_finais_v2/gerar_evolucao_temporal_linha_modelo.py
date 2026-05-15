#!/usr/bin/env python3
"""
Dois gráficos de evolução temporal agregados (gpt + gemini + copilot).

Eixo X: semanas 0–12 após o evento; grade em quadrícula (major + minor); linhas com marcadores.

Um PNG para launch_initial e outro para latest_version.

Entrada: dataset_final_v2_days_filled.json (week_index recomendado 0–12).
Saída: resultados_finais_v2/fig_evolucao_temporal_linha_versao_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dataset_final_v2_days_filled.json"
OUT_DIR = Path(__file__).resolve().parent
FIG_INITIAL = OUT_DIR / "fig_evolucao_temporal_linha_versao_inicial.png"
FIG_RECENT = OUT_DIR / "fig_evolucao_temporal_linha_versao_recente.png"

# Positivo = azul, Neutro = cinza, Negativo = vermelho (rótulos em português na legenda)
COLORS = {
    "NEGATIVO": "#d62728",
    "NEUTRO": "#7f7f7f",
    "POSITIVO": "#1f77b4",
}
LABEL_PT = {"NEGATIVO": "Negativo", "NEUTRO": "Neutro", "POSITIVO": "Positivo"}
PLOT_ORDER = ["NEGATIVO", "NEUTRO", "POSITIVO"]

# Semanas após o evento (0–12)
WEEK_XMIN, WEEK_XMAX = 0, 12
# Anotações percentuais em alguns pontos (evita poluir todas as 13 semanas)
LABEL_WEEKS = [0, 3, 6, 9, 12]


def load_df() -> pd.DataFrame:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def weekly_pct_aggregated(df: pd.DataFrame, event: str) -> pd.DataFrame:
    sub = df[df["event"] == event]
    counts = (
        sub.groupby(["week_index", "final_sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    for s in PLOT_ORDER:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[PLOT_ORDER]
    return counts.div(counts.sum(axis=1), axis=0) * 100.0


def plot_template_style(pct: pd.DataFrame, out_path: Path) -> None:
    """Semanas 0–12, quadrícula, legenda com borda fina."""
    weeks_plot = np.arange(WEEK_XMIN, WEEK_XMAX + 1)
    pct_win = pct.reindex(weeks_plot)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.edgecolor": "#bdbdbd",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))

    # Grade “quadradinha”: major (eixo Y a cada 20%, X a cada 1 semana) + minor mais suave
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both",
        color="#d0d0d0",
        linestyle="-",
        linewidth=0.85,
        alpha=0.95,
    )
    # Linhas horizontais intermediárias (a cada 10%) = “quadradinhos” no eixo Y
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.grid(
        True,
        which="minor",
        axis="y",
        color="#e8e8e8",
        linestyle="-",
        linewidth=0.55,
        alpha=0.95,
    )

    for sent in PLOT_ORDER:
        y = pct_win[sent].to_numpy(dtype=float)
        ax.plot(
            weeks_plot,
            y,
            marker="o",
            markersize=8,
            markeredgewidth=1.0,
            markeredgecolor="white",
            markerfacecolor=COLORS[sent],
            linewidth=2.2,
            label=LABEL_PT[sent],
            color=COLORS[sent],
            clip_on=False,
            solid_capstyle="round",
        )
        for w in LABEL_WEEKS:
            if w not in pct_win.index or np.isnan(pct_win.loc[w, sent]):
                continue
            val = float(pct_win.loc[w, sent])
            ax.annotate(
                f"{val:.0f}%",
                xy=(w, val),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#212121",
            )

    ax.set_title(
        "Evolução Temporal dos Sentimentos",
        fontsize=14,
        fontweight="normal",
        pad=16,
    )
    ax.set_xlabel("Semanas após o evento", fontsize=11, labelpad=8)
    ax.set_ylabel("Porcentagem (%)", fontsize=11, labelpad=8)

    ax.set_xlim(WEEK_XMIN - 0.35, WEEK_XMAX + 0.35)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(list(range(WEEK_XMIN, WEEK_XMAX + 1)))

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10,
        title="Sentimento",
        title_fontsize=10,
        edgecolor="#9e9e9e",
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
    )
    leg.get_frame().set_linewidth(0.9)

    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    df = load_df()
    if "week_index" not in df.columns:
        raise SystemExit(
            "Gere antes dataset_final_v2_days_filled.json com build_filled_dataset_and_temporal_charts.py."
        )

    pct_init = weekly_pct_aggregated(df, "launch_initial")
    pct_rec = weekly_pct_aggregated(df, "latest_version")

    plot_template_style(pct_init, FIG_INITIAL)
    plot_template_style(pct_rec, FIG_RECENT)

    print(f"Salvo: {FIG_INITIAL}")
    print(f"Salvo: {FIG_RECENT}")


if __name__ == "__main__":
    main()
