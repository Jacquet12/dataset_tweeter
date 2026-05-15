#!/usr/bin/env python3
"""
Gera artefatos para o artigo (estilo SBC / impressão), distintos dos gráficos
"tcc_moderno_*" e "evolucao_temporal_linha.png".

Entrada: dataset_final_v2_days_filled.json (gerado pelo pipeline de preenchimento).

Saídas:
  - serie_evolucao_sentimento_semanal.csv — série tabular (evento × modelo × semana × sentimento)
  - fig_sbc_evolucao_area_iniciais.png — áreas empilhadas 100% (launch_initial)
  - fig_sbc_evolucao_area_recentes.png — áreas empilhadas 100% (latest_version)
  - fig_sbc_distribuicao_global_sentimentos.png — distribuição global (final_sentiment)

Semana: week_index no JSON com min(dias//7, 12), ou seja, semanas 0–12.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(__file__).resolve().parent / "dataset_final_v2_days_filled.json"
OUT_DIR = Path(__file__).resolve().parent
CSV_OUT = OUT_DIR / "serie_evolucao_sentimento_semanal.csv"
FIG_AREA_INIT = OUT_DIR / "fig_sbc_evolucao_area_iniciais.png"
FIG_AREA_REC = OUT_DIR / "fig_sbc_evolucao_area_recentes.png"
FIG_GLOBAL = OUT_DIR / "fig_sbc_distribuicao_global_sentimentos.png"

# Ordem visual de baixo para cima no stackplot / barras empilhadas
STACK_ORDER = ["NEGATIVO", "NEUTRO", "POSITIVO"]
# Paleta alinhada à referência “Distribuição no conjunto analisado”:
# NEG vermelho, NEU cinza-azulado claro, POS azul-marinho escuro
COLORS = {
    "NEGATIVO": "#d62728",
    "NEUTRO": "#7f7f7f",
    "POSITIVO": "#1f77b4",
}
LABEL_PT = {"NEGATIVO": "Negativo", "NEUTRO": "Neutro", "POSITIVO": "Positivo"}

MODEL_LABEL = {
    "gpt": "GPT (ChatGPT)",
    "gemini": "Gemini / Bard",
    "copilot": "Copilot",
}

EVENT_LAUNCH = "launch_initial"
EVENT_LATEST = "latest_version"


def load_df() -> pd.DataFrame:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if "week_index" not in df.columns:
        raise SystemExit(
            "Coluna week_index ausente. Gere dataset_final_v2_days_filled.json com "
            "build_filled_dataset_and_temporal_charts.py primeiro."
        )
    return df


def weekly_long_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["event", "model", "week_index", "final_sentiment"])
        .size()
        .reset_index(name="n")
    )
    totals = (
        agg.groupby(["event", "model", "week_index"])["n"]
        .sum()
        .reset_index(name="week_n")
    )
    out = agg.merge(totals, on=["event", "model", "week_index"], how="left")
    out["pct"] = out["n"] / out["week_n"] * 100.0
    return out.sort_values(["event", "model", "week_index", "final_sentiment"])


def plot_stacked_areas(df_long: pd.DataFrame, event: str, title: str, path: Path) -> None:
    models = ["gpt", "gemini", "copilot"]
    weeks = np.arange(0, 13)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight="semibold", y=0.98)

    handles = None
    labels = None

    for ax, mod in zip(axes, models):
        sub = df_long[(df_long["event"] == event) & (df_long["model"] == mod)]
        wide = sub.pivot_table(
            index="week_index",
            columns="final_sentiment",
            values="pct",
            aggfunc="first",
        )
        for c in STACK_ORDER:
            if c not in wide.columns:
                wide[c] = np.nan
        wide = wide.reindex(weeks)
        series = [wide[c].to_numpy(dtype=float) for c in STACK_ORDER]
        # stackplot não aceita NaN: preenche 0 onde não há semana (mantém proporção só onde há dado)
        series = [np.nan_to_num(s, nan=0.0) for s in series]

        ax.stackplot(
            weeks,
            series,
            labels=[LABEL_PT[s] for s in STACK_ORDER],
            colors=[COLORS[s] for s in STACK_ORDER],
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("%")
        ax.set_title(MODEL_LABEL[mod], loc="left", fontsize=10, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.axhline(50, color="#e2e8f0", linewidth=0.6, zorder=0)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel("Semanas após o evento")
    axes[-1].set_xticks(list(range(0, 13)))
    fig.legend(
        handles,
        labels,
        title="Sentimento",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.28)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_global_distribution(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Painel 1: corpus completo (NEG na base, POS no topo)
    vc = df["final_sentiment"].value_counts()
    order = [s for s in STACK_ORDER if s in vc.index]
    counts = [int(vc[s]) for s in order]
    colors = [COLORS[s] for s in order]
    y = np.arange(len(order))
    axes[0].barh(y, counts, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL_PT.get(s, s) for s in order])
    axes[0].set_xlabel("Número de tweets")
    axes[0].set_title("Distribuição global\n(N = {:,})".format(len(df)))
    xmax = max(counts) if counts else 1
    for yi, c in zip(y, counts):
        axes[0].text(
            c + xmax * 0.012,
            yi,
            f"{c:,}",
            va="center",
            fontsize=9,
            color="#212121",
        )

    # Painel 2: proporção por tipo de evento (2 barras empilhadas 100%)
    ev_order = [EVENT_LAUNCH, EVENT_LATEST]
    ev_labels = ["Lançamento inicial", "Versão recente"]
    bottom = np.zeros(len(ev_order))
    x = np.arange(len(ev_order))
    for sent in STACK_ORDER:
        heights = []
        for ev in ev_order:
            sub = df[df["event"] == ev]
            tot = len(sub)
            h = (sub["final_sentiment"] == sent).sum() / tot * 100.0 if tot else 0.0
            heights.append(h)
        heights = np.array(heights)
        axes[1].bar(
            x,
            heights,
            bottom=bottom,
            label=LABEL_PT[sent],
            color=COLORS[sent],
            edgecolor="white",
            linewidth=0.5,
        )
        for i, (b, h) in enumerate(zip(bottom, heights)):
            if h >= 8:
                axes[1].text(
                    i,
                    b + h / 2,
                    f"{h:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="semibold",
                )
        bottom += heights

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ev_labels, fontsize=9)
    axes[1].set_ylabel("Porcentagem (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Distribuição por tipo de evento")
    leg = axes[1].legend(
        title="Sentimento",
        loc="upper right",
        fontsize=8,
        edgecolor="#9e9e9e",
        fancybox=False,
        framealpha=1.0,
    )
    leg.get_frame().set_linewidth(0.85)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4, color="#bdbdbd")

    fig.suptitle(
        "Distribuição de sentimentos no conjunto analisado",
        fontsize=11,
        fontweight="semibold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_GLOBAL, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_df()
    long_df = weekly_long_table(df)
    long_df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    plot_stacked_areas(
        long_df,
        EVENT_LAUNCH,
        "Evolução temporal — eventos iniciais\n"
        "Proporção semanal por sentimento",
        FIG_AREA_INIT,
    )
    plot_stacked_areas(
        long_df,
        EVENT_LATEST,
        "Evolução temporal — versões recentes\n"
        "Proporção semanal por sentimento",
        FIG_AREA_REC,
    )
    plot_global_distribution(df)

    print(
        "\n".join(
            [
                f"CSV: {CSV_OUT}",
                f"Figura: {FIG_AREA_INIT}",
                f"Figura: {FIG_AREA_REC}",
                f"Figura: {FIG_GLOBAL}",
            ]
        )
    )


if __name__ == "__main__":
    main()
