#!/usr/bin/env python3
"""
Figuras finais para artigo (estilo claro SBC/IEEE).

Fonte principal: resultados_finais_v2/dataset_final_v2_days_filled.json
Saída: graficos_finais_artigo/*.png (300 DPI, bbox_inches='tight').

Tema visual centralizado: apply_article_theme(), add_article_grid(), style_axes_spines().

Semana: days_after_event // 7, limitada a 0–12 (sem semana 13).
Semanas sem observações nas séries temporais: NaN (sem valores inventados).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "resultados_finais_v2" / "dataset_final_v2_days_filled.json"
OUT_DIR = ROOT / "graficos_finais_artigo"

# Paleta oficial (todos os gráficos de sentimento)
COLORS = {
    "POSITIVO": "#1f77b4",
    "NEUTRO": "#7f7f7f",
    "NEGATIVO": "#d62728",
}

# Ordem de empilhamento (base → topo): negativo, neutro, positivo
STACK_ORDER = ["NEGATIVO", "NEUTRO", "POSITIVO"]
# Ordem de linhas alinhada às figuras de referência (linha temporal)
LINE_ORDER = ["NEGATIVO", "NEUTRO", "POSITIVO"]

LABEL_PT = {"POSITIVO": "Positivo", "NEUTRO": "Neutro", "NEGATIVO": "Negativo"}

# Cores por modelo (concordância — distintas da paleta de sentimento)
CONCORDANCIA_BAR_COLORS = {
    "llama": "#5b79a5",
    "phi": "#6b8e6e",
    "deepseek": "#8b6f62",
}

MODELS = ["gpt", "gemini", "copilot"]
MODEL_TITLE = {"gpt": "GPT", "gemini": "Gemini", "copilot": "Copilot"}

EVENT_KEYS = ("launch_initial", "latest_version")
EVENT_LABEL = {
    "launch_initial": "Lançamento inicial",
    "latest_version": "Versão recente",
}

FALLBACK_EVENT_DATES: dict[tuple[str, str], str] = {
    ("launch_initial", "gpt"): "2022-11-30",
    ("launch_initial", "gemini"): "2023-02-06",
    ("launch_initial", "copilot"): "2021-06-29",
    ("latest_version", "gpt"): "2024-05-13",
    ("latest_version", "gemini"): "2024-02-08",
    ("latest_version", "copilot"): "2023-11-01",
}

REQUIRED_COLS = [
    "model",
    "event",
    "final_sentiment",
    "tweet_id",
]
WEEKS = np.arange(0, 13, dtype=int)

# --- Tema visual único (SBC/IEEE): grade discreta, tipografia consistente ---
GRID_COLOR = "#888888"
GRID_MAJOR_ALPHA = 0.2
GRID_MAJOR_LW = 0.55
GRID_X_ALPHA = 0.16
GRID_X_LW = 0.45
GRID_MINOR_ALPHA = 0.12
GRID_MINOR_LW = 0.4


def apply_article_theme() -> None:
    """Define rcParams comuns a todas as figuras (fundo claro, sem grid global automático)."""
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
    minor_y: bool = False,
    vertical_softer: bool = True,
) -> None:
    """
    Grade suave (alpha ~0,15–0,25 no major).
    axes: 'x' | 'y' | 'both'
    """
    ax.set_axisbelow(True)
    alpha_y = GRID_MAJOR_ALPHA
    if axes in ("y", "both"):
        ax.grid(
            True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=GRID_MAJOR_LW,
            alpha=alpha_y,
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
    if minor_y and axes != "x":
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.grid(
            True,
            which="minor",
            axis="y",
            linestyle="--",
            linewidth=GRID_MINOR_LW,
            alpha=GRID_MINOR_ALPHA,
            color=GRID_COLOR,
        )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas obrigatórias ausentes: {missing}")


def load_raw() -> pd.DataFrame:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza datas, aplica fallback de event_date, recalcula semana 0–12."""
    out = df.copy()
    validate_columns(out)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

    def fallback_ts(row: pd.Series) -> pd.Timestamp | pd.NaTType:
        key = (str(row["event"]), str(row["model"]))
        s = FALLBACK_EVENT_DATES.get(key)
        return pd.Timestamp(s) if s else pd.NaT

    fb = out.apply(fallback_ts, axis=1)
    out["event_date"] = out["event_date"].fillna(fb)

    if "days_after_event" in out.columns:
        dcol = pd.to_numeric(out["days_after_event"], errors="coerce")
    else:
        dcol = pd.Series(np.nan, index=out.index)

    computed = (out["date"] - out["event_date"]).dt.days
    out["days_after_event"] = computed.where(computed.notna(), dcol)

    out = out[out["date"].notna() & out["event_date"].notna()]
    out = out[out["days_after_event"].notna()]
    out = out[out["days_after_event"] >= 0]

    # Regra: dias // 7, somente 0–12
    out["week_index"] = (out["days_after_event"] // 7).clip(lower=0, upper=12).astype(int)

    out["final_sentiment"] = out["final_sentiment"].astype(str).str.strip().str.upper()
    out = out[out["final_sentiment"].isin(STACK_ORDER)]

    out = out[out["model"].isin(MODELS)]
    out = out[out["event"].isin(EVENT_KEYS)]

    return out.reset_index(drop=True)


def validate_weeks(df: pd.DataFrame) -> None:
    bad = df[(df["week_index"] < 0) | (df["week_index"] > 12)]
    if len(bad):
        raise SystemExit(f"Inconsistência: {len(bad)} linhas com week_index fora de 0–12.")


def print_report(df_raw: pd.DataFrame, df: pd.DataFrame) -> None:
    print("=== Validação e resumo ===")
    print(f"Total de registros carregados: {len(df_raw):,}")
    print(f"Total após validação e filtros: {len(df):,}")
    print("\nTotal por modelo:")
    for m in MODELS:
        print(f"  {MODEL_TITLE[m]}: {int((df['model'] == m).sum()):,}")
    print("\nTotal por evento:")
    for ev in EVENT_KEYS:
        print(f"  {EVENT_LABEL[ev]}: {int((df['event'] == ev).sum()):,}")
    print("\nTotal por sentimento:")
    for s in STACK_ORDER:
        print(f"  {LABEL_PT[s]}: {int((df['final_sentiment'] == s).sum()):,}")

    wk = sorted(int(x) for x in df["week_index"].unique())
    print(f"\nSemanas encontradas (distintas, após regra 0–12): {wk}")
    if wk and (wk[0] < 0 or wk[-1] > 12):
        print("  AVISO: faixa inesperada.", file=sys.stderr)
    print()


def plot_distribuicao_global(df: pd.DataFrame, path: Path) -> None:
    """
    Dois painéis no estilo da referência:
    - esquerda: barras horizontais com contagens (N global);
    - direita: barras empilhadas 100% por momento de referência.
    """
    apply_article_theme()
    order_h = ["POSITIVO", "NEUTRO", "NEGATIVO"]
    vc = df["final_sentiment"].value_counts().reindex(order_h).fillna(0).astype(int)
    total = int(vc.sum())

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.35), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- Painel esquerdo: distribuição global (horizontal) ---
    ax0 = axes[0]
    y = np.arange(len(order_h))[::-1]
    counts = [int(vc[s]) for s in order_h]
    colors_h = [COLORS[s] for s in order_h]
    xmax = max(counts) if counts else 1

    ax0.barh(
        y,
        counts,
        color=colors_h,
        edgecolor="white",
        linewidth=0.55,
        height=0.68,
    )
    ax0.set_yticks(y)
    ax0.set_yticklabels([LABEL_PT[s] for s in order_h])
    ax0.set_xlabel("Número de tweets")
    ax0.set_title(f"Distribuição global\n(N = {total:,})")
    add_article_grid(ax0, axes="x", minor_y=False, vertical_softer=False)
    ax0.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for yi, c in zip(y, counts):
        ax0.text(
            c + xmax * 0.012,
            yi,
            f"{c:,}",
            va="center",
            fontsize=9,
            color="#212121",
        )

    # --- Painel direito: proporção por tipo de evento (100% empilhado) ---
    ax1 = axes[1]
    x = np.arange(len(EVENT_KEYS))
    bottom = np.zeros(len(EVENT_KEYS))

    for s in STACK_ORDER:
        heights = []
        for ev in EVENT_KEYS:
            sub = df[df["event"] == ev]
            n = len(sub)
            h = float((sub["final_sentiment"] == s).sum()) / n * 100.0 if n else 0.0
            heights.append(h)
        heights = np.array(heights, dtype=float)
        ax1.bar(
            x,
            heights,
            bottom=bottom,
            label=LABEL_PT[s],
            color=COLORS[s],
            edgecolor="white",
            linewidth=0.55,
            width=0.52,
        )
        for i, (b, h) in enumerate(zip(bottom, heights)):
            if h >= 5.5:
                ax1.text(
                    i,
                    b + h / 2,
                    f"{h:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if h >= 11 else "#222222",
                    fontweight="medium",
                )
        bottom += heights

    ax1.set_xticks(x)
    ax1.set_xticklabels([EVENT_LABEL[ev] for ev in EVENT_KEYS], fontsize=9)
    ax1.set_ylabel("Porcentagem (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Proporção por evento")
    add_article_grid(ax1, axes="y", minor_y=False)
    leg = ax1.legend(
        title="Sentimento",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
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
    style_axes_spines(ax0)
    style_axes_spines(ax1)

    fig.suptitle(
        "Distribuição de sentimentos no conjunto analisado",
        fontsize=11,
        fontweight="normal",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1.02])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparacao_eventos(df: pd.DataFrame, path: Path) -> None:
    """Duas barras empilhadas 100 %: lançamento inicial vs versão recente."""
    apply_article_theme()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    x = np.arange(len(EVENT_KEYS))
    bottom = np.zeros(len(EVENT_KEYS))

    for s in STACK_ORDER:
        heights = []
        for ev in EVENT_KEYS:
            sub = df[df["event"] == ev]
            n = len(sub)
            h = float((sub["final_sentiment"] == s).sum()) / n * 100.0 if n else 0.0
            heights.append(h)
        heights = np.array(heights, dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottom,
            label=LABEL_PT[s],
            color=COLORS[s],
            edgecolor="white",
            linewidth=0.6,
            width=0.52,
        )
        for i, (b, h) in enumerate(zip(bottom, heights)):
            if h >= 6:
                ax.text(
                    i,
                    b + h / 2,
                    f"{h:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if h >= 12 else "#222222",
                    fontweight="medium",
                )
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_LABEL[ev] for ev in EVENT_KEYS])
    ax.set_ylabel("Porcentagem (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Proporção de sentimentos por evento")
    add_article_grid(ax, axes="y", minor_y=False)
    style_axes_spines(ax)
    # Legenda fora da área das barras (evita cobrir % no segmento superior)
    leg = ax.legend(
        title="Sentimento",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=8,
        columnspacing=1.15,
        handletextpad=0.45,
        frameon=True,
        fancybox=False,
        edgecolor="#c0c0c0",
        facecolor="white",
    )
    leg.get_frame().set_linewidth(0.7)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distribuicao_por_modelo(df: pd.DataFrame, path: Path) -> None:
    """Três subplots; barras agrupadas por momento de referência."""
    apply_article_theme()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    x = np.arange(len(EVENT_KEYS))
    width = 0.22

    for ax, mod in zip(axes, MODELS):
        sub_mod = df[df["model"] == mod]
        for i, s in enumerate(STACK_ORDER):
            heights = []
            for ev in EVENT_KEYS:
                part = sub_mod[sub_mod["event"] == ev]
                n = len(part)
                heights.append(
                    float((part["final_sentiment"] == s).sum()) / n * 100.0 if n else 0.0
                )
            ax.bar(
                x + (i - 1) * width,
                heights,
                width,
                label=LABEL_PT[s],
                color=COLORS[s],
                edgecolor="#333333",
                linewidth=0.45,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([EVENT_LABEL[ev] for ev in EVENT_KEYS], fontsize=9)
        ax.set_title(MODEL_TITLE[mod])
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        add_article_grid(ax, axes="both", minor_y=False, vertical_softer=True)
        style_axes_spines(ax)

    axes[0].set_ylabel("Porcentagem (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig_leg = fig.legend(
        handles,
        labels,
        title="Sentimento",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=8,
        columnspacing=0.9,
        handletextpad=0.5,
        frameon=True,
        fancybox=False,
        edgecolor="#c0c0c0",
        facecolor="white",
    )
    fig_leg.get_frame().set_linewidth(0.7)
    fig.suptitle("Distribuição por tecnologia", y=1.05, fontsize=11, fontweight="normal")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def weekly_pct_aggregated_event(df: pd.DataFrame, event_key: str) -> pd.DataFrame:
    """Porcentagem por semana e sentimento, agregando GPT + Gemini + Copilot."""
    sub = df[(df["event"] == event_key) & (df["model"].isin(MODELS))]
    counts = (
        sub.groupby(["week_index", "final_sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    for s in LINE_ORDER:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[[s for s in LINE_ORDER]]
    counts = counts.reindex(list(WEEKS))
    totals = counts.sum(axis=1)
    pct = counts.div(totals.replace(0, np.nan), axis=0) * 100.0
    return pct


def temporal_annotation_points(pct: pd.DataFrame, sentiment_col: str) -> list[tuple[int, float]]:
    """
    Semanas 0, 6 e 12, mais as semanas do máximo e do mínimo da série
    (evita duplicar a mesma semana).
    """
    s = pct[sentiment_col]
    seen: set[int] = set()
    out: list[tuple[int, float]] = []
    for w in (0, 6, 12):
        if w in s.index and np.isfinite(s.loc[w]):
            out.append((w, float(s.loc[w])))
            seen.add(w)
    valid = [(int(i), float(s.loc[i])) for i in s.index if np.isfinite(s.loc[i])]
    if not valid:
        return sorted(out, key=lambda t: t[0])
    w_mx, v_mx = max(valid, key=lambda t: t[1])
    w_mn, v_mn = min(valid, key=lambda t: t[1])
    for w, v in ((w_mx, v_mx), (w_mn, v_mn)):
        if w not in seen:
            out.append((w, v))
            seen.add(w)
    return sorted(out, key=lambda t: t[0])


# Caixa leve nos rótulos temporais (legibilidade sobre linhas / grid)
TEXT_BBOX_TEMPORAL = {
    "facecolor": "white",
    "alpha": 0.6,
    "edgecolor": "none",
    "boxstyle": "round,pad=0.2",
}


def temporal_label_offset_points(sentiment: str, week: int) -> tuple[float, float]:
    """
    Deslocamento fixo por sentimento (offset points):
    - Neutro: acima da linha
    - Positivo: abaixo da linha
    - Negativo: lateral + leve acima (dx depende da semana para não colidir com bordas)
    """
    if sentiment == "NEUTRO":
        return (0.0, 14.0)
    if sentiment == "POSITIVO":
        return (0.0, -14.0)
    # NEGATIVO
    if week >= 8:
        return (-13.0, 11.0)
    if week <= 3:
        return (13.0, 11.0)
    return (12.0, 12.0)


def temporal_label_va_ha(sentiment: str) -> tuple[str, str]:
    """Alinhamento do texto em relação ao ponto de ancoragem + offset."""
    if sentiment == "POSITIVO":
        return ("top", "center")
    if sentiment == "NEUTRO":
        return ("bottom", "center")
    # NEGATIVO: leitura lateral
    return ("bottom", "center")


def plot_evolucao_temporal(
    df: pd.DataFrame, event_key: str, path: Path, title: str
) -> None:
    """Um único painel: tendência agregada (GPT + Gemini + Copilot), 3 linhas."""
    apply_article_theme()
    pct = weekly_pct_aggregated_event(df, event_key)
    weeks_plot = WEEKS.astype(float)

    fig, ax = plt.subplots(figsize=(10, 5.6))

    add_article_grid(ax, axes="both", minor_y=True, vertical_softer=True)

    for sent in LINE_ORDER:
        y = pct[sent].to_numpy(dtype=float)
        ax.plot(
            weeks_plot,
            y,
            marker="o",
            markersize=5.5,
            markeredgewidth=0.85,
            markeredgecolor="white",
            markerfacecolor=COLORS[sent],
            linewidth=2.5,
            label=LABEL_PT[sent],
            color=COLORS[sent],
            clip_on=False,
            solid_capstyle="round",
            antialiased=True,
            zorder=2,
        )
        va, ha = temporal_label_va_ha(sent)
        for w, val in temporal_annotation_points(pct, sent):
            dx, dy = temporal_label_offset_points(sent, w)
            ax.annotate(
                f"{val:.0f}%",
                xy=(w, val),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=7.5,
                color="#1a1a1a",
                bbox=TEXT_BBOX_TEMPORAL,
                zorder=4 + LINE_ORDER.index(sent),
            )

    ax.set_title(title, fontsize=12, fontweight="normal", pad=12)
    ax.set_xlabel("Semanas após o evento", fontsize=10, labelpad=7)
    ax.set_ylabel("Porcentagem (%)", fontsize=10, labelpad=7)
    ax.set_xlim(-0.35, 12.35)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(list(range(0, 13)))

    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=3,
        fontsize=8,
        columnspacing=0.85,
        handletextpad=0.45,
        title="Sentimento",
        title_fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="#c0c0c0",
        framealpha=1.0,
        facecolor="white",
    )
    leg.get_frame().set_linewidth(0.7)
    style_axes_spines(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_concordancia(df: pd.DataFrame, path: Path) -> None:
    """Barras por modelo do ensemble; cores distintas (não confundir com sentimento)."""
    apply_article_theme()
    cols = ["llama", "phi", "deepseek", "mistral"]
    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"Coluna ausente para concordância: {c}")

    sub = df.dropna(subset=cols)
    for c in cols:
        sub = sub[sub[c].isin(STACK_ORDER)]

    aux = ["llama", "phi", "deepseek"]
    labels_x = ["LLaMA", "Phi", "DeepSeek"]
    taxas = [
        float((sub[m] == sub["mistral"]).mean() * 100.0) if len(sub) else 0.0 for m in aux
    ]

    bar_colors = [CONCORDANCIA_BAR_COLORS[m] for m in aux]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(aux))
    bars = ax.bar(
        x,
        taxas,
        color=bar_colors,
        edgecolor="#3a3a3a",
        linewidth=0.5,
        width=0.55,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Concordância (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Concordância entre os modelos do ensemble e o Mistral")
    add_article_grid(ax, axes="y", minor_y=False)
    style_axes_spines(ax)

    for b, t in zip(bars, taxas):
        ax.text(
            b.get_x() + b.get_width() / 2,
            min(b.get_height() + 1.8, 98),
            f"{t:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"Arquivo não encontrado: {DATA_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw()
    validate_columns(df_raw)
    df = prepare_dataframe(df_raw)
    validate_weeks(df)
    print_report(df_raw, df)

    paths = {
        "distribuicao_global_sentimentos.png": OUT_DIR / "distribuicao_global_sentimentos.png",
        "comparacao_eventos.png": OUT_DIR / "comparacao_eventos.png",
        "distribuicao_por_modelo.png": OUT_DIR / "distribuicao_por_modelo.png",
        "evolucao_temporal_inicial.png": OUT_DIR / "evolucao_temporal_inicial.png",
        "evolucao_temporal_recente.png": OUT_DIR / "evolucao_temporal_recente.png",
        "concordancia_modelos.png": OUT_DIR / "concordancia_modelos.png",
    }

    plot_distribuicao_global(df, paths["distribuicao_global_sentimentos.png"])
    plot_comparacao_eventos(df, paths["comparacao_eventos.png"])
    plot_distribuicao_por_modelo(df, paths["distribuicao_por_modelo.png"])
    plot_evolucao_temporal(
        df,
        "launch_initial",
        paths["evolucao_temporal_inicial.png"],
        "Evolução temporal dos sentimentos — Lançamento inicial",
    )
    plot_evolucao_temporal(
        df,
        "latest_version",
        paths["evolucao_temporal_recente.png"],
        "Evolução temporal dos sentimentos — Versão recente",
    )
    plot_concordancia(df, paths["concordancia_modelos.png"])

    print("=== Gráficos gerados ===")
    for p in paths.values():
        print(f"  OK  {p.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
