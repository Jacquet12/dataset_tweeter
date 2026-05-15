#!/usr/bin/env python3
"""
Evolução temporal da proporção de final_sentiment por tecnologia (GPT, Gemini, Copilot).

Cada figura: dois subplots lado a lado (lançamento inicial | versão recente),
cada um com três linhas (Positivo, Negativo, Neutro). Suavização por média móvel.

Dados: data/reference_labels_final_complete.json
Estilo alinhado ao gráfico temporal clássico do TCC (PDF/LaTeX, 300 DPI).
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

DAYS = np.arange(0, 91, dtype=int)

MODELS: list[str] = ["gpt", "gemini", "copilot"]
MODEL_DISPLAY = {"gpt": "GPT", "gemini": "Gemini", "copilot": "Copilot"}

EVENT_LEFT = "launch_initial"
EVENT_RIGHT = "latest_version"
EVENT_TITLE = {
    EVENT_LEFT: "Lançamento inicial",
    EVENT_RIGHT: "Versão recente",
}

OUTPUT_BY_MODEL: dict[str, Path] = {
    "gpt": REPO_ROOT / "results" / "figures" / "gpt_temporal_sentiment.png",
    "gemini": REPO_ROOT / "results" / "figures" / "gemini_temporal_sentiment.png",
    "copilot": REPO_ROOT / "results" / "figures" / "copilot_temporal_sentiment.png",
}

# Mesma paleta de sentimentos que generate_general_sentiment_distribution.py
SENTIMENT_ORDER = ["POSITIVO", "NEGATIVO", "NEUTRO"]
SENTIMENT_COLOR = {
    "POSITIVO": "#1f77b4",
    "NEGATIVO": "#d62728",
    "NEUTRO": "#7f7f7f",
}
SENTIMENT_LABEL_PT = {
    "POSITIVO": "Positivo",
    "NEGATIVO": "Negativo",
    "NEUTRO": "Neutro",
}

ROLLING_WINDOW_DAYS = 7


def apply_temporal_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.size": 10,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": False,
        }
    )


def style_temporal_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        linewidth=0.85,
        alpha=0.55,
        color="#888888",
    )


def _accumulate_row(
    r: dict,
    allowed: set[tuple[str, str]],
    out: dict[tuple[str, str], dict[int, Counter[str]]],
) -> None:
    model = str(r.get("model") or "").strip().lower()
    event = str(r.get("event") or "").strip().lower()
    key = (model, event)
    if key not in allowed:
        return
    d_raw = r.get("days_after_event")
    if d_raw is None:
        return
    try:
        di = int(d_raw)
    except (TypeError, ValueError):
        return
    if di < 0 or di > 90:
        return
    s = str(r.get("final_sentiment") or "").strip().upper()
    if s not in SENTIMENT_ORDER:
        return
    out[key][di][s] += 1


def load_day_counts(path: Path) -> dict[tuple[str, str], dict[int, Counter[str]]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("JSON deve ser uma lista de objetos")

    allowed = {(m, e) for m in MODELS for e in (EVENT_LEFT, EVENT_RIGHT)}
    out: dict[tuple[str, str], dict[int, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for r in rows:
        if isinstance(r, dict):
            _accumulate_row(r, allowed, out)

    return dict(out)


def daily_proportions_three(
    day_counts: dict[tuple[str, str], dict[int, Counter[str]]],
    model: str,
    event: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Três vetores (91,) em % para POS, NEG, NEU; NaN se não houver tweets nesse dia."""
    dc = day_counts.get((model, event), {})
    yp = np.full(91, np.nan, dtype=float)
    yn = np.full(91, np.nan, dtype=float)
    yu = np.full(91, np.nan, dtype=float)
    for d in DAYS:
        ctr = dc.get(int(d), Counter())
        tot = sum(ctr.values())
        if tot <= 0:
            continue
        yp[int(d)] = 100.0 * float(ctr["POSITIVO"]) / float(tot)
        yn[int(d)] = 100.0 * float(ctr["NEGATIVO"]) / float(tot)
        yu[int(d)] = 100.0 * float(ctr["NEUTRO"]) / float(tot)
    return yp, yn, yu


def rolling_mean_ignore_nan(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    half = window // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = y[lo:hi]
        valid = chunk[np.isfinite(chunk)]
        if valid.size == 0:
            out[i] = np.nan
        else:
            out[i] = float(np.mean(valid))
    return out


def smooth_series(y_raw: np.ndarray, window: int) -> np.ndarray:
    return np.clip(rolling_mean_ignore_nan(y_raw, window), 0.0, 100.0)


def count_tweets_event(
    day_counts: dict[tuple[str, str], dict[int, Counter[str]]],
    model: str,
    event: str,
) -> int:
    dc = day_counts.get((model, event), {})
    return int(sum(sum(dc[d].values()) for d in dc))


def count_finite(y: np.ndarray) -> int:
    return int(np.sum(np.isfinite(y)))


def _plot_three_lines(ax, y_pos: np.ndarray, y_neg: np.ndarray, y_neu: np.ndarray) -> list:
    handles = []
    for y, sent in zip(
        (y_pos, y_neg, y_neu),
        SENTIMENT_ORDER,
        strict=True,
    ):
        color = SENTIMENT_COLOR[sent]
        (ln,) = ax.plot(
            DAYS,
            y,
            color=color,
            linewidth=1.75,
            linestyle="-",
            marker="o",
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            alpha=0.95,
            label=SENTIMENT_LABEL_PT[sent],
        )
        handles.append(ln)
    ax.set_xlim(-0.5, 90.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlabel("Dias após o evento (0–90)", fontsize=10)
    style_temporal_ax(ax)
    return handles


def plot_technology_figure(
    model: str,
    day_counts: dict[tuple[str, str], dict[int, Counter[str]]],
    path: Path,
) -> tuple[list[int], list[int]]:
    """Devolve (pontos válidos subplot esq., pontos válidos subplot dir.) por série [pos, neg, neu]."""
    apply_temporal_theme()
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.05),
        sharey=True,
        constrained_layout=False,
    )

    name = MODEL_DISPLAY[model]

    ypl, ynl, yul = daily_proportions_three(day_counts, model, EVENT_LEFT)
    ypr, ynr, yur = daily_proportions_three(day_counts, model, EVENT_RIGHT)

    spl = smooth_series(ypl, ROLLING_WINDOW_DAYS), smooth_series(
        ynl, ROLLING_WINDOW_DAYS
    ), smooth_series(yul, ROLLING_WINDOW_DAYS)
    spr = smooth_series(ypr, ROLLING_WINDOW_DAYS), smooth_series(
        ynr, ROLLING_WINDOW_DAYS
    ), smooth_series(yur, ROLLING_WINDOW_DAYS)

    handles = _plot_three_lines(ax_l, spl[0], spl[1], spl[2])
    _plot_three_lines(ax_r, spr[0], spr[1], spr[2])

    ax_l.set_title(EVENT_TITLE[EVENT_LEFT], fontsize=10.5, fontweight="normal", pad=8)
    ax_r.set_title(EVENT_TITLE[EVENT_RIGHT], fontsize=10.5, fontweight="normal", pad=8)
    ax_l.set_ylabel("Proporção (%)", fontsize=10)

    labels = [SENTIMENT_LABEL_PT[s] for s in SENTIMENT_ORDER]
    # rect: margens para título principal (topo), legenda (base) e rótulos
    fig.tight_layout(rect=[0.03, 0.19, 0.97, 0.76])
    st = fig.suptitle(
        f"{name} — evolução temporal da referência (final_sentiment)",
        fontsize=11,
        fontweight="normal",
        y=0.92,
    )
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#b0b0b0",
        fontsize=9.0,
        columnspacing=1.35,
        handlelength=2.4,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
        bbox_extra_artists=[st, leg],
    )
    plt.close(fig)

    pts_l = [count_finite(spl[i]) for i in range(3)]
    pts_r = [count_finite(spr[i]) for i in range(3)]
    return pts_l, pts_r


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"ERRO: {DATA_PATH} não encontrado", file=sys.stderr)
        return 1

    print("=" * 72)
    print("Séries temporais — final_sentiment (por tecnologia, 2 eventos × 3 sentimentos)")
    print("=" * 72)
    print(f"Fonte: {DATA_PATH.relative_to(REPO_ROOT)}")
    print()
    print("Suavização: média móvel centrada (só dias com tweets entram na média da janela).")
    print(f"Janela de suavização: {ROLLING_WINDOW_DAYS} dias")
    print()

    day_counts = load_day_counts(DATA_PATH)

    print("Quantidade de tweets por evento (total no intervalo 0–90 dias):")
    for model in MODELS:
        mn = MODEL_DISPLAY[model]
        n_left = count_tweets_event(day_counts, model, EVENT_LEFT)
        n_right = count_tweets_event(day_counts, model, EVENT_RIGHT)
        print(
            f"  {mn:8s}  {EVENT_TITLE[EVENT_LEFT]:22s}  {n_left:5,}  |  "
            f"{EVENT_TITLE[EVENT_RIGHT]:18s}  {n_right:5,}"
        )
    print()

    print("Pontos válidos por série (após suavização, valores finitos no eixo Y; esperado 91):")
    for model in MODELS:
        mn = MODEL_DISPLAY[model]
        pts_l, pts_r = plot_technology_figure(model, day_counts, OUTPUT_BY_MODEL[model])
        print(f"  [{mn}] subplot esquerdo (lançamento inicial):  Pos={pts_l[0]}  Neg={pts_l[1]}  Neu={pts_l[2]}")
        print(f"         subplot direito (versão recente):     Pos={pts_r[0]}  Neg={pts_r[1]}  Neu={pts_r[2]}")
    print()

    print("Figuras geradas:")
    for model in MODELS:
        p = OUTPUT_BY_MODEL[model]
        abs_p = p.resolve()
        print(f"  {p.relative_to(REPO_ROOT)}")
        print(f"    → {abs_p}")
        if abs_p.is_file():
            print(f"    Tamanho: {abs_p.stat().st_size:,} bytes")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
