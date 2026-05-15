#!/usr/bin/env python3
"""
Preenche days_after_event (e event_date quando ausente) a partir de (event, model),
exporta dataset_final_v2_days_filled.json e gera gráficos temporais (eventos iniciais vs recentes).
Semana relativa: week_index = min(days_after_event // 7, 12) — semanas 0–12.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = ROOT / "dataset_final_v2.json"
OUT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = OUT_DIR / "dataset_final_v2_days_filled.json"
FIG_INITIAL = OUT_DIR / "fig_temporal_eventos_iniciais.png"
FIG_RECENT = OUT_DIR / "fig_temporal_versoes_recentes.png"

SENT_ORDER = ["POSITIVO", "NEUTRO", "NEGATIVO"]
COLORS = {"POSITIVO": "#1f5f8b", "NEUTRO": "#6b6b6b", "NEGATIVO": "#c0392b"}

MODEL_TITLES = {
    "gpt": "GPT (ChatGPT)",
    "gemini": "Gemini / Bard",
    "copilot": "Copilot",
}

EVENT_FILTER_INITIAL = "launch_initial"
EVENT_FILTER_RECENT = "latest_version"


def _infer_event_date_map(df: pd.DataFrame) -> dict[tuple[str, str], pd.Timestamp]:
    known = df[df["event_date"].notna()].copy()
    if known.empty:
        return {}
    out: dict[tuple[str, str], pd.Timestamp] = {}
    for (ev, mod), g in known.groupby(["event", "model"]):
        vc = g["event_date"].value_counts()
        out[(ev, mod)] = pd.Timestamp(vc.index[0])
    return out


def fill_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    em_map = _infer_event_date_map(df)
    inferred = df.apply(lambda r: em_map.get((r["event"], r["model"]), pd.NaT), axis=1)
    df["event_date"] = df["event_date"].fillna(inferred)
    df["days_after_event"] = (df["date"] - df["event_date"]).dt.days
    df["week_index"] = (df["days_after_event"] // 7).clip(upper=12).astype(int)
    return df


def weekly_percentages(df: pd.DataFrame, event: str) -> pd.DataFrame:
    sub = df[df["event"] == event].copy()
    agg = (
        sub.groupby(["model", "week_index", "final_sentiment"])
        .size()
        .reset_index(name="n")
    )
    totals = (
        agg.groupby(["model", "week_index"])["n"]
        .sum()
        .reset_index(name="week_n")
    )
    agg = agg.merge(totals, on=["model", "week_index"], how="left")
    agg["pct"] = agg["n"] / agg["week_n"] * 100.0
    return agg


def smooth_series(y: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def plot_event_panels(
    df_weekly: pd.DataFrame,
    event: str,
    title: str,
    out_path: Path,
) -> None:
    models = ["gpt", "gemini", "copilot"]
    weeks = np.arange(0, 13)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    fig.suptitle(title, fontsize=12, fontweight="semibold", y=1.02)

    for ax, mod in zip(axes, models):
        ax.set_title(MODEL_TITLES[mod], fontsize=10)
        ax.set_xlabel("Semanas após o evento")
        if ax is axes[0]:
            ax.set_ylabel("Porcentagem (%)")
        ax.set_xlim(-0.2, 12.2)
        ax.set_ylim(0, 100)
        ax.set_xticks(list(range(0, 13)))

        for sent in SENT_ORDER:
            wk = df_weekly[
                (df_weekly["model"] == mod) & (df_weekly["final_sentiment"] == sent)
            ]
            if wk.empty:
                continue
            pivot = wk.set_index("week_index")["pct"].reindex(weeks)
            y = pivot.to_numpy(dtype=float)
            y_smooth = smooth_series(np.nan_to_num(y, nan=np.nan), window=3)
            ax.plot(weeks, y_smooth, label=sent, color=COLORS[sent], linewidth=2.2, alpha=0.95)

        ax.legend(title="Sentimento", fontsize=8, title_fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df_filled = fill_dataset(df)

    export = df_filled.copy()
    for col in ("date", "event_date"):
        export[col] = export[col].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None
        )
    export["days_after_event"] = export["days_after_event"].astype(int)
    export["week_index"] = export["week_index"].astype(int)
    records = export.replace({np.nan: None}).to_dict(orient="records")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    w_init = weekly_percentages(df_filled, EVENT_FILTER_INITIAL)
    w_rec = weekly_percentages(df_filled, EVENT_FILTER_RECENT)

    plot_event_panels(
        w_init,
        EVENT_FILTER_INITIAL,
        "Evolução temporal dos sentimentos — eventos iniciais\n"
        "(semanas 0–12 após o evento)",
        FIG_INITIAL,
    )
    plot_event_panels(
        w_rec,
        EVENT_FILTER_RECENT,
        "Evolução temporal dos sentimentos — versões recentes\n"
        "(semanas 0–12 após o evento)",
        FIG_RECENT,
    )

    n = len(df_filled)
    print(f"Salvo: {OUTPUT_JSON} ({n} registros)")
    print(f"Gráfico A: {FIG_INITIAL}")
    print(f"Gráfico B: {FIG_RECENT}")
    print("days_after_event min/max:", int(df_filled["days_after_event"].min()), int(df_filled["days_after_event"].max()))
    print("week_index max:", int(df_filled["week_index"].max()))


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
