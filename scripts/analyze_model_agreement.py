#!/usr/bin/env python3
"""
Análise de concordância entre os rotuladores automáticos LLaMA, Phi e DeepSeek.

Usa data/reference_labels.json (campos llama_sentiment, phi_sentiment, deepseek_sentiment).
Não incorpora revisão manual de empates; não usa Mistral.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

INPUT_JSON = DATA_DIR / "reference_labels.json"
OUTPUT_JSON = DATA_DIR / "agreement_analysis.json"
OUTPUT_PAIRWISE_CSV = DATA_DIR / "pairwise_agreement.csv"
OUTPUT_DISAGREE_CLASS_CSV = DATA_DIR / "disagreement_by_class.csv"

LABELS = ("POSITIVO", "NEGATIVO", "NEUTRO")
COL_LLAMA = "llama_sentiment"
COL_PHI = "phi_sentiment"
COL_DS = "deepseek_sentiment"

PAIRS: tuple[tuple[str, str, str], ...] = (
    ("llama_vs_phi", COL_LLAMA, COL_PHI),
    ("llama_vs_deepseek", COL_LLAMA, COL_DS),
    ("phi_vs_deepseek", COL_PHI, COL_DS),
)

RANDOM_SEED = 42
N_EXAMPLES = 8


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Esperado array JSON na raiz")
    return data


def pairwise_stats(
    rows: list[dict[str, Any]], col_a: str, col_b: str
) -> dict[str, Any]:
    n_total = 0
    n_agree = 0
    for r in rows:
        a, b = r.get(col_a), r.get(col_b)
        if a not in LABELS or b not in LABELS:
            continue
        n_total += 1
        if a == b:
            n_agree += 1
    n_disagree = n_total - n_agree
    pct = (100.0 * n_agree / n_total) if n_total else 0.0
    return {
        "n_total_valid": n_total,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "pct_agree": round(pct, 4),
        "pct_disagree": round(100.0 - pct, 4) if n_total else 0.0,
    }


def stratified_agreement_by_class(
    rows: list[dict[str, Any]], col_anchor: str, col_other: str
) -> list[dict[str, Any]]:
    """
    Para cada classe C, entre tweets com anchor==C, quantos concordam com other==C.
    """
    out: list[dict[str, Any]] = []
    for c in LABELS:
        n_anchor = 0
        n_agree = 0
        for r in rows:
            av, ov = r.get(col_anchor), r.get(col_other)
            if av not in LABELS or ov not in LABELS:
                continue
            if av != c:
                continue
            n_anchor += 1
            if ov == c:
                n_agree += 1
        pct = (100.0 * n_agree / n_anchor) if n_anchor else 0.0
        out.append(
            {
                "anchor_column": col_anchor,
                "other_column": col_other,
                "sentiment_class": c,
                "n_tweets_anchor_class": n_anchor,
                "n_agree_same_class": n_agree,
                "n_disagree": n_anchor - n_agree,
                "pct_agree_within_anchor_class": round(pct, 4),
            }
        )
    return out


def full_disagreement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("agreement_level") == 1]


def triple_pattern_key(r: dict[str, Any]) -> tuple[str, str, str] | None:
    a, p, d = r.get(COL_LLAMA), r.get(COL_PHI), r.get(COL_DS)
    if a not in LABELS or p not in LABELS or d not in LABELS:
        return None
    return (a, p, d)


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairwise: list[dict[str, Any]] = []
    for pair_id, ca, cb in PAIRS:
        st = pairwise_stats(rows, ca, cb)
        pairwise.append(
            {
                "pair": pair_id,
                "column_a": ca,
                "column_b": cb,
                **st,
            }
        )

    stratified: dict[str, list[dict[str, Any]]] = {}
    for pair_id, ca, cb in PAIRS:
        stratified[pair_id] = stratified_agreement_by_class(rows, ca, cb)

    triple_dis = full_disagreement_rows(rows)
    n_all = len(rows)
    n_triple = len(triple_dis)
    pct_triple = (100.0 * n_triple / n_all) if n_all else 0.0

    pattern_counts: Counter[tuple[str, str, str]] = Counter()
    for r in triple_dis:
        k = triple_pattern_key(r)
        if k:
            pattern_counts[k] += 1

    rng = random.Random(RANDOM_SEED)
    pool = [r for r in triple_dis if r.get("tweet_id") and r.get("content") is not None]
    k_sample = min(N_EXAMPLES, len(pool))
    examples = []
    if k_sample:
        for r in rng.sample(pool, k=k_sample):
            examples.append(
                {
                    "tweet_id": r.get("tweet_id"),
                    "content": (r.get("content") or "")[:400],
                    COL_LLAMA: r.get(COL_LLAMA),
                    COL_PHI: r.get(COL_PHI),
                    COL_DS: r.get(COL_DS),
                }
            )

    return {
        "source_file": str(INPUT_JSON.relative_to(REPO_ROOT)),
        "n_tweets": n_all,
        "pairwise": pairwise,
        "stratified_agreement_by_anchor_class": stratified,
        "triple_disagreement": {
            "agreement_level": 1,
            "count": n_triple,
            "percentage_of_all_tweets": round(pct_triple, 4),
            "ordered_triplet_counts": {
                f"{a}|{p}|{d}": c for (a, p, d), c in sorted(pattern_counts.items())
            },
            "n_distinct_patterns": len(pattern_counts),
            "random_examples": examples,
        },
    }


def write_pairwise_csv(pairwise: Iterable[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "pair",
        "column_a",
        "column_b",
        "n_total_valid",
        "n_agree",
        "n_disagree",
        "pct_agree",
        "pct_disagree",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in pairwise:
            w.writerow(row)


def write_disagreement_by_class_csv(stratified: dict[str, list[dict[str, Any]]], path: Path) -> None:
    fieldnames = [
        "pair",
        "anchor_column",
        "other_column",
        "sentiment_class",
        "n_tweets_anchor_class",
        "n_agree_same_class",
        "n_disagree",
        "pct_agree_within_anchor_class",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for pair_id, rows in stratified.items():
            for r in rows:
                w.writerow({"pair": pair_id, **r})


def print_terminal_report(analysis: dict[str, Any]) -> None:
    pw = analysis["pairwise"]
    print("=" * 72)
    print("CONCORDÂNCIA PAR A PAR (LLaMA, Phi, DeepSeek)")
    print("=" * 72)
    for p in pw:
        print(
            f"  {p['pair']:<22}  concorda: {p['n_agree']:5d} / {p['n_total_valid']:5d}  "
            f"({p['pct_agree']:.2f}%)  |  discorda: {p['n_disagree']:5d}"
        )

    best = max(pw, key=lambda x: x["pct_agree"])
    worst = min(pw, key=lambda x: x["pct_agree"])
    print()
    print(f"Maior concordância:   {best['pair']} ({best['pct_agree']:.2f}%)")
    print(f"Menor concordância:   {worst['pair']} ({worst['pct_agree']:.2f}%)")

    td = analysis["triple_disagreement"]
    print()
    print("=" * 72)
    print("DISCORDÂNCIA TOTAL (agreement_level == 1, três rótulos distintos)")
    print("=" * 72)
    print(f"  Quantidade:   {td['count']}")
    print(f"  Percentual:   {td['percentage_of_all_tweets']:.2f}% dos tweets")
    print(f"  Padrões (L|Phi|DS) distintos: {td['n_distinct_patterns']}")
    print()
    print("  Distribuição por tripla ordenada (topo = mais frequente):")
    items = sorted(
        td["ordered_triplet_counts"].items(), key=lambda kv: kv[1], reverse=True
    )
    for key, cnt in items:
        pct = (100.0 * cnt / td["count"]) if td["count"] else 0
        print(f"    {key:40s}  {cnt:5d}  ({pct:.2f}% dos casos nível 1)")

    print()
    print("=" * 72)
    print(f"EXEMPLOS ALEATÓRIOS (seed={RANDOM_SEED}, n={len(td['random_examples'])})")
    print("=" * 72)
    for i, ex in enumerate(td["random_examples"], 1):
        print(f"\n--- Exemplo {i} | {ex.get('tweet_id')} ---")
        print(f"  LLaMA:    {ex.get(COL_LLAMA)}")
        print(f"  Phi:      {ex.get(COL_PHI)}")
        print(f"  DeepSeek: {ex.get(COL_DS)}")
        body = (ex.get("content") or "").replace("\n", " ")
        if len(body) > 280:
            body = body[:277] + "..."
        print(f"  Texto:    {body}")

    print()
    print("=" * 72)
    print("CONCORDÂNCIA POR CLASSE (estratificação: rótulo da 1ª coluna do par)")
    print("=" * 72)
    for pair_id, strat_rows in analysis["stratified_agreement_by_anchor_class"].items():
        print(f"\n  [{pair_id}]")
        for r in strat_rows:
            print(
                f"    {r['sentiment_class']:<10}  anchor_n={r['n_tweets_anchor_class']:5d}  "
                f"concorda={r['n_agree_same_class']:5d}  ({r['pct_agree_within_anchor_class']:.2f}%)"
            )


def main() -> None:
    if not INPUT_JSON.is_file():
        raise FileNotFoundError(INPUT_JSON)

    rows = load_rows(INPUT_JSON)
    analysis = analyze(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
        f.write("\n")

    write_pairwise_csv(analysis["pairwise"], OUTPUT_PAIRWISE_CSV)
    write_disagreement_by_class_csv(
        analysis["stratified_agreement_by_anchor_class"], OUTPUT_DISAGREE_CLASS_CSV
    )

    print_terminal_report(analysis)

    print()
    print("Ficheiros escritos:")
    print(f"  {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"  {OUTPUT_PAIRWISE_CSV.relative_to(REPO_ROOT)}")
    print(f"  {OUTPUT_DISAGREE_CLASS_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
