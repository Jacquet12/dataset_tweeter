#!/usr/bin/env python3
"""
Gera rótulos de referência por votação majoritária entre LLaMA, Phi e DeepSeek.

O Mistral não entra na votação; permanece no registo para avaliação posterior.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

INPUT_JSON = DATA_DIR / "consolidated_predictions.json"
OUTPUT_REFERENCE = DATA_DIR / "reference_labels.json"
OUTPUT_MANUAL_CSV = DATA_DIR / "manual_review.csv"

VOTE_KEYS = ("llama_sentiment", "phi_sentiment", "deepseek_sentiment")
VALID_LABELS = frozenset({"POSITIVO", "NEGATIVO", "NEUTRO"})

NEW_FIELDS = ("final_sentiment", "agreement_level", "needs_manual_review")


def _normalize_vote(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        return v if v in VALID_LABELS else None
    return None


def majority_vote_labels(
    llama: Any, phi: Any, deepseek: Any
) -> tuple[str | None, int | None, bool]:
    """
    Retorna (final_sentiment, agreement_level, needs_manual_review).

    agreement_level: 3 = unânime; 2 = maioria 2x1; 1 = três rótulos distintos;
    None = voto incompleto ou rótulo inválido em algum modelo.

    needs_manual_review é True apenas quando os três modelos discordam (nível 1).
    """
    votes = (
        _normalize_vote(llama),
        _normalize_vote(phi),
        _normalize_vote(deepseek),
    )
    if any(v is None for v in votes):
        return None, None, False

    counts = Counter(votes)
    n_distinct = len(counts)
    most_common = counts.most_common(1)[0][0]

    if n_distinct == 1:
        return most_common, 3, False
    if n_distinct == 2:
        return most_common, 2, False
    # três rótulos diferentes
    return None, 1, True


def enrich_record(row: dict[str, Any]) -> dict[str, Any]:
    """Preserva todos os campos e acrescenta final_sentiment, agreement_level, needs_manual_review."""
    final, level, manual = majority_vote_labels(
        row.get("llama_sentiment"),
        row.get("phi_sentiment"),
        row.get("deepseek_sentiment"),
    )
    out = {**row}
    out["final_sentiment"] = final
    out["agreement_level"] = level
    out["needs_manual_review"] = manual
    return out


def print_statistics(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    n_full = sum(1 for r in rows if r["agreement_level"] == 3)
    n_partial = sum(1 for r in rows if r["agreement_level"] == 2)
    n_split = sum(1 for r in rows if r["agreement_level"] == 1)
    n_incomplete = sum(1 for r in rows if r["agreement_level"] is None)

    def pct(n: int) -> float:
        return (100.0 * n / total) if total else 0.0

    print(f"Total de tweets: {total}")
    print()
    print("Concordância (LLaMA, Phi, DeepSeek):")
    print(f"  Total (3 iguais):     {n_full:6d}  ({pct(n_full):5.2f}%)")
    print(f"  Parcial (2 iguais):   {n_partial:6d}  ({pct(n_partial):5.2f}%)")
    print(f"  Empate (3 distintos): {n_split:6d}  ({pct(n_split):5.2f}%)")
    if n_incomplete:
        print(f"  Incompleto/inválido:  {n_incomplete:6d}  ({pct(n_incomplete):5.2f}%)")
    n_manual_flag = sum(1 for r in rows if r["needs_manual_review"])
    print(f"  needs_manual_review:  {n_manual_flag:6d}  ({pct(n_manual_flag):5.2f}%)")
    print()
    print("Distribuição de final_sentiment (apenas rótulo definido):")
    finals = [r["final_sentiment"] for r in rows if r["final_sentiment"] is not None]
    fc = Counter(finals)
    for label in sorted(VALID_LABELS):
        c = fc.get(label, 0)
        denom = len(finals) if finals else 0
        p = (100.0 * c / denom) if denom else 0.0
        print(f"  {label:10s}  {c:6d}  ({p:5.2f}% do subconjunto com rótulo final)")


def write_manual_review_csv(rows: list[dict[str, Any]], path: Path) -> int:
    fieldnames = [
        "tweet_id",
        "content",
        "llama_sentiment",
        "phi_sentiment",
        "deepseek_sentiment",
        "final_sentiment",
    ]
    manual_rows = [r for r in rows if r["needs_manual_review"]]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in manual_rows:
            w.writerow(
                {
                    "tweet_id": r.get("tweet_id", ""),
                    "content": r.get("content", ""),
                    "llama_sentiment": r.get("llama_sentiment", ""),
                    "phi_sentiment": r.get("phi_sentiment", ""),
                    "deepseek_sentiment": r.get("deepseek_sentiment", ""),
                    "final_sentiment": "",
                }
            )
    return len(manual_rows)


def main() -> None:
    if not INPUT_JSON.is_file():
        raise FileNotFoundError(f"Entrada não encontrada: {INPUT_JSON}")

    with INPUT_JSON.open(encoding="utf-8") as f:
        consolidated: list[dict[str, Any]] = json.load(f)

    enriched = [enrich_record(r) for r in consolidated]

    print_statistics(enriched)
    n_manual = write_manual_review_csv(enriched, OUTPUT_MANUAL_CSV)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_REFERENCE.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print(f"Linhas em {OUTPUT_MANUAL_CSV.relative_to(REPO_ROOT)} (revisão manual): {n_manual}")
    print(f"Salvo: {OUTPUT_REFERENCE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
