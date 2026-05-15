#!/usr/bin/env python3
"""
1) Lê data/manual_review.csv (apenas casos agreement_level == 1 exportados antes).
2) Atribui manual_sentiment analisando só o campo content (sem usar votos dos modelos).
3) Grava data/manual_review_completed.csv
4) Atualiza data/reference_labels.json → data/reference_labels_final.json:
   preenche manual_sentiment, final_sentiment e needs_manual_review nos casos resolvidos.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from manual_review_semantics import classify_from_content_only

REPO_ROOT = _SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

INPUT_MANUAL = DATA_DIR / "manual_review.csv"
OUTPUT_COMPLETED = DATA_DIR / "manual_review_completed.csv"
INPUT_REFERENCE = DATA_DIR / "reference_labels.json"
OUTPUT_REFERENCE_FINAL = DATA_DIR / "reference_labels_final.json"

N_EXAMPLES = 6
EXAMPLE_SEED = 17


def read_manual_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_completed_csv(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    fieldnames = [
        "tweet_id",
        "content",
        "llama_sentiment",
        "phi_sentiment",
        "deepseek_sentiment",
        "final_sentiment",
        "manual_sentiment",
    ]
    out_rows: list[dict[str, str]] = []
    for r in rows:
        content = r.get("content") or ""
        label = classify_from_content_only(content)
        out_rows.append(
            {
                "tweet_id": r.get("tweet_id", ""),
                "content": content,
                "llama_sentiment": r.get("llama_sentiment", ""),
                "phi_sentiment": r.get("phi_sentiment", ""),
                "deepseek_sentiment": r.get("deepseek_sentiment", ""),
                "final_sentiment": r.get("final_sentiment", ""),
                "manual_sentiment": label,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    return out_rows


def print_manual_distribution(rows: list[dict[str, str]], title: str) -> Counter[str]:
    c = Counter(r["manual_sentiment"] for r in rows)
    print(title)
    total = sum(c.values()) or 1
    for lab in ("POSITIVO", "NEGATIVO", "NEUTRO"):
        n = c.get(lab, 0)
        print(f"  {lab:12s}  {n:5d}  ({100.0 * n / total:5.2f}%)")
    print(f"  Total:        {sum(c.values())}")
    return c


def print_examples(rows: list[dict[str, str]]) -> None:
    rng = random.Random(EXAMPLE_SEED)
    sample = rng.sample(rows, k=min(N_EXAMPLES, len(rows)))
    print()
    print(f"Exemplos (seed={EXAMPLE_SEED}, n={len(sample)}):")
    for i, r in enumerate(sample, 1):
        body = (r.get("content") or "").replace("\n", " ")
        if len(body) > 220:
            body = body[:217] + "..."
        print(f"\n  [{i}] {r.get('tweet_id')} → manual_sentiment={r.get('manual_sentiment')}")
        print(f"      {body}")


def merge_reference(
    reference: list[dict[str, Any]],
    manual_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in reference:
        r = deepcopy(row)
        tid = str(r.get("tweet_id", ""))
        if tid not in manual_by_id:
            r["manual_sentiment"] = None
            out.append(r)
            continue
        label = manual_by_id[tid]
        r["manual_sentiment"] = label
        r["final_sentiment"] = label
        r["needs_manual_review"] = False
        out.append(r)
    return out


def print_final_stats(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    finals = [r.get("final_sentiment") for r in rows]
    c = Counter(x for x in finals if x is not None)
    nulls = sum(1 for x in finals if x is None)
    print()
    print("=" * 60)
    print("REFERENCE LABELS (FINAL)")
    print("=" * 60)
    print(f"Tweets totais:              {total}")
    print(f"Sem final_sentiment (null): {nulls}")
    denom = total - nulls or 1
    print()
    print("Distribuição de final_sentiment (apenas definidos):")
    for lab in ("POSITIVO", "NEGATIVO", "NEUTRO"):
        n = c.get(lab, 0)
        print(f"  {lab:12s}  {n:5d}  ({100.0 * n / denom:5.2f}% do subconjunto com rótulo)")
    print()
    print(f"Soma com rótulo: {sum(c.values())}")


def main() -> None:
    if not INPUT_MANUAL.is_file():
        raise FileNotFoundError(INPUT_MANUAL)
    if not INPUT_REFERENCE.is_file():
        raise FileNotFoundError(INPUT_REFERENCE)

    manual_rows = read_manual_review_rows(INPUT_MANUAL)
    completed = write_completed_csv(manual_rows, OUTPUT_COMPLETED)

    print("=" * 60)
    print("REVISÃO MANUAL SIMULADA (só campo content)")
    print("=" * 60)
    print_manual_distribution(completed, "Distribuição de manual_sentiment:")
    print_examples(completed)

    manual_by_id = {r["tweet_id"]: r["manual_sentiment"] for r in completed}

    with INPUT_REFERENCE.open(encoding="utf-8") as f:
        reference = json.load(f)

    merged = merge_reference(reference, manual_by_id)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_REFERENCE_FINAL.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_final_stats(merged)

    print()
    print("Ficheiros:")
    print(f"  {OUTPUT_COMPLETED.relative_to(REPO_ROOT)}")
    print(f"  {OUTPUT_REFERENCE_FINAL.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
