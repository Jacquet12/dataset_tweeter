#!/usr/bin/env python3
"""
Consolida classificações de sentimento dos quatro modelos (Mistral, LLaMA, Phi, DeepSeek)
num único dataset, unido por tweet_id.

Não aplica votação majoritária nem remove tweets sem classificação completa.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED_DIR = REPO_ROOT / "classifiedSentiments"
DATA_DIR = REPO_ROOT / "data"

MODEL_FILES: list[tuple[str, Path]] = [
    ("mistral_sentiment", CLASSIFIED_DIR / "mistral" / "classified_sentiment_mistral.json"),
    ("llama_sentiment", CLASSIFIED_DIR / "llama" / "classified_sentiment_llama.json"),
    ("phi_sentiment", CLASSIFIED_DIR / "phi" / "classified_sentiment_phi4.json"),
    ("deepseek_sentiment", CLASSIFIED_DIR / "deepseek" / "classified_sentiment_deepseek.json"),
]

METADATA_KEYS = [
    "tweet_id",
    "content",
    "date",
    "event",
    "event_date",
    "days_after_event",
    "model",
    "source",
]

SENTIMENT_COLS = [name for name, _ in MODEL_FILES]

OUTPUT_CSV = DATA_DIR / "consolidated_predictions.csv"
OUTPUT_JSON = DATA_DIR / "consolidated_predictions.json"


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Esperado lista JSON em {path}")
    return data


def _rows_by_tweet_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = row.get("tweet_id")
        if tid is None:
            continue
        if tid in out:
            # Mantém primeira ocorrência (JSONs já deduplicados; evita sobrescrever)
            continue
        out[str(tid)] = row
    return out


def _pick_metadata(
    tid: str, by_model: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Metadados: primeira linha não vazia na ordem Mistral → LLaMA → Phi → DeepSeek."""
    order = ["mistral_sentiment", "llama_sentiment", "phi_sentiment", "deepseek_sentiment"]
    for key in order:
        row = by_model[key].get(tid)
        if not row:
            continue
        return {k: row.get(k) for k in METADATA_KEYS}
    return {k: tid if k == "tweet_id" else None for k in METADATA_KEYS}


def main() -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    all_ids: set[str] = set()

    for col_name, path in MODEL_FILES:
        rows = _load_json(path)
        m = _rows_by_tweet_id(rows)
        by_model[col_name] = m
        all_ids |= set(m.keys())
        print(f"Carregado {col_name}: {len(rows)} linhas -> {len(m)} tweet_id únicos ({path.relative_to(REPO_ROOT)})")

    ordered_ids = sorted(all_ids)
    consolidated: list[dict[str, Any]] = []

    for tid in ordered_ids:
        meta = _pick_metadata(tid, by_model)
        record: dict[str, Any] = {**meta}
        for col_name, _path in MODEL_FILES:
            row = by_model[col_name].get(tid)
            record[col_name] = row.get("sentiment") if row else None
        consolidated.append(record)

    total = len(consolidated)
    missing_any = sum(
        1
        for r in consolidated
        if any(r[c] is None for c in SENTIMENT_COLS)
    )

    print()
    print(f"Tweets consolidados (união por tweet_id): {total}")
    print(f"Tweets sem classificação em algum modelo: {missing_any}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    fieldnames = METADATA_KEYS + SENTIMENT_COLS
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in consolidated:
            row = {k: ("" if r.get(k) is None else r[k]) for k in fieldnames}
            w.writerow(row)

    # JSON (null para ausentes)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print(f"CSV:  {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"JSON: {OUTPUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
