#!/usr/bin/env python3
"""
Deduplica por tweet_id, unifica chaves e ordem dos campos nos JSONs por modelo
em classifiedSentiments/*/ .
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Ordem fixa: todos os objetos terão exatamente estas chaves (JSON null se ausente).
CANONICAL_KEYS = [
    "tweet_id",
    "content",
    "date",
    "event",
    "event_date",
    "days_after_event",
    "model",
    "source",
    "relevante",
    "sentiment",
    "author",
]

OUTPUTS = [
    ("mistral", BASE / "mistral" / "classified_sentiment_mistral.json"),
    ("deepseek", BASE / "deepseek" / "classified_sentiment_deepseek.json"),
    ("llama", BASE / "llama" / "classified_sentiment_llama.json"),
    ("phi", BASE / "phi" / "classified_sentiment_phi4.json"),
]


def _merge_duplicate_rows(rows: list[dict]) -> dict:
    """Une registos com o mesmo tweet_id: preenche None/'' com o primeiro valor não vazio."""
    merged: dict = {}
    for r in rows:
        for k, v in r.items():
            if k not in merged:
                merged[k] = v
                continue
            cur = merged[k]
            if cur in (None, "") and v not in (None, ""):
                merged[k] = v
    return merged


def dedupe_by_tweet_id(rows: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        tid = r["tweet_id"]
        if tid not in buckets:
            order.append(tid)
            buckets[tid] = []
        buckets[tid].append(r)
    return {tid: _merge_duplicate_rows(buckets[tid]) for tid in order}


def normalize_record(r: dict) -> dict:
    return {k: r.get(k) for k in CANONICAL_KEYS}


def main() -> None:
    global_order: list[str] | None = None

    for _name, path in OUTPUTS:
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_id = dedupe_by_tweet_id(rows)
        ids_sorted = sorted(by_id.keys())
        if global_order is None:
            global_order = ids_sorted
        else:
            missing = set(global_order) - set(by_id.keys())
            extra = set(by_id.keys()) - set(global_order)
            if missing or extra:
                raise SystemExit(
                    f"Incompatível {path}: missing={len(missing)} extra={len(extra)}"
                )

        normalized = [normalize_record(by_id[tid]) for tid in ids_sorted]
        path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{path.relative_to(BASE.parent)}: {len(rows)} -> {len(normalized)} registos")


if __name__ == "__main__":
    main()
