#!/usr/bin/env python3
"""
Auditoria read-only de data/reference_labels_final.json.
Não altera o dataset; gera data/dataset_audit_report.json e resumo no terminal.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INPUT_JSON = DATA_DIR / "reference_labels_final.json"
OUTPUT_REPORT = DATA_DIR / "dataset_audit_report.json"

REQUIRED_KEYS = [
    "tweet_id",
    "content",
    "date",
    "event",
    "event_date",
    "days_after_event",
    "model",
    "mistral_sentiment",
    "llama_sentiment",
    "phi_sentiment",
    "deepseek_sentiment",
    "final_sentiment",
]

SENTIMENT_FIELDS = [
    "mistral_sentiment",
    "llama_sentiment",
    "phi_sentiment",
    "deepseek_sentiment",
    "final_sentiment",
]

VALID_LABELS = frozenset({"POSITIVO", "NEGATIVO", "NEUTRO"})
VALID_AGREEMENT_LEVELS = frozenset({1, 2, 3})
EXPECTED_MODELS = frozenset({"gpt", "gemini", "copilot"})
EXPECTED_EVENTS = frozenset({"launch_initial", "latest_version"})

MAX_EXAMPLES = 12


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _norm_label(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return str(value).strip().upper()


def _parse_days(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _append_capped(bucket: list[Any], item: Any) -> None:
    if len(bucket) < MAX_EXAMPLES:
        bucket.append(item)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Esperado array JSON na raiz")
    return data


def _row_has_blocking_issue(
    row: dict[str, Any],
    tid_counter: Counter[str],
) -> bool:
    tid_s = str(row.get("tweet_id", "")).strip()
    if not tid_s or _is_blank(row.get("content")) or _is_blank(row.get("final_sentiment")):
        return True
    if tid_counter.get(tid_s, 0) > 1:
        return True
    for sf in SENTIMENT_FIELDS:
        lab = _norm_label(row.get(sf))
        if lab is not None and lab not in VALID_LABELS:
            return True
    return False


@dataclass
class AuditAccumulator:
    null_counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(REQUIRED_KEYS, 0)
    )
    null_examples: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    invalid_sentiment: list[dict[str, Any]] = field(default_factory=list)
    invalid_sentiment_count: int = 0
    invalid_agreement: list[dict[str, Any]] = field(default_factory=list)
    invalid_agreement_count: int = 0
    unexpected_model: list[dict[str, str]] = field(default_factory=list)
    unexpected_event: list[dict[str, str]] = field(default_factory=list)
    unexpected_model_count: int = 0
    unexpected_event_count: int = 0
    days_negative: list[dict[str, Any]] = field(default_factory=list)
    days_over_90: list[dict[str, Any]] = field(default_factory=list)
    duplicate_examples: list[dict[str, Any]] = field(default_factory=list)
    seen_first: dict[str, int] = field(default_factory=dict)

    def _tweet_id_str(self, row: dict[str, Any]) -> str:
        tid_raw = row.get("tweet_id")
        return str(tid_raw).strip() if tid_raw is not None else ""

    def on_row(self, idx: int, row: dict[str, Any], duplicate_ids: set[str]) -> None:
        tid_s = self._tweet_id_str(row)
        self._dup_example(idx, tid_s, duplicate_ids)
        self._nulls(row, tid_s)
        self._sentiments(row, tid_s)
        self._agreement(row, tid_s)
        self._model_event(row, tid_s)
        self._days_bounds(row, tid_s)

    def _dup_example(self, idx: int, tid_s: str, duplicate_ids: set[str]) -> None:
        if not tid_s:
            return
        if tid_s in self.seen_first and tid_s in duplicate_ids:
            _append_capped(
                self.duplicate_examples,
                {
                    "tweet_id": tid_s,
                    "detail": f"ocorrência extra (índices {self.seen_first[tid_s]} e {idx})",
                },
            )
        else:
            self.seen_first.setdefault(tid_s, idx)

    def _nulls(self, row: dict[str, Any], tid_s: str) -> None:
        for key in REQUIRED_KEYS:
            val = row.get(key) if key in row else None
            if key not in row or _is_blank(val):
                self.null_counts[key] += 1
                _append_capped(
                    self.null_examples[key],
                    {"tweet_id": tid_s or "(vazio)", "field": key},
                )

    def _sentiments(self, row: dict[str, Any], tid_s: str) -> None:
        for sf in SENTIMENT_FIELDS:
            lab = _norm_label(row.get(sf))
            if lab is None:
                continue
            if lab not in VALID_LABELS:
                self.invalid_sentiment_count += 1
                _append_capped(
                    self.invalid_sentiment,
                    {"tweet_id": tid_s, "field": sf, "value": row.get(sf)},
                )

    def _agreement(self, row: dict[str, Any], tid_s: str) -> None:
        al = row.get("agreement_level")
        if al is not None and al not in VALID_AGREEMENT_LEVELS:
            self.invalid_agreement_count += 1
            _append_capped(
                self.invalid_agreement,
                {"tweet_id": tid_s, "agreement_level": al},
            )

    def _model_event(self, row: dict[str, Any], tid_s: str) -> None:
        m = row.get("model")
        if not _is_blank(m) and str(m).strip() not in EXPECTED_MODELS:
            self.unexpected_model_count += 1
            _append_capped(
                self.unexpected_model,
                {"tweet_id": tid_s, "model": str(m)},
            )
        ev = row.get("event")
        if not _is_blank(ev) and str(ev).strip() not in EXPECTED_EVENTS:
            self.unexpected_event_count += 1
            _append_capped(
                self.unexpected_event,
                {"tweet_id": tid_s, "event": str(ev)},
            )

    def _days_bounds(self, row: dict[str, Any], tid_s: str) -> None:
        days = _parse_days(row.get("days_after_event"))
        if days is None:
            return
        if days < 0:
            _append_capped(
                self.days_negative,
                {"tweet_id": tid_s, "days_after_event": days},
            )
        if days > 90:
            _append_capped(
                self.days_over_90,
                {"tweet_id": tid_s, "days_after_event": days},
            )


def run_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tid_counter: Counter[str] = Counter()
    for row in rows:
        tid_raw = row.get("tweet_id")
        tid_s = str(tid_raw).strip() if tid_raw is not None else ""
        tid_counter[tid_s] += 1

    duplicate_ids = {tid for tid, n in tid_counter.items() if n > 1 and tid}
    redundant_rows = sum(tid_counter[tid] - 1 for tid in duplicate_ids)

    acc = AuditAccumulator()
    for idx, row in enumerate(rows):
        acc.on_row(idx, row, duplicate_ids)

    dist_model = Counter(str(r.get("model")).strip() for r in rows if not _is_blank(r.get("model")))
    dist_event = Counter(str(r.get("event")).strip() for r in rows if not _is_blank(r.get("event")))
    dist_final = Counter(
        _norm_label(r.get("final_sentiment")) or ""
        for r in rows
        if not _is_blank(r.get("final_sentiment"))
    )
    if "" in dist_final:
        del dist_final[""]

    n_rows = len(rows)
    n_dup_distinct = len(duplicate_ids)
    t_neg = sum(
        1
        for r in rows
        if (d := _parse_days(r.get("days_after_event"))) is not None and d < 0
    )
    t_over = sum(
        1
        for r in rows
        if (d := _parse_days(r.get("days_after_event"))) is not None and d > 90
    )

    blocking = bool(
        acc.null_counts.get("tweet_id", 0) > 0
        or acc.null_counts.get("content", 0) > 0
        or acc.null_counts.get("final_sentiment", 0) > 0
        or n_dup_distinct > 0
        or acc.invalid_sentiment_count > 0
    )

    temporal_warnings = (
        acc.null_counts.get("event_date", 0) > 0
        or acc.null_counts.get("days_after_event", 0) > 0
        or t_neg > 0
        or t_over > 0
    )
    other_warnings = (
        acc.invalid_agreement_count > 0
        or acc.unexpected_model_count > 0
        or acc.unexpected_event_count > 0
    )
    has_warnings = temporal_warnings or other_warnings

    problematic_rows = sum(1 for r in rows if _row_has_blocking_issue(r, tid_counter))
    valid_rows = n_rows - problematic_rows

    return {
        "input_file": str(INPUT_JSON.relative_to(REPO_ROOT)),
        "total_tweets": n_rows,
        "null_or_blank_counts": dict(acc.null_counts),
        "null_or_blank_examples": {k: v for k, v in acc.null_examples.items() if v},
        "duplicates": {
            "distinct_tweet_ids_with_duplicates": n_dup_distinct,
            "redundant_row_count": redundant_rows,
            "examples": acc.duplicate_examples,
        },
        "invalid_sentiment": {
            "count": acc.invalid_sentiment_count,
            "examples": acc.invalid_sentiment,
        },
        "invalid_agreement_level": {
            "count": acc.invalid_agreement_count,
            "examples": acc.invalid_agreement,
        },
        "unexpected_model": {
            "count": acc.unexpected_model_count,
            "examples": acc.unexpected_model,
        },
        "unexpected_event": {
            "count": acc.unexpected_event_count,
            "examples": acc.unexpected_event,
        },
        "temporal": {
            "days_after_event_negative_count": t_neg,
            "days_after_event_over_90_count": t_over,
            "event_date_null_or_blank_count": acc.null_counts.get("event_date", 0),
            "days_after_event_null_or_blank_count": acc.null_counts.get("days_after_event", 0),
            "examples_days_negative": acc.days_negative,
            "examples_days_over_90": acc.days_over_90,
            "examples_missing_event_date": acc.null_examples.get("event_date", [])[:MAX_EXAMPLES],
        },
        "distributions": {
            "by_model": dict(dist_model),
            "by_event": dict(dist_event),
            "by_final_sentiment": dict(dist_final),
        },
        "summary": {
            "total_records": n_rows,
            "valid_records_no_blocking_issue": valid_rows,
            "problematic_records_blocking": problematic_rows,
            "duplicate_distinct_ids": n_dup_distinct,
            "redundant_duplicate_rows": redundant_rows,
            "rows_with_any_null_required_field": sum(
                1
                for r in rows
                if any(k not in r or _is_blank(r.get(k)) for k in REQUIRED_KEYS)
            ),
            "invalid_sentiment_row_hits": acc.invalid_sentiment_count,
        },
        "readiness": {
            "ready_for_metrics": not blocking,
            "ready_for_charts": not blocking,
            "ready_for_tcc_results": not blocking,
            "blocking_issues": blocking,
            "has_warnings": has_warnings,
            "notes": [
                "Bloqueante: tweet_id duplicado; tweet_id/content/final_sentiment ausentes; qualquer sentimento fora de POSITIVO|NEGATIVO|NEUTRO.",
                "Aviso: event_date ou days_after_event ausentes; dias fora de [0,90]; agreement_level ∉ {1,2,3}; model ∉ {gpt,gemini,copilot}; event ∉ {launch_initial,latest_version}.",
            ],
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("AUDITORIA FINAL — reference_labels_final.json")
    print("=" * 72)
    print(f"Ficheiro: {report['input_file']}")
    print(f"1) Total de tweets: {report['total_tweets']}")
    print()
    print("2) Null ou ausentes (por campo):")
    any_null = False
    for k in REQUIRED_KEYS:
        n = report["null_or_blank_counts"].get(k, 0)
        if n:
            any_null = True
            print(f"   {k:26s}  {n:5d}")
    if not any_null:
        print("   (nenhum)")
    print()
    print("3) Integridade estrutural:")
    dup = report["duplicates"]
    print(f"   tweet_id com duplicata (ids distintos): {dup['distinct_tweet_ids_with_duplicates']}")
    print(f"   Linhas redundantes (soma count-1):       {dup['redundant_row_count']}")
    invs = report["invalid_sentiment"]
    print(f"   Ocorrências de sentimento inválido:    {invs['count']}")
    inva = report["invalid_agreement_level"]
    print(f"   agreement_level ∉ {{1,2,3}}:           {inva['count']}")
    print()
    print("4) Consistência temporal:")
    t = report["temporal"]
    print(f"   days_after_event < 0 (linhas):         {t['days_after_event_negative_count']}")
    print(f"   days_after_event > 90 (linhas):        {t['days_after_event_over_90_count']}")
    print(f"   event_date null/vazio:                 {t['event_date_null_or_blank_count']}")
    print(f"   days_after_event null/vazio:           {t['days_after_event_null_or_blank_count']}")
    print()
    print("5) Distribuições:")
    print("   por model:", report["distributions"]["by_model"])
    print("   por event:", report["distributions"]["by_event"])
    print("   por final_sentiment:", report["distributions"]["by_final_sentiment"])
    print()
    print("6) Resumo:")
    s = report["summary"]
    print(f"   Total de registos:                    {s['total_records']}")
    print(f"   Registos com algum campo obrigatório vazio: {s['rows_with_any_null_required_field']}")
    print(f"   Registos sem problemas bloqueantes:    {s['valid_records_no_blocking_issue']}")
    print(f"   Registos com problemas bloqueantes:    {s['problematic_records_blocking']}")
    print(f"   Ocorrências inválidas de sentimento:  {s['invalid_sentiment_row_hits']}")
    print()
    print("   Exemplos — duplicados de tweet_id:")
    for ex in dup["examples"][:5]:
        print(f"      {ex}")
    print("   Exemplos — sentimento inválido:")
    for ex in invs["examples"][:5]:
        print(f"      {ex}")
    print("   Exemplos — days_after_event < 0:")
    for ex in t["examples_days_negative"][:5]:
        print(f"      {ex}")
    print("   Exemplos — days_after_event > 90:")
    for ex in t["examples_days_over_90"][:5]:
        print(f"      {ex}")
    print("   Exemplos — event_date ausente (tweet_id):")
    for ex in t.get("examples_missing_event_date", [])[:5]:
        print(f"      {ex}")
    print()
    print("7) Conclusão (pronto para TCC):")
    rd = report["readiness"]
    print(f"   Métricas:          {'SIM' if rd['ready_for_metrics'] else 'NÃO'}  (bloqueante={rd['blocking_issues']})")
    print(f"   Gráficos:         {'SIM' if rd['ready_for_charts'] else 'NÃO'}")
    print(f"   Resultados TCC:  {'SIM' if rd['ready_for_tcc_results'] else 'NÃO'}")
    print(f"   Existem avisos:   {rd['has_warnings']}")
    print("=" * 72)


def main() -> None:
    if not INPUT_JSON.is_file():
        print(f"ERRO: não encontrado {INPUT_JSON}", file=sys.stderr)
        sys.exit(1)

    rows = load_dataset(INPUT_JSON)
    report = run_audit(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_report(report)
    print()
    print(f"Relatório JSON: {OUTPUT_REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
