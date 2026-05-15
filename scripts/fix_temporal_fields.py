#!/usr/bin/env python3
"""
Preenche event_date e days_after_event em reference_labels_final.json
com base na tabela de eventos do TCC. Não altera sentimentos.
"""
from __future__ import annotations

import json
import statistics
import sys
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INPUT_JSON = DATA_DIR / "reference_labels_final.json"
OUTPUT_JSON = DATA_DIR / "reference_labels_final_complete.json"

# (model, event) -> event_date ISO
EVENT_DATE_TABLE: dict[tuple[str, str], str] = {
    ("gpt", "launch_initial"): "2022-11-30",
    ("gpt", "latest_version"): "2024-05-13",
    ("gemini", "launch_initial"): "2023-02-06",
    ("gemini", "latest_version"): "2024-02-08",
    ("copilot", "launch_initial"): "2021-06-29",
    ("copilot", "latest_version"): "2023-11-01",
}

N_EXAMPLE_ROWS = 8


def parse_iso_date(value: Any) -> date | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    s = str(value).strip()[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def lookup_event_date(model: str, event: str) -> str | None:
    m = model.strip().lower()
    e = event.strip().lower()
    return EVENT_DATE_TABLE.get((m, e))


def compute_days_after_event(tweet_date: date, event_date: date) -> int:
    return (tweet_date - event_date).days


def fix_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Devolve (linha_atualizada, metadados_alteração).
    metadados: {changed: bool, reason: str|None, had_null_event_date: bool, had_null_days: bool}
    """
    out = deepcopy(row)
    meta = {
        "changed": False,
        "reason": None,
        "warning": None,
        "had_null_event_date": out.get("event_date") in (None, ""),
        "had_null_days": out.get("days_after_event") in (None, ""),
    }

    model = str(out.get("model") or "").strip()
    event = str(out.get("event") or "").strip()
    if not model or not event:
        meta["reason"] = "model ou event ausente"
        return out, meta

    ed_str = lookup_event_date(model, event)
    if ed_str is None:
        meta["reason"] = f"combinação desconhecida: model={model!r} event={event!r}"
        return out, meta

    # Só preenche event_date se estiver null/vazio; caso contrário mantém (normalizado)
    if out.get("event_date") in (None, ""):
        out["event_date"] = ed_str
        meta["changed"] = True
    else:
        existing = parse_iso_date(out.get("event_date"))
        table_d = parse_iso_date(ed_str)
        if existing and table_d and existing != table_d:
            meta["warning"] = (
                f"event_date existente difere da tabela (mantido): "
                f"{out.get('event_date')!r} vs tabela {ed_str!r}"
            )

    event_d = parse_iso_date(out.get("event_date"))
    tweet_d = parse_iso_date(out.get("date"))
    if event_d is None:
        meta["reason"] = "event_date inválido após lookup"
        return out, meta
    if tweet_d is None:
        meta["reason"] = "date do tweet inválida"
        return out, meta

    days = compute_days_after_event(tweet_d, event_d)
    old_days = _safe_int(out.get("days_after_event"))
    if old_days != days:
        meta["changed"] = True
    out["days_after_event"] = days

    return out, meta


def _append_issue(
    issues: dict[str, list[dict[str, Any]]],
    key: str,
    item: dict[str, Any],
) -> None:
    if len(issues[key]) < N_EXAMPLE_ROWS:
        issues[key].append(item)


def _days_int(row: dict[str, Any]) -> int | None:
    return _safe_int(row.get("days_after_event"))


def validate_all(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: dict[str, list[dict[str, Any]]] = {
        "null_event_date": [],
        "null_days": [],
        "negative_days": [],
        "over_90_days": [],
    }
    cnt_null_ed = 0
    cnt_null_days = 0
    cnt_neg = 0
    cnt_over = 0

    for row in rows:
        tid = str(row.get("tweet_id", ""))
        if row.get("event_date") in (None, ""):
            cnt_null_ed += 1
            _append_issue(issues, "null_event_date", {"tweet_id": tid})

        d = _days_int(row)
        if d is None:
            cnt_null_days += 1
            _append_issue(
                issues,
                "null_days",
                {"tweet_id": tid, "value": row.get("days_after_event")},
            )
            continue

        if d < 0:
            cnt_neg += 1
            _append_issue(issues, "negative_days", {"tweet_id": tid, "days_after_event": d})
        if d > 90:
            cnt_over += 1
            _append_issue(issues, "over_90_days", {"tweet_id": tid, "days_after_event": d})

    passed = cnt_null_ed == 0 and cnt_null_days == 0 and cnt_neg == 0 and cnt_over == 0
    return {
        "passed": passed,
        "null_event_date_count": cnt_null_ed,
        "null_days_after_event_count": cnt_null_days,
        "negative_days_count": cnt_neg,
        "over_90_days_count": cnt_over,
        "example_issues": issues,
    }


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def stats_days(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    vals = []
    for r in rows:
        d = _safe_int(r.get("days_after_event"))
        if d is not None:
            vals.append(d)
    if not vals:
        return {"min": 0, "max": 0, "mean": 0.0, "n": 0}
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": round(statistics.mean(vals), 4),
        "n": len(vals),
    }


def print_report(
    original: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    n_had_null_temporal: int,
    n_changed: int,
    table_mismatch_warnings: int,
    n_errors: int,
    validation: dict[str, Any],
    dist_event: dict[str, int],
    day_stats: dict[str, float | int],
) -> None:
    orig_by_id = {str(r.get("tweet_id")): r for r in original}

    print("=" * 72)
    print("fix_temporal_fields — reference_labels_final → _complete")
    print("=" * 72)
    print(f"Entrada:  {INPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Saída:    {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Total tweets: {len(fixed_rows)}")
    print(f"Tweets com event_date ou days_after_event ausentes antes: {n_had_null_temporal}")
    print(f"Linhas em que event_date/days_after_event foram atualizados: {n_changed}")
    print(f"Avisos event_date ≠ tabela (mantido valor existente): {table_mismatch_warnings}")
    print(f"Erros (model/event inválidos ou datas ilegíveis): {n_errors}")
    print()
    print("Distribuição por event (após correção):")
    for k in sorted(dist_event.keys()):
        print(f"   {k:20s}  {dist_event[k]:5d}")
    print()
    print("days_after_event (estatísticas):")
    print(f"   n válidos: {day_stats['n']}")
    print(f"   mínimo:    {day_stats['min']}")
    print(f"   máximo:    {day_stats['max']}")
    print(f"   média:     {day_stats['mean']}")
    print()
    print("Validação:")
    print(f"   event_date null:           {validation['null_event_date_count']}")
    print(f"   days_after_event null:     {validation['null_days_after_event_count']}")
    print(f"   days_after_event < 0:      {validation['negative_days_count']}")
    print(f"   days_after_event > 90:     {validation['over_90_days_count']}")
    print(f"   Validação passou:          {validation['passed']}")
    print()
    print("Exemplos corrigidos (tinham event_date ou days null no original):")
    shown = 0
    for fr in fixed_rows:
        if shown >= 6:
            break
        tid = str(fr.get("tweet_id"))
        o = orig_by_id.get(tid)
        if not o:
            continue
        if o.get("event_date") in (None, "") or o.get("days_after_event") in (None, ""):
            print(
                f"   tweet_id={tid}\n"
                f"      model={fr.get('model')}  event={fr.get('event')}\n"
                f"      date={fr.get('date')}\n"
                f"      event_date: {o.get('event_date')!r} → {fr.get('event_date')!r}\n"
                f"      days_after_event: {o.get('days_after_event')!r} → {fr.get('days_after_event')!r}"
            )
            shown += 1

    if validation["over_90_days_count"] or validation["negative_days_count"]:
        print()
        print("Exemplos fora do intervalo [0, 90]:")
        for ex in validation["example_issues"].get("negative_days", [])[:5]:
            print(f"   {ex}")
        for ex in validation["example_issues"].get("over_90_days", [])[:5]:
            print(f"   {ex}")

    print("=" * 72)


def _main_compute() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with INPUT_JSON.open(encoding="utf-8") as f:
        original = json.load(f)

    fixed_rows: list[dict[str, Any]] = []
    n_changed = 0
    n_errors = 0
    table_mismatch_warnings = 0

    for row in original:
        new_row, meta = fix_row(row)
        fixed_rows.append(new_row)
        if meta.get("changed"):
            n_changed += 1
        if meta.get("reason"):
            n_errors += 1
        if meta.get("warning"):
            table_mismatch_warnings += 1

    validation = validate_all(fixed_rows)
    dist_event: dict[str, int] = {}
    for r in fixed_rows:
        ev = str(r.get("event") or "").strip()
        dist_event[ev] = dist_event.get(ev, 0) + 1

    n_had_null_temporal = sum(
        1
        for o in original
        if o.get("event_date") in (None, "") or o.get("days_after_event") in (None, "")
    )

    day_stats = stats_days(fixed_rows)

    extras = {
        "n_had_null_temporal": n_had_null_temporal,
        "n_changed": n_changed,
        "table_mismatch_warnings": table_mismatch_warnings,
        "n_errors": n_errors,
        "validation": validation,
        "dist_event": dist_event,
        "day_stats": day_stats,
    }
    return original, fixed_rows, extras


def main() -> None:
    if not INPUT_JSON.is_file():
        print(f"ERRO: {INPUT_JSON} não encontrado", file=sys.stderr)
        sys.exit(1)

    original, fixed_rows, x = _main_compute()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(fixed_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_report(
        original,
        fixed_rows,
        x["n_had_null_temporal"],
        x["n_changed"],
        x["table_mismatch_warnings"],
        x["n_errors"],
        x["validation"],
        x["dist_event"],
        x["day_stats"],
    )


if __name__ == "__main__":
    main()
