#!/usr/bin/env python3
"""
Avaliação do modelo principal (Mistral) contra final_sentiment (referência).
Read-only sobre o dataset; gera JSON, CSVs e resumo no terminal.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INPUT_JSON = DATA_DIR / "reference_labels_final_complete.json"
OUTPUT_EVAL = DATA_DIR / "mistral_evaluation.json"
OUTPUT_CM_CSV = DATA_DIR / "mistral_confusion_matrix.csv"
OUTPUT_ERRORS_CSV = DATA_DIR / "mistral_classification_errors.csv"

LABELS = ["POSITIVO", "NEGATIVO", "NEUTRO"]
RANDOM_SEED = 42
N_RANDOM_EXAMPLES = 8


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Esperado array JSON")
    return data


def extract_labels(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped: list[dict[str, Any]] = []
    for r in rows:
        t = r.get("final_sentiment")
        p = r.get("mistral_sentiment")
        ts = str(t).strip().upper() if t is not None else ""
        ps = str(p).strip().upper() if p is not None else ""
        if ts not in LABELS or ps not in LABELS:
            skipped.append(
                {
                    "tweet_id": r.get("tweet_id"),
                    "final_sentiment": t,
                    "mistral_sentiment": p,
                }
            )
            continue
        y_true.append(ts)
        y_pred.append(ps)
    return y_true, y_pred, skipped


def confusion_pairs(y_true: list[str], y_pred: list[str]) -> Counter[tuple[str, str]]:
    c: Counter[tuple[str, str]] = Counter()
    for t, p in zip(y_true, y_pred):
        c[(t, p)] += 1
    return c


def top_misclassifications(
    pairs: Counter[tuple[str, str]], top_n: int = 12
) -> list[dict[str, Any]]:
    errors = [(t, p, n) for (t, p), n in pairs.items() if t != p]
    errors.sort(key=lambda x: -x[2])
    out = []
    for t, p, n in errors[:top_n]:
        out.append(
            {
                "true_final_sentiment": t,
                "pred_mistral_sentiment": p,
                "pair_label": f"{p} → {t}",
                "count": n,
            }
        )
    return out


def format_confusion_matrix_text(cm, labels: list[str]) -> str:
    """Linhas = verdadeiro (final), colunas = predito (Mistral)."""
    w = max(8, max(len(x) for x in labels) + 2)
    header = "true\\pred".ljust(w) + "".join(l[:8].ljust(w) for l in labels)
    lines = [header, "-" * len(header)]
    for i, row_label in enumerate(labels):
        line = row_label[:8].ljust(w) + "".join(str(int(cm[i, j])).ljust(w) for j in range(len(labels)))
        lines.append(line)
    return "\n".join(lines)


def write_confusion_csv(cm, labels: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["final_sentiment_true"] + [f"mistral_pred_{l}" for l in labels])
        for i, lab in enumerate(labels):
            w.writerow([lab] + [int(cm[i, j]) for j in range(len(labels))])


def write_errors_csv(rows: list[dict[str, Any]], path: Path) -> int:
    fieldnames = [
        "tweet_id",
        "content",
        "mistral_sentiment",
        "final_sentiment",
        "model",
        "event",
        "days_after_event",
    ]
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            t = str(r.get("final_sentiment", "")).strip().upper()
            p = str(r.get("mistral_sentiment", "")).strip().upper()
            if t not in LABELS or p not in LABELS or t == p:
                continue
            writer.writerow(
                {
                    "tweet_id": r.get("tweet_id", ""),
                    "content": r.get("content", ""),
                    "mistral_sentiment": p,
                    "final_sentiment": t,
                    "model": r.get("model", ""),
                    "event": r.get("event", ""),
                    "days_after_event": r.get("days_after_event", ""),
                }
            )
            n += 1
    return n


def build_evaluation_report(
    rows: list[dict[str, Any]],
    y_true: list[str],
    y_pred: list[str],
    skipped: list[dict[str, Any]],
    cm,
) -> dict[str, Any]:
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    wrong = n - correct
    pct = (100.0 * correct / n) if n else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )

    per_class = {}
    for i, lab in enumerate(LABELS):
        per_class[lab] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1_score": float(f1_per[i]),
            "support": int(sup[i]),
        }

    report_dict = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )

    pairs = confusion_pairs(y_true, y_pred)

    return {
        "input_file": str(INPUT_JSON.relative_to(REPO_ROOT)),
        "n_rows_total": len(rows),
        "n_evaluated": n,
        "n_skipped_invalid_label": len(skipped),
        "skipped_examples": skipped[:20],
        "metrics": {
            "accuracy": acc,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
        },
        "per_class": per_class,
        "sklearn_classification_report": report_dict,
        "confusion_matrix": {
            "description": "Linhas = final_sentiment (verdadeiro); colunas = mistral_sentiment (predito).",
            "labels_true_rows": LABELS,
            "labels_pred_cols": LABELS,
            "matrix": cm.tolist(),
        },
        "agreement": {
            "correct": correct,
            "wrong": wrong,
            "agreement_percentage": round(pct, 4),
        },
        "top_confusions": top_misclassifications(pairs, top_n=15),
        "all_error_pair_counts": {
            f"{p}→{t}": pairs[(t, p)]
            for t in LABELS
            for p in LABELS
            if t != p and pairs[(t, p)] > 0
        },
    }


def print_report(
    report: dict[str, Any],
    cm,
    error_rows_for_samples: list[dict[str, Any]],
) -> None:
    m = report["metrics"]
    ag = report["agreement"]
    print("=" * 72)
    print("AVALIAÇÃO MISTRAL 7B vs final_sentiment (referência)")
    print("=" * 72)
    print(f"Entrada: {report['input_file']}")
    print(f"Registos no ficheiro: {report['n_rows_total']}")
    print(f"Avaliados (rótulos válidos): {report['n_evaluated']}")
    print(f"Ignorados (rótulo inválido): {report['n_skipped_invalid_label']}")
    print()
    print("Métricas globais:")
    print(f"   accuracy:          {m['accuracy']:.4f}")
    print(f"   precision (macro): {m['precision_macro']:.4f}")
    print(f"   recall (macro):    {m['recall_macro']:.4f}")
    print(f"   F1 (macro):        {m['f1_macro']:.4f}")
    print()
    print("Por classe (final_sentiment como referência):")
    for lab in LABELS:
        pc = report["per_class"][lab]
        print(
            f"   {lab:12s}  P={pc['precision']:.4f}  R={pc['recall']:.4f}  "
            f"F1={pc['f1_score']:.4f}  support={pc['support']}"
        )
    print()
    print("Acertos / erros:")
    print(f"   acertos:   {ag['correct']}")
    print(f"   erros:     {ag['wrong']}")
    print(f"   % acordo:  {ag['agreement_percentage']:.2f}%")
    print()
    print("Matriz de confusão [linha=final, coluna=Mistral]:")
    print(format_confusion_matrix_text(cm, LABELS))
    print()
    print("Top confusões (predito → verdadeiro):")
    for item in report["top_confusions"]:
        print(
            f"   {item['pair_label']:28s}  n={item['count']:5d}  "
            f"(Mistral={item['pred_mistral_sentiment']}, ref={item['true_final_sentiment']})"
        )
    print()
    rng = random.Random(RANDOM_SEED)
    pool = [r for r in error_rows_for_samples if r.get("tweet_id")]
    k = min(N_RANDOM_EXAMPLES, len(pool))
    print(f"Exemplos aleatórios de divergências (seed={RANDOM_SEED}, n={k}):")
    if k:
        for r in rng.sample(pool, k=k):
            body = (r.get("content") or "").replace("\n", " ")
            if len(body) > 200:
                body = body[:197] + "..."
            print(
                f"   tweet_id={r.get('tweet_id')}\n"
                f"      Mistral={r.get('mistral_sentiment')}  ref={r.get('final_sentiment')}\n"
                f"      {body}"
            )
    print("=" * 72)


def main() -> None:
    if not INPUT_JSON.is_file():
        print(f"ERRO: {INPUT_JSON} não encontrado", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(INPUT_JSON)
    y_true, y_pred, skipped = extract_labels(rows)

    if not y_true:
        print("ERRO: nenhum par válido para avaliar.", file=sys.stderr)
        sys.exit(1)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = build_evaluation_report(rows, y_true, y_pred, skipped, cm)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_EVAL.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    write_confusion_csv(cm, LABELS, OUTPUT_CM_CSV)
    n_err_rows = write_errors_csv(rows, OUTPUT_ERRORS_CSV)

    error_rows = [
        r
        for r in rows
        if str(r.get("final_sentiment", "")).strip().upper() in LABELS
        and str(r.get("mistral_sentiment", "")).strip().upper() in LABELS
        and str(r.get("final_sentiment", "")).strip().upper()
        != str(r.get("mistral_sentiment", "")).strip().upper()
    ]

    print_report(report, cm, error_rows)
    print()
    print("Ficheiros escritos:")
    print(f"   {OUTPUT_EVAL.relative_to(REPO_ROOT)}")
    print(f"   {OUTPUT_CM_CSV.relative_to(REPO_ROOT)}")
    print(f"   {OUTPUT_ERRORS_CSV.relative_to(REPO_ROOT)}  ({n_err_rows} linhas)")


if __name__ == "__main__":
    main()
