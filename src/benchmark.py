"""Сравнить задержку последовательных и параллельных вызовов на настоящем API."""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.catalog import Catalog
from src.agent.dialog_manager import DialogManager
from src.agent.llm import DialogLLM
from src.config import API_MAX_RETRIES, API_TIMEOUT_SECONDS, PROJECT_ROOT

PHRASES = [
    "Хочу записаться к лору завтра в Астане по ДМС",
    "Какие документы нужны для выплаты по КАСКО?",
    "КАСКО можно оплатить в рассрочку?",
    "Алматыдағы кеңсе мекенжайын айтыңызшы",
    "Выплату одобрили, но сумма слишком маленькая, я не согласен",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / ".runtime" / "latency-benchmark.json"
    )
    options = parser.parse_args()
    if not 1 <= options.runs <= 10:
        parser.error("--runs должен быть от 1 до 10")
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("Задайте OPENAI_API_KEY в .env.", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.WARNING)
    catalog = Catalog()
    rows = []
    with OpenAI(api_key=key, timeout=API_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES) as client:
        for run in range(options.runs):
            for index, text in enumerate(PHRASES):
                # Чередование порядка уменьшает преимущество прогретого соединения.
                for parallel in (False, True) if (run + index) % 2 == 0 else (True, False):
                    manager = DialogManager(
                        DialogLLM(client, catalog),
                        catalog,
                        parallel=parallel,
                    )
                    start = time.perf_counter()
                    result = manager.handle(text)
                    row = {
                        "run": run + 1,
                        "sample": index + 1,
                        "parallel": parallel,
                        "dialog_ms": round((time.perf_counter() - start) * 1000),
                        "scenarios": result.trace["scenarios"],
                        "reply": result.text,
                        "language": result.trace["language"],
                        "latency_ms": result.trace["latency_ms"],
                        "ok": bool(result.trace["scenarios"])
                        and result.trace["event"] != "model_error",
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "run",
                                    "sample",
                                    "parallel",
                                    "dialog_ms",
                                    "scenarios",
                                    "ok",
                                )
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    summary = {}
    for parallel, name in ((False, "sequential"), (True, "parallel")):
        measurements = [r["dialog_ms"] for r in rows if r["parallel"] == parallel and r["ok"]]
        summary[name] = {
            "n": len(measurements),
            "median_dialog_ms": statistics.median(measurements) if measurements else None,
        }
    pairs_match = all(
        rows[i]["ok"] and rows[i + 1]["ok"] and rows[i]["scenarios"] == rows[i + 1]["scenarios"]
        for i in range(0, len(rows), 2)
    )
    report = {
        "scope": "Live API dialog only; excludes STT, TTS, audio delivery and client playback",
        "explanations": "Excluded: web explanations run after routing in the background",
        "model": "gpt-4o-mini",
        "summary": summary,
        "scenario_pairs_match": pairs_match,
        "rows": rows,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Совпадение сценариев в парах: {pairs_match}. Отчёт: {options.output}")
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
