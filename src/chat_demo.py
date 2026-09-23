"""Текстовая демонстрация: python src/chat_demo.py [--trace]."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Поддерживаются и python src/chat_demo.py, и python -m src.chat_demo.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from src.agent.catalog import Catalog, mask_private
from src.agent.dialog_manager import DialogManager
from src.agent.llm import DialogLLM
from src.main_router import PROJECT_ROOT


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Диалог с учебным ботом Saqta Insurance")
    parser.add_argument("--trace", action="store_true", help="Показывать сценарии, слоты и действия после ответа")
    parser.add_argument("--script", type=Path, help="Прогнать JSON-файл со списком реплик через настоящий API")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("Задайте OPENAI_API_KEY в .env в корне проекта.", file=sys.stderr)
            return 1
        catalog = Catalog()
        turns = None
        if options.script:
            with options.script.open(encoding="utf-8") as file:
                turns = json.load(file)
            if not isinstance(turns, list) or any(not isinstance(text, str) for text in turns):
                raise ValueError("Файл --script должен содержать JSON-список строк.")
        with OpenAI(api_key=api_key, timeout=30.0, max_retries=2) as client:
            llm = DialogLLM(client, catalog)
            manager = DialogManager(llm, catalog)
            print("Saqta Insurance — текстовая демонстрация.")
            print(f"Учебная дата: {catalog.today}. Операции выполняются на учебных данных в памяти сеанса.")
            print("Команды: /reset — новый сеанс, /state — состояние, /quit — выход.")
            print("Бот: Здравствуйте! Чем помочь? / Сәлеметсіз бе! Қалай көмектесе аламын?", flush=True)
            iterator = iter(turns) if turns is not None else None
            while True:
                if iterator is not None:
                    text = next(iterator, None)
                    if text is None:
                        break
                    print("Вы:", mask_private(text), flush=True)
                else:
                    try:
                        text = input("Вы: ").strip()
                    except EOFError:
                        break
                if text in {"/quit", "/exit"}:
                    break
                if text == "/reset":
                    manager = DialogManager(llm, catalog)
                    print("Бот: Начат новый сеанс, учебные данные восстановлены.", flush=True)
                    continue
                if text == "/state":
                    state = manager.state
                    print(json.dumps(mask_private({
                        "language": state.language, "client_id": state.client_id,
                        "active_scenario": state.active.scenario_id if state.active else None,
                        "slots": state.active.slots if state.active else {},
                        "awaiting_confirmation": state.pending is not None,
                        "completed_actions": [event["action"] for event in manager.backend.events],
                    }), ensure_ascii=False, indent=2), flush=True)
                    continue
                if text.startswith("/"):
                    print("Команды: /reset, /state, /quit.", flush=True)
                    continue
                result = manager.handle(text)
                print("Бот:", result.text, flush=True)
                if options.trace:
                    print(json.dumps(result.trace, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, OpenAIError) as exc:
        print(f"Не удалось запустить чат: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nСеанс завершён.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
