"""Голосовое демо в Telegram: python src/telegram_bot.py. Ctrl+C — остановка."""

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.catalog import Catalog, mask_private
from src.agent.dialog_manager import DialogManager
from src.agent.llm import DialogLLM
from src.main_router import PROJECT_ROOT
from src.tools.audio import AudioError, AudioService, MAX_AUDIO_BYTES
from src.tools.telegram_api import TelegramAPI, TelegramError


LOGGER = logging.getLogger(__name__)
WELCOME = (
    "Здравствуйте! Я AI-бот Saqta Insurance. Сәлеметсіз бе! Қалай көмектесе аламын?\n\n"
    "Отправьте текст или голосовое сообщение на русском или казахском (до 60 секунд). "
    "Я отвечу текстом и синтезированным голосом. Используем учебные данные; "
    "реальные полисы и записи не создаются. Сообщения и аудио обрабатываются OpenAI.\n\n"
    "/reset — новый разговор\n/voice_off — только текстовые ответы\n"
    "/voice_on — включить озвучивание\n/trace_on — показывать трассировку\n"
    "/trace_off — скрыть трассировку\n/id — ваш Telegram ID"
)


@dataclass
class ChatSession:
    manager: DialogManager
    voice: bool = True
    trace: bool = False
    last_used: float = field(default_factory=time.monotonic)


class TelegramBot:
    def __init__(self, api, audio, manager_factory, allowed_users: set[int] | None = None):
        self.api, self.audio = api, audio
        self.manager_factory = manager_factory
        self.allowed_users = allowed_users or set()
        self.sessions: dict[int, ChatSession] = {}
        self.last_update_id = -1

    def process_update(self, update: dict):
        update_id = update.get("update_id")
        if not isinstance(update_id, int) or update_id <= self.last_update_id:
            return
        # Повтор доставки не должен повторять «да» и действие внутри разговора.
        self.last_update_id = update_id
        message = update.get("message", {})
        chat = message.get("chat", {})
        user = message.get("from", {})
        if chat.get("type") != "private" or user.get("is_bot"):
            return
        chat_id, user_id = chat.get("id"), user.get("id")
        if not isinstance(chat_id, int) or not isinstance(user_id, int):
            return
        text = message.get("text", "").strip()
        command = text.split(maxsplit=1)[0].split("@")[0] if text else ""
        if command == "/id":
            self.api.send_text(chat_id, f"Ваш Telegram ID: {user_id}")
            return
        if self.allowed_users and user_id not in self.allowed_users:
            self.api.send_text(chat_id, "Доступ к этому демо ограничен. Ваш Telegram ID: " + str(user_id))
            return
        now = time.monotonic()
        self.sessions = {key: s for key, s in self.sessions.items() if now - s.last_used < 3600}
        if chat_id not in self.sessions:
            if len(self.sessions) >= 64:
                self.api.send_text(chat_id, "Демо занято. Попробуйте позже.")
                return
            self.sessions[chat_id] = ChatSession(self.manager_factory())
        session = self.sessions[chat_id]
        session.last_used = now
        if command in {"/start", "/help"}:
            self.api.send_text(chat_id, WELCOME)
            return
        if command == "/reset":
            session.manager = self.manager_factory()
            self.api.send_text(chat_id, "Начат новый разговор. Жаңа әңгіме басталды.")
            return
        if command in {"/voice_on", "/voice_off", "/trace_on", "/trace_off"}:
            if command.startswith("/voice"):
                session.voice = command == "/voice_on"
                self.api.send_text(chat_id, "Озвучивание включено." if session.voice else "Буду отвечать текстом.")
            else:
                session.trace = command == "/trace_on"
                self.api.send_text(chat_id, "Трассировка включена." if session.trace else "Трассировка выключена.")
            return
        if command.startswith("/"):
            self.api.send_text(chat_id, "Список команд: /help")
            return
        started = time.perf_counter()
        stt_ms = 0
        voice = message.get("voice")
        if voice:
            if voice.get("duration", 0) > 60 or voice.get("file_size", 0) > MAX_AUDIO_BYTES:
                session.manager.state.pending = None
                self.api.send_text(chat_id, "Запишите сообщение до 60 секунд и 10 МБ.")
                return
            try:
                content = self.api.download_voice(voice["file_id"])
                text = self.audio.transcribe(content)
            except (AudioError, ValueError) as exc:
                session.manager.state.pending = None
                self.api.send_text(chat_id, str(exc))
                return
            stt_ms = round((time.perf_counter() - started) * 1000)
            self.api.send_text(chat_id, "Распознано / Танылған мәтін: " + mask_private(text))
        elif not text:
            self.api.send_text(chat_id, "Отправьте текст или голосовое сообщение через микрофон Telegram.")
            return
        if len(text) > 4000:
            session.manager.state.pending = None
            self.api.send_text(chat_id, "Напишите короче: до 4000 символов.")
            return
        try:
            self.api.call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except TelegramError:
            pass  # Необязательный индикатор не должен мешать самому ответу.
        dialog_started = time.perf_counter()
        result = session.manager.handle(text)
        dialog_ms = round((time.perf_counter() - dialog_started) * 1000)
        self.api.send_text(chat_id, result.text)
        tts_ms = 0
        if session.voice:
            tts_started = time.perf_counter()
            try:
                speech = self.audio.synthesize(result.text, session.manager.state.language)
                tts_ms = round((time.perf_counter() - tts_started) * 1000)
                self.api.send_voice(chat_id, speech)
            except AudioError as exc:
                self.api.send_text(chat_id, str(exc))
        result.trace["latency_ms"] = {"stt": stt_ms, "dialog": dialog_ms, "tts": tts_ms,
                                      "total_processing": round((time.perf_counter() - started) * 1000)}
        if session.trace:
            self.api.send_trace(chat_id, result.trace)
        LOGGER.info("Обработана реплика: %s мс", result.trace["latency_ms"]["total_processing"])


def save_offset(path: Path, offset: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"offset": offset}), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Сторонние HTTP-логи могут содержать текст запросов; оставляем только ошибки.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not token or not key:
        print("Заполните TELEGRAM_BOT_TOKEN и OPENAI_API_KEY в .env в корне проекта.", file=sys.stderr)
        return 1
    try:
        allowed = {int(value.strip()) for value in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").split(",") if value.strip()}
        if any(user_id <= 0 for user_id in allowed):
            raise ValueError("TELEGRAM_ALLOWED_USER_IDS: нужны положительные числовые ID через запятую.")
        api = TelegramAPI(token)
        identity = api.call("getMe")
        if api.call("getWebhookInfo").get("url"):
            print("У бота уже настроен webhook. Используйте отдельного бота для этого демо.", file=sys.stderr)
            return 1
        path = PROJECT_ROOT / ".runtime" / f"telegram-{identity['id']}.json"
        offset = json.loads(path.read_text(encoding="utf-8"))["offset"] if path.exists() else 0
        if type(offset) is not int or offset < 0:
            raise ValueError("Некорректный offset в .runtime.")
        catalog = Catalog()
        with OpenAI(api_key=key, timeout=30, max_retries=2) as client:
            llm = DialogLLM(client, catalog)
            bot = TelegramBot(api, AudioService(client), lambda: DialogManager(llm, catalog), allowed)
            bot.last_update_id = offset - 1
            print(f"Бот @{identity['username']} запущен. Откройте его в Telegram и нажмите Start. Ctrl+C — остановить.", flush=True)
            while True:
                try:
                    updates = api.call("getUpdates", {"offset": offset, "timeout": 25, "allowed_updates": ["message"]})
                except TelegramError as exc:
                    if exc.code in {401, 403, 404, 409}:
                        print(f"{exc}. Проверьте токен и остановите второй экземпляр бота.", file=sys.stderr)
                        return 1
                    LOGGER.warning("%s; повтор получения сообщений через %s с", exc, exc.retry_after)
                    time.sleep(exc.retry_after)
                    continue
                for update in updates:
                    if update["update_id"] < offset:
                        continue
                    offset = update["update_id"] + 1
                    # Сохраняем ДО обработки: после сбоя не повторяем подтверждённую
                    # операцию. Незавершённую реплику пользователь сможет послать заново.
                    save_offset(path, offset)
                    try:
                        bot.process_update(update)
                    except TelegramError as exc:
                        LOGGER.warning("%s; сообщение не повторяется автоматически", exc)
                        time.sleep(exc.retry_after)
                    except Exception as exc:
                        # Ошибка одного чата не останавливает остальных; без PII/ключей.
                        LOGGER.error("Ошибка обработки сообщения: %s", type(exc).__name__)
                        chat_id = update.get("message", {}).get("chat", {}).get("id")
                        if chat_id in bot.sessions:
                            bot.sessions[chat_id].manager.state.pending = None
                            try:
                                api.send_text(chat_id, "Не удалось обработать сообщение. Повторите его или начните /reset.")
                            except TelegramError:
                                pass
    except (OSError, ValueError, KeyError, TelegramError) as exc:
        print(f"Не удалось запустить Telegram-бота: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nTelegram-бот остановлен.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
