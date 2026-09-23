"""Локальное голосовое демо: python src/web_demo.py → http://localhost:8000."""

import argparse
import base64
import binascii
import json
import logging
import os
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock, Thread
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.catalog import Catalog, mask_private
from src.agent.dialog_manager import DialogManager
from src.agent.llm import DialogLLM
from src.config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    MAX_AUDIO_BYTES,
    MAX_UTTERANCE_CHARS,
    PROJECT_ROOT,
    SESSION_TTL_SECONDS,
)
from src.tools.audio import AudioError, AudioService

ASSETS = Path(__file__).with_name("web")
LOGGER = logging.getLogger(__name__)


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class WebSession:
    manager: DialogManager
    lock: Lock = field(default_factory=Lock)
    records: dict = field(default_factory=dict)
    last_used: float = field(default_factory=time.monotonic)


class WebApp:
    """Изолированные сеансы; параллельные вкладки не выполняют одно подтверждение дважды."""

    def __init__(self, manager_factory, audio):
        self.manager_factory = manager_factory
        self.audio = audio
        self.sessions: dict[str, WebSession] = {}
        self.lock = Lock()
        self.capacity = BoundedSemaphore(4)
        self.explanation_capacity = BoundedSemaphore(2)

    def explain(self, manager, text, trace):
        """Панель дополняется независимо от ответа; объяснение не меняет выбранные ID."""
        method = getattr(manager.llm, "explain_route", None)
        if (
            not trace["router_called"]
            or not trace["scenarios"]
            or trace["event"] in {"model_error", "continuation", "resume"}
            or not callable(method)
        ):
            trace["explanation_status"] = "not_requested"
            return
        if not self.explanation_capacity.acquire(blocking=False):
            trace["explanation_status"] = "busy"
            return
        trace["explanation_status"] = "pending"
        selected = list(trace["scenarios"])

        def work():
            started = time.perf_counter()
            try:
                explanation = method(text, selected)
                trace["routing"] = mask_private(explanation)
                trace["alternatives"] = trace["routing"]["alternatives"]
                trace["explanation_status"] = "ready"
            except Exception as exc:
                LOGGER.warning("Объяснение недоступно: %s", type(exc).__name__)
                trace["explanation_status"] = "unavailable"
            finally:
                trace["latency_ms"]["explanation"] = round((time.perf_counter() - started) * 1000)
                self.explanation_capacity.release()

        Thread(target=work, name="route-explanation", daemon=True).start()

    def session(self, cookie: str) -> tuple[str, WebSession]:
        parsed = SimpleCookie()
        try:
            parsed.load(cookie)
        except Exception:
            parsed = SimpleCookie()
        sid = parsed["saqta_session"].value if "saqta_session" in parsed else ""
        with self.lock:
            now = time.monotonic()
            self.sessions = {
                key: value
                for key, value in self.sessions.items()
                if now - value.last_used < SESSION_TTL_SECONDS or value.lock.locked()
            }
            if sid not in self.sessions:
                if len(self.sessions) >= 16:
                    raise RequestError("Все сеансы заняты. Попробуйте позже.", 503)
                sid = uuid4().hex
                self.sessions[sid] = WebSession(self.manager_factory())
            session = self.sessions[sid]
            session.last_used = now
            return sid, session

    def turn(self, session: WebSession, payload: dict) -> dict:
        started = time.perf_counter()
        text = payload.get("text", "")
        stt_ms = 0
        if "audio" in payload:
            duration = payload.get("duration_ms", 0)
            if type(duration) not in (int, float) or not 0 <= duration <= 61000:
                raise RequestError("Запишите сообщение до 60 секунд.")
            if text or not isinstance(payload["audio"], str):
                raise RequestError("Отправьте либо текст, либо аудио.")
            formats = {
                "audio/webm": "voice.webm",
                "audio/ogg": "voice.ogg",
                "audio/mp4": "voice.mp4",
                "audio/wav": "voice.wav",
            }
            mime = str(payload.get("mime", "")).split(";")[0]
            if mime not in formats:
                raise RequestError("Браузер прислал неподдерживаемый формат аудио.")
            try:
                content = base64.b64decode(payload["audio"], validate=True)
            except (ValueError, binascii.Error):
                raise RequestError("Не удалось прочитать аудио.") from None
            if not content or len(content) > MAX_AUDIO_BYTES:
                raise RequestError("Запишите сообщение до 60 секунд и 10 МБ.")
            before = time.perf_counter()
            text = self.audio.transcribe(
                content, expected_slot=session.manager.state.expected_slot, filename=formats[mime]
            )
            stt_ms = round((time.perf_counter() - before) * 1000)
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_UTTERANCE_CHARS:
            raise RequestError("Нужна непустая реплика до 4000 символов.")
        result = session.manager.handle(text)
        result.trace["latency_ms"].update(
            stt=stt_ms,
            tts_first_chunk=None,
            tts=None,
            total=None,
            server_to_text=round((time.perf_counter() - started) * 1000),
        )
        turn_id = uuid4().hex
        response = {
            "turn_id": turn_id,
            "text": result.text,
            "transcript": mask_private(text),
            "trace": result.trace,
            "audio_url": f"/api/audio/{turn_id}",
        }
        session.records[turn_id] = {
            "response": response,
            "language": session.manager.state.language,
            "audio": None,
            "audio_lock": Lock(),
        }
        # Достаточно для десяти реплик жюри; аудио и история остаются только в памяти.
        while len(session.records) > 10:
            session.records.pop(next(iter(session.records)))
        self.explain(session.manager, text, result.trace)
        return response


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app: WebApp):
        super().__init__(address, WebHandler)
        self.app = app


class WebHandler(BaseHTTPRequestHandler):
    """Сервер доступен только на loopback; ключ OpenAI никогда не передаётся браузеру."""

    def log_message(self, *_):
        pass

    def _check_origin(self):
        host = self.headers.get("Host", "")
        if host not in {
            f"localhost:{self.server.server_port}",
            f"127.0.0.1:{self.server.server_port}",
        }:
            raise RequestError("Откройте демо через localhost.", 403)
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            raise RequestError("Запрос с другого сайта отклонён.", 403)

    def _headers(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'",
        )
        if length is not None:
            self.send_header("Content-Length", str(length))
        if getattr(self, "session_id", None):
            self.send_header(
                "Set-Cookie", f"saqta_session={self.session_id}; Path=/; HttpOnly; SameSite=Strict"
            )
        self.end_headers()

    def _json(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(content))
        self.wfile.write(content)

    def _session(self):
        self.session_id, session = self.server.app.session(self.headers.get("Cookie", ""))
        return session

    def _payload(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RequestError("Некорректный размер запроса.") from None
        if not 0 < size <= MAX_AUDIO_BYTES * 4 // 3 + 4096:
            raise RequestError("Сообщение слишком большое или пустое.", 413)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise RequestError("Ожидается JSON.", 415)
        try:
            payload = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeError):
            raise RequestError("Некорректный JSON.") from None
        if not isinstance(payload, dict):
            raise RequestError("Ожидается объект JSON.")
        return payload

    def do_GET(self):
        try:
            self._check_origin()
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if self.path in assets:
                name, kind = assets[self.path]
                content = (ASSETS / name).read_bytes()
                self._headers(200, kind + "; charset=utf-8", len(content))
                self.wfile.write(content)
                return
            if self.path == "/favicon.ico":
                self._headers(204, "image/x-icon", 0)
                return
            if self.path != "/api/session" and not self.path.startswith(
                ("/api/audio/", "/api/trace/")
            ):
                raise RequestError("Страница не найдена.", 404)
            session = self._session()
            if self.path == "/api/session":
                self._json(
                    {
                        "demo": session.manager.catalog.demo_help().replace(
                            "/reset —", "«Новый разговор» —"
                        ),
                        "date": str(session.manager.catalog.today),
                    }
                )
                return
            if self.path.startswith(("/api/audio/", "/api/trace/")):
                record = session.records.get(self.path.rsplit("/", 1)[-1])
                if record is None:
                    raise RequestError("Реплика уже сброшена. Начните новый запрос.", 404)
                if self.path.startswith("/api/trace/"):
                    self._json(record["response"]["trace"])
                else:
                    self._audio(record)
                return
            raise RequestError("Страница не найдена.", 404)
        except RequestError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _audio(self, record):
        if not record["audio_lock"].acquire(blocking=False):
            raise RequestError("Озвучивание уже загружается.", 409)
        if not self.server.app.capacity.acquire(blocking=False):
            record["audio_lock"].release()
            raise RequestError("Демо занято. Повторите воспроизведение.", 503)
        started = time.perf_counter()
        sent_headers = False
        cached = record["audio"] is not None
        try:
            if record["audio"] is not None:
                self._headers(200, "audio/mpeg", len(record["audio"]))
                sent_headers = True
                self.wfile.write(record["audio"])
                return
            chunks = []
            size = 0
            with closing(
                self.server.app.audio.stream_speech(record["response"]["text"], record["language"])
            ) as stream:
                for chunk in stream:
                    if not sent_headers:
                        record["response"]["trace"]["latency_ms"]["tts_first_chunk"] = round(
                            (time.perf_counter() - started) * 1000
                        )
                        self._headers(200, "audio/mpeg")
                        sent_headers = True
                    size += len(chunk)
                    if size > MAX_AUDIO_BYTES:
                        raise AudioError("Голосовой ответ слишком большой. Прочитайте текст.")
                    chunks.append(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
            if not sent_headers:
                raise AudioError("Озвучивание вернуло пустой ответ.")
            record["audio"] = b"".join(chunks)
        except (BrokenPipeError, ConnectionResetError):
            record["response"]["trace"]["audio_error"] = "Воспроизведение прервано."
        except Exception as exc:
            LOGGER.warning("Ошибка озвучивания: %s", type(exc).__name__)
            record["response"]["trace"]["audio_error"] = (
                "Озвучивание недоступно; ответ есть в тексте."
            )
            if not sent_headers:
                self._json({"error": record["response"]["trace"]["audio_error"]}, 502)
        finally:
            if not cached:
                record["response"]["trace"]["latency_ms"]["tts"] = round(
                    (time.perf_counter() - started) * 1000
                )
            self.server.app.capacity.release()
            record["audio_lock"].release()

    def do_POST(self):
        session = None
        locked = False
        capacity = False
        try:
            self._check_origin()
            session = self._session()
            if not session.lock.acquire(blocking=False):
                raise RequestError("Дождитесь ответа на предыдущую реплику.", 409)
            locked = True
            payload = self._payload()
            if self.path == "/api/reset":
                session.manager = self.server.app.manager_factory()
                session.records.clear()
                self._json({"ok": True})
            elif self.path == "/api/turn":
                capacity = self.server.app.capacity.acquire(blocking=False)
                if not capacity:
                    raise RequestError("Демо занято. Попробуйте через несколько секунд.", 503)
                self._json(self.server.app.turn(session, payload))
            else:
                raise RequestError("Страница не найдена.", 404)
        except (RequestError, AudioError) as exc:
            if session is not None and locked:
                session.manager.state.pending = None
            self._json({"error": str(exc)}, getattr(exc, "status", 502))
        except (BrokenPipeError, ConnectionResetError):
            if session is not None and locked:
                session.manager.state.pending = None
        except Exception as exc:
            LOGGER.error("Ошибка веб-демо: %s", type(exc).__name__)
            if session is not None and locked:
                session.manager.state.pending = None
            self._json(
                {"error": "Не удалось обработать запрос. Повторите его или сбросьте разговор."}, 500
            )
        finally:
            if capacity:
                self.server.app.capacity.release()
            if locked:
                session.lock.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Голосовое веб-демо SaqtaDauys")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("Задайте OPENAI_API_KEY в .env.", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.WARNING)
    try:
        catalog = Catalog()
        with OpenAI(
            api_key=key, timeout=API_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES
        ) as client:
            app = WebApp(
                lambda: DialogManager(DialogLLM(client, catalog), catalog),
                AudioService(client),
            )
            with WebServer(("127.0.0.1", options.port), app) as server:
                print(
                    f"SaqtaDauys: http://localhost:{server.server_port}\nCtrl+C — остановить.",
                    flush=True,
                )
                server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as exc:
        print(f"Не удалось запустить веб-демо: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
