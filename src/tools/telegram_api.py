"""Минимальный клиент Telegram Bot API без дополнительного SDK."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from src.tools.audio import MAX_AUDIO_BYTES


class TelegramError(Exception):
    def __init__(self, code: int = 0, retry_after: int = 3):
        # URL Telegram содержит токен, поэтому никогда не печатаем исходное исключение.
        super().__init__(f"Telegram API: ошибка {code or 'соединения'}")
        self.code = code
        self.retry_after = max(1, min(retry_after, 60))


class TelegramAPI:
    def __init__(self, token: str):
        if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", token):
            raise ValueError("Проверьте TELEGRAM_BOT_TOKEN в .env: нужен токен от BotFather.")
        self._base = f"https://api.telegram.org/bot{token}/"
        self._files = f"https://api.telegram.org/file/bot{token}/"

    def call(self, method: str, payload: dict | None = None, *, upload=None, timeout=40):
        if upload is None:
            body = json.dumps(payload or {}).encode("utf-8")
            content_type = "application/json"
        else:
            field, filename, content, mime = upload
            boundary = "saqta-" + uuid4().hex
            parts = []
            for key, value in (payload or {}).items():
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
            parts.extend([
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode(),
                content, f"\r\n--{boundary}--\r\n".encode(),
            ])
            body = b"".join(parts)
            content_type = f"multipart/form-data; boundary={boundary}"
        request = Request(self._base + method, data=body, headers={"Content-Type": content_type})
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            retry_after = 3
            try:
                detail = json.loads(exc.read(8192))
                retry_after = int(detail.get("parameters", {}).get("retry_after", 3))
            except (ValueError, TypeError, AttributeError):
                pass
            raise TelegramError(exc.code, retry_after) from None
        except (URLError, OSError, ValueError):
            raise TelegramError() from None
        if not isinstance(result, dict) or not result.get("ok"):
            raise TelegramError(result.get("error_code", 0) if isinstance(result, dict) else 0)
        return result["result"]

    def send_text(self, chat_id: int, text: str):
        # Без parse_mode: реплики клиента не могут внедрить Markdown или HTML.
        for start in range(0, len(text), 3500):
            self.call("sendMessage", {"chat_id": chat_id, "text": text[start:start + 3500]})

    def send_voice(self, chat_id: int, content: bytes):
        self.call("sendVoice", {"chat_id": chat_id},
                  upload=("voice", "reply.mp3", content, "audio/mpeg"))

    def send_trace(self, chat_id: int, trace: dict):
        self.call("sendDocument", {"chat_id": chat_id, "caption": "Трассировка последней реплики"},
                  upload=("document", "trace.json", json.dumps(trace, ensure_ascii=False, indent=2).encode(), "application/json"))

    def download_voice(self, file_id: str) -> bytes:
        info = self.call("getFile", {"file_id": file_id})
        path = info.get("file_path", "")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or ".." in path or path.startswith("/"):
            raise TelegramError()
        if info.get("file_size", 0) > MAX_AUDIO_BYTES:
            raise ValueError("Голосовое сообщение больше 10 МБ.")
        try:
            with urlopen(self._files + path, timeout=30) as response:
                content = response.read(MAX_AUDIO_BYTES + 1)
        except (URLError, OSError):
            raise TelegramError() from None
        if not content or len(content) > MAX_AUDIO_BYTES:
            raise ValueError("Отправьте непустое голосовое сообщение до 10 МБ.")
        if not content.startswith(b"OggS"):
            raise ValueError("Отправьте голосовое сообщение, записанное микрофоном Telegram.")
        return content
