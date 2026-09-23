"""Ссылка и QR-код текущего Telegram-бота для показа в терминале."""

import io
import os
import re
import sys
from typing import TextIO


def print_bot_link(username: str, *, out: TextIO | None = None) -> None:
    """Взять username из getMe; отсутствие QR не должно мешать запуску бота."""
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_]+", username):
        raise ValueError("Telegram не вернул корректный username бота.")
    out = out if out is not None else sys.stdout
    url = f"https://t.me/{username}"
    print(f"\nОткрыть бота: {url}", file=out, flush=True)
    try:
        # Необязательная визуализация: старое окружение может ещё не иметь Segno.
        import segno

        qr = segno.make_qr(url, error="m")
        buffer = io.StringIO()
        qr.terminal(out=buffer, compact=True, border=4)
        rendered = buffer.getvalue()
        # Проверяем кодировку до печати, чтобы не оставить половину QR на экране.
        rendered.encode(getattr(out, "encoding", None) or "utf-8")
    except ImportError:
        print("Для QR установите зависимости: python -m pip install -r requirements.txt", file=out)
        return
    except (UnicodeError, ValueError):
        print("QR недоступен в этой кодировке терминала. Откройте ссылку выше.", file=out)
        return

    use_color = out.isatty() and os.getenv("TERM") != "dumb"
    for line in rendered.splitlines():
        # Segno рисует светлые модули блоками: фиксируем контраст на любой теме.
        print(f"\033[97;40m{line}\033[0m" if use_color else line, file=out)
    print("Наведите камеру на QR-код и нажмите Start в Telegram.\n", file=out, flush=True)
