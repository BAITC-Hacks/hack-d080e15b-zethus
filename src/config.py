"""Общие настройки хакатонного демо.

Секреты загружаются из .env в точках входа. Здесь только параметры приложения;
модель маршрутизатора фиксирована условиями выбранной реализации.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Модель маршрутизатора и отдельные модели голосового канала.
ROUTER_MODEL = "gpt-4o-mini"
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
IDENTIFIER_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
SPEECH_MODEL = "gpt-4o-mini-tts"
SPEECH_VOICE = "coral"

API_TIMEOUT_SECONDS = 30.0
API_MAX_RETRIES = 2
MAX_UTTERANCE_CHARS = 4000

# Лимиты одного Telegram-процесса; каждое общение хранится в отдельном сеансе.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_VOICE_SECONDS = 60
MAX_CHAT_SESSIONS = 64
SESSION_TTL_SECONDS = 3600
TELEGRAM_POLL_SECONDS = 25
