"""Распознавание и озвучивание сообщений; аудио хранится только в памяти."""

from openai import OpenAI, OpenAIError

from src.config import (
    IDENTIFIER_TRANSCRIPTION_MODEL,
    MAX_AUDIO_BYTES,
    MAX_UTTERANCE_CHARS,
    SPEECH_MODEL,
    SPEECH_VOICE,
    TRANSCRIPTION_MODEL,
)


class AudioError(Exception):
    """Ошибка голосового канала с сообщением, которое можно показать клиенту."""


class AudioService:
    """Переводит голос в текст и ответ бота в аудио без файлов на диске."""

    def __init__(self, client: OpenAI):
        self.client = client

    def transcribe(self, content: bytes, *, expected_slot: str | None = None) -> str:
        if not content or len(content) > MAX_AUDIO_BYTES:
            raise AudioError("Отправьте голосовое сообщение размером до 10 МБ.")
        try:
            identifiers = expected_slot in {"phone", "iin"}
            prompt = (
                "Разговор на русском и казахском языках о страховании. "
                "Қазақша немесе орысша сөйлеуді кириллицамен жазыңыз. Do not translate."
            )
            if identifiers:
                prompt += (
                    " Сейчас клиент может диктовать номер телефона или ИИН, в том числе "
                    "по одной цифре или группами. Сохраните все произнесённые цифры и "
                    "нули в исходном порядке, без добавления кода страны и пропущенных цифр. "
                    "Числа записывайте цифрами. "
                    "Если вместо номера задан другой вопрос, запишите его полностью."
                )
            # Telegram voice — OGG/Opus. Имя с .ogg указывает API формат файла.
            # Язык не фиксируем: в одном сообщении возможны русский и казахский.
            result = self.client.audio.transcriptions.create(
                model=IDENTIFIER_TRANSCRIPTION_MODEL if identifiers else TRANSCRIPTION_MODEL,
                file=("voice.ogg", content, "audio/ogg"),
                response_format="json",
                temperature=0,
                prompt=prompt,
            )
            text = result.text.strip()
            if not text or len(text) > MAX_UTTERANCE_CHARS:
                raise AudioError("Не удалось разобрать короткую реплику. Повторите запись.")
            return text
        except OpenAIError as exc:
            # В SDK-исключении могут быть детали запроса: показываем только тип.
            raise AudioError(
                f"Распознавание недоступно ({type(exc).__name__}). Можно написать текстом."
            ) from None

    def synthesize(self, text: str, language: str) -> bytes:
        try:
            result = self.client.audio.speech.create(
                model=SPEECH_MODEL,
                voice=SPEECH_VOICE,
                input=text,
                response_format="mp3",
                instructions=(
                    "Speak calmly and clearly in Kazakh. Read the supplied text exactly."
                    if language == "kk"
                    else "Говори спокойно и чётко по-русски. Прочитай переданный текст без добавлений."
                ),
            )
            content = result.content
            if not content:
                raise AudioError("Озвучивание вернуло пустую запись. Ответ доступен текстом.")
            return content
        except OpenAIError as exc:
            raise AudioError(
                f"Озвучивание недоступно ({type(exc).__name__}). Ответ доступен текстом."
            ) from None
