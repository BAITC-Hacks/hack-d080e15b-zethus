"""Распознавание и озвучивание сообщений; аудио хранится только в памяти."""

from openai import OpenAI, OpenAIError


MAX_AUDIO_BYTES = 10 * 1024 * 1024


class AudioError(Exception):
    pass


class AudioService:
    def __init__(self, client: OpenAI):
        self.client = client

    def transcribe(self, content: bytes) -> str:
        if not content or len(content) > MAX_AUDIO_BYTES:
            raise AudioError("Отправьте голосовое сообщение размером до 10 МБ.")
        try:
            # Telegram voice — OGG/Opus. Имя с .ogg указывает API формат файла.
            # Язык не фиксируем: в одном сообщении возможны русский и казахский.
            result = self.client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=("voice.ogg", content, "audio/ogg"),
                response_format="json", temperature=0,
                prompt=("Разговор на русском и казахском языках о страховании. "
                        "Қазақша немесе орысша сөйлеуді кириллицамен жазыңыз. Do not translate."),
            )
            text = result.text.strip()
            if not text or len(text) > 4000:
                raise AudioError("Не удалось разобрать короткую реплику. Повторите запись.")
            return text
        except OpenAIError as exc:
            # В SDK-исключении могут быть детали запроса: показываем только тип.
            raise AudioError(f"Распознавание недоступно ({type(exc).__name__}). Можно написать текстом.") from None

    def synthesize(self, text: str, language: str) -> bytes:
        try:
            result = self.client.audio.speech.create(
                model="gpt-4o-mini-tts", voice="coral", input=text,
                response_format="mp3",
                instructions=("Speak calmly and clearly in Kazakh. Read the supplied text exactly."
                              if language == "kk" else
                              "Говори спокойно и чётко по-русски. Прочитай переданный текст без добавлений."),
            )
            content = result.content
            if not content:
                raise AudioError("Озвучивание вернуло пустую запись. Ответ доступен текстом.")
            return content
        except OpenAIError as exc:
            raise AudioError(f"Озвучивание недоступно ({type(exc).__name__}). Ответ доступен текстом.") from None
