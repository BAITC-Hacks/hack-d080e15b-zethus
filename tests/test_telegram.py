"""Проверка голосового канала без подключения к Telegram и OpenAI."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from test_dialog import FakeLLM

from src import telegram_bot
from src.agent.dialog_manager import DialogManager
from src.agent.state import PendingAction, Task
from src.telegram_bot import TelegramBot, save_offset
from src.tools.audio import AudioError, AudioService
from src.tools.telegram_api import TelegramAPI, TelegramError
from src.tools.terminal_qr import print_bot_link


def update(number, *, user=1, text=None, voice=None, kind="private"):
    message = {"chat": {"id": user, "type": kind}, "from": {"id": user, "is_bot": False}}
    if text is not None:
        message["text"] = text
    if voice is not None:
        message["voice"] = voice
    return {"update_id": number, "message": message}


class TelegramBotTests(unittest.TestCase):
    def setUp(self):
        self.api, self.audio = MagicMock(), MagicMock()
        self.audio.transcribe.return_value = "Здравствуйте!"
        self.audio.synthesize.return_value = b"ID3 voice"
        self.api.download_voice.return_value = b"OggS speech"
        self.bot = TelegramBot(self.api, self.audio, lambda: DialogManager(FakeLLM()))

    def test_voice_runs_complete_pipeline_and_deduplicates_updates(self):
        item = update(1, voice={"file_id": "voice-1", "duration": 3})
        self.bot.process_update(item)
        self.bot.process_update(item)
        self.audio.transcribe.assert_called_once_with(b"OggS speech")
        self.api.send_voice.assert_called_once_with(1, b"ID3 voice")
        self.assertIn("Здравствуйте", self.audio.synthesize.call_args.args[0])
        self.assertEqual(self.bot.sessions[1].manager.state.turn, 1)

    def test_speech_starts_while_text_is_delivered(self):
        synthesis_started, text_sent = Event(), Event()

        def synthesize(*args):
            synthesis_started.set()
            self.assertTrue(text_sent.wait(2), "Text waited for complete synthesis")
            return b"audio"

        def send_text(*args):
            self.assertTrue(synthesis_started.wait(2), "Synthesis waited for text delivery")
            text_sent.set()

        self.audio.synthesize.side_effect = synthesize
        self.api.send_text.side_effect = send_text
        self.bot.process_update(update(1, voice={"file_id": "voice-1", "duration": 3}))
        self.api.send_text.assert_called_once()
        self.assertIn("Распознано", self.api.send_text.call_args.args[1])
        self.api.send_voice.assert_called_once_with(1, b"audio")

    def test_chats_have_separate_state_and_reset(self):
        self.bot.process_update(update(1, text="Здравствуйте"))
        self.bot.process_update(update(2, user=2, text="Сәлеметсіз бе"))
        first = self.bot.sessions[1].manager
        self.assertEqual(first.state.language, "ru")
        self.assertEqual(self.bot.sessions[2].manager.state.language, "kk")
        self.bot.process_update(update(3, text="/reset"))
        self.assertIsNot(self.bot.sessions[1].manager, first)
        self.assertEqual(self.bot.sessions[1].manager.state.turn, 0)
        self.assertEqual(self.bot.sessions[2].manager.state.turn, 1)

    def test_demo_shows_dataset_numbers_without_identifying_user(self):
        self.bot.process_update(update(1, text="/demo"))
        text = self.api.send_text.call_args.args[1]
        self.assertIn("+77010000002", text)
        self.assertIn("ДМС", text)
        self.assertIsNone(self.bot.sessions[1].manager.state.client_id)
        self.assertEqual(self.bot.sessions[1].manager.state.turn, 0)
        self.audio.synthesize.assert_not_called()

    def test_phone_voice_uses_slot_context_and_literal_transcript_digits(self):
        self.bot.process_update(update(1, text="/start"))
        manager = self.bot.sessions[1].manager
        manager.state.active = Task("SC25")
        manager.state.expected_slot = "phone"
        manager.llm.failure = True
        self.audio.transcribe.return_value = "701-000-00-02"
        self.bot.process_update(update(2, voice={"file_id": "x", "duration": 7}))
        self.audio.transcribe.assert_called_once_with(b"OggS speech", expected_slot="phone")
        self.assertEqual(manager.state.client_id, "C002")
        self.assertIn("SQ-DMS-604220", self.audio.synthesize.call_args.args[0])

    def test_only_private_allowed_chats_reach_model(self):
        self.bot.allowed_users = {1}
        self.bot.process_update(update(1, user=2, voice={"file_id": "x"}))
        self.bot.process_update(update(2, text="Здравствуйте", kind="group"))
        self.assertFalse(self.bot.sessions)
        self.audio.transcribe.assert_not_called()
        self.bot.process_update(update(3, user=2, text="/id"))
        self.assertIn("2", self.api.send_text.call_args.args[1])

    def test_voice_off_and_trace_do_not_change_dialog(self):
        self.bot.process_update(update(1, text="/voice_off"))
        self.bot.process_update(update(2, text="/trace_on"))
        self.bot.process_update(update(3, text="Здравствуйте!"))
        self.audio.synthesize.assert_not_called()
        self.api.send_trace.assert_called_once()
        trace = self.api.send_trace.call_args.args[1]
        self.assertEqual(trace["event"], "greeting")
        self.assertEqual(trace["latency_ms"]["stt"], 0)
        self.assertIsNone(trace["latency_ms"]["total"])
        self.assertFalse(trace["parallel_routing"])
        self.assertEqual(self.bot.sessions[1].manager.state.turn, 1)

    def test_long_recording_is_rejected_before_download(self):
        self.bot.process_update(update(1, voice={"file_id": "x", "duration": 61}))
        self.api.download_voice.assert_not_called()
        self.audio.transcribe.assert_not_called()

    def test_transcription_failure_does_not_advance_dialog(self):
        self.bot.process_update(update(1, text="/start"))
        self.bot.sessions[1].manager.state.pending = PendingAction("book_appointment", {}, {})
        self.audio.transcribe.side_effect = AudioError("Не слышно речи")
        self.bot.process_update(update(2, voice={"file_id": "x", "duration": 2}))
        self.assertEqual(self.bot.sessions[1].manager.state.turn, 0)
        self.assertIsNone(self.bot.sessions[1].manager.state.pending)
        self.audio.synthesize.assert_not_called()
        self.assertEqual(self.api.send_text.call_args.args[1], "Не слышно речи")

    def test_synthesis_failure_keeps_text_answer(self):
        self.audio.synthesize.side_effect = AudioError("Озвучивание недоступно")
        self.bot.process_update(update(1, text="Здравствуйте!"))
        messages = [c.args[1] for c in self.api.send_text.call_args_list]
        self.assertTrue(any("Здравствуйте" in text for text in messages))
        self.assertIn("Озвучивание недоступно", messages)
        self.api.send_voice.assert_not_called()

    def test_download_failure_invalidates_previous_confirmation(self):
        self.bot.process_update(update(1, text="/start"))
        manager = self.bot.sessions[1].manager
        manager.state.pending = PendingAction("book_appointment", {}, {})
        self.api.download_voice.side_effect = TelegramError()
        self.bot.process_update(update(2, voice={"file_id": "x", "duration": 2}))
        self.assertIsNone(manager.state.pending)
        self.assertEqual(manager.state.turn, 0)
        self.assertIn("загрузить", self.api.send_text.call_args.args[1])
        self.audio.transcribe.assert_not_called()


class StartupTests(unittest.TestCase):
    def test_startup_encodes_current_bot_from_get_me(self):
        import segno

        rendered_codes = []
        for username in ("JuryFirst_bot", "JurySecond_bot"):
            with self.subTest(username=username), TemporaryDirectory() as directory:
                output = io.StringIO()

                def api_call(method, *args):
                    if method == "getMe":
                        return {"id": 12345, "username": username}
                    if method == "getWebhookInfo":
                        return {}
                    if method == "getUpdates":
                        raise KeyboardInterrupt
                    raise AssertionError(f"Unexpected method: {method}")

                with (
                    patch.object(telegram_bot, "load_dotenv"),
                    patch.object(telegram_bot, "PROJECT_ROOT", Path(directory)),
                    patch.dict(
                        os.environ,
                        {
                            "TELEGRAM_BOT_TOKEN": "12345:TEST_TOKEN",
                            "OPENAI_API_KEY": "test-api-key",
                            "TELEGRAM_ALLOWED_USER_IDS": "",
                        },
                    ),
                    patch.object(telegram_bot, "TelegramAPI") as api,
                    patch.object(telegram_bot, "OpenAI"),
                    patch("segno.make_qr", wraps=segno.make_qr) as make_qr,
                    redirect_stdout(output),
                ):
                    api.return_value.call.side_effect = api_call
                    self.assertEqual(telegram_bot.main(), 0)
                    make_qr.assert_called_once_with(f"https://t.me/{username}", error="m")
                text = output.getvalue()
                self.assertIn(f"https://t.me/{username}", text)
                self.assertIn("█", text)
                self.assertNotIn("TEST_TOKEN", text)
                self.assertNotIn("test-api-key", text)
                rendered_codes.append("\n".join(line for line in text.splitlines() if "█" in line))
        self.assertNotEqual(*rendered_codes)

    def test_link_remains_available_without_qr_dependency(self):
        output = io.StringIO()
        with patch.dict("sys.modules", {"segno": None}):
            print_bot_link("JuryDemo_bot", out=output)
        self.assertIn("https://t.me/JuryDemo_bot", output.getvalue())
        self.assertIn("pip install", output.getvalue())

    def test_terminal_without_block_characters_keeps_readable_link(self):
        class LegacyTerminal(io.StringIO):
            encoding = "cp1251"

        output = LegacyTerminal()
        print_bot_link("JuryDemo_bot", out=output)
        self.assertIn("https://t.me/JuryDemo_bot", output.getvalue())
        self.assertNotIn("█", output.getvalue())


class AudioTests(unittest.TestCase):
    def test_phone_uses_dedicated_context_without_example_numbers(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="701-000-00-02")
        result = AudioService(client).transcribe(b"OggS speech", expected_slot="phone")
        self.assertEqual(result, "701-000-00-02")
        args = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(args["model"], "gpt-4o-transcribe")
        self.assertIn("без добавления", args["prompt"])
        self.assertNotIn("701", args["prompt"])
        self.assertNotIn("language", args)

    def test_transcription_keeps_source_language_and_rejects_empty_text(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="Сәлем!", segments=[]
        )
        service = AudioService(client)
        self.assertEqual(service.transcribe(b"OggS speech"), "Сәлем!")
        args = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(args["model"], "gpt-4o-mini-transcribe")
        self.assertIn("Do not translate", args["prompt"])
        self.assertNotIn("language", args)
        self.assertEqual(args["file"][0], "voice.ogg")
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="   ")
        with self.assertRaises(AudioError):
            service.transcribe(b"OggS silence")

    def test_speech_is_mp3_and_uses_reply_language(self):
        client = MagicMock()
        client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3 audio")
        self.assertEqual(AudioService(client).synthesize("Сәлем!", "kk"), b"ID3 audio")
        args = client.audio.speech.create.call_args.kwargs
        self.assertEqual(args["response_format"], "mp3")
        self.assertIn("Kazakh", args["instructions"])


class TelegramAPITests(unittest.TestCase):
    def test_network_errors_never_expose_token(self):
        token = "123456:SECRET_TOKEN"
        with patch(
            "src.tools.telegram_api.urlopen",
            side_effect=URLError("https://api.telegram.org/bot" + token),
        ):
            with self.assertRaises(TelegramError) as error:
                TelegramAPI(token).call("getMe")
        self.assertNotIn(token, str(error.exception))

    def test_send_voice_uploads_file_with_matching_type(self):
        with patch(
            "src.tools.telegram_api.urlopen", return_value=io.BytesIO(b'{"ok":true,"result":{}}')
        ) as request:
            TelegramAPI("123:FAKE").send_voice(1, b"ID3 audio")
        body = request.call_args.args[0].data
        self.assertIn(b'filename="reply.mp3"', body)
        self.assertIn(b"Content-Type: audio/mpeg", body)
        self.assertIn(b"ID3 audio", body)

    def test_checkpoint_survives_restart_and_replaces_atomically(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state" / "offset.json"
            save_offset(path, 42)
            save_offset(path, 43)
            self.assertEqual(json.loads(path.read_text())["offset"], 43)
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
