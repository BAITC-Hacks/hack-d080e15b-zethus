"""HTTP-контракты веб-демо на loopback, без внешних API и микрофона."""

import base64
import http.client
import json
import unittest
from threading import Event, Thread
from unittest.mock import MagicMock

from test_dialog import FakeLLM

from src.agent.dialog_manager import DialogManager
from src.agent.llm import ModelError
from src.agent.state import PendingAction
from src.tools.audio import AudioError
from src.web_demo import WebApp, WebServer


class WebTests(unittest.TestCase):
    def setUp(self):
        self.audio = MagicMock()
        self.audio.transcribe.return_value = "Здравствуйте"

        def speech(*args):
            yield b"ID3-audio"

        self.audio.stream_speech.side_effect = speech
        self.app = WebApp(lambda: DialogManager(FakeLLM()), self.audio)
        self.server = WebServer(("127.0.0.1", 0), self.app)
        self.thread = Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True
        )
        self.thread.start()
        self.cookie = ""

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        combined = {"Cookie": self.cookie}
        body = None
        if payload is not None:
            body = json.dumps(payload)
            combined["Content-Type"] = "application/json"
        combined.update(headers or {})
        connection.request(method, path, body=body, headers=combined)
        response = connection.getresponse()
        status, content = response.status, response.read()
        if response.getheader("Set-Cookie"):
            self.cookie = response.getheader("Set-Cookie").split(";")[0]
        kind = response.getheader("Content-Type", "")
        connection.close()
        return status, json.loads(content) if "application/json" in kind else content

    def test_text_voice_and_trace_use_same_session(self):
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/api/session")[0], 200)
        status, result = self.request("POST", "/api/turn", {"text": "Здравствуйте"})
        self.assertEqual(status, 200)
        self.assertEqual(result["trace"]["event"], "greeting")
        self.assertIsNone(result["trace"]["latency_ms"]["total"])
        self.assertEqual(self.request("GET", result["audio_url"]), (200, b"ID3-audio"))
        self.assertEqual(self.request("GET", result["audio_url"]), (200, b"ID3-audio"))
        self.audio.stream_speech.assert_called_once()
        status, trace = self.request("GET", "/api/trace/" + result["turn_id"])
        self.assertEqual(status, 200)
        self.assertIsInstance(trace["latency_ms"]["tts_first_chunk"], int)
        self.assertIsNone(trace["latency_ms"]["total"])

    def test_browser_audio_uses_expected_format(self):
        status, result = self.request(
            "POST",
            "/api/turn",
            {"audio": base64.b64encode(b"webm-audio").decode(), "mime": "audio/webm;codecs=opus"},
        )
        self.assertEqual(status, 200)
        self.audio.transcribe.assert_called_once_with(
            b"webm-audio", expected_slot=None, filename="voice.webm"
        )
        self.assertEqual(result["transcript"], "Здравствуйте")

    def test_http_stream_sends_first_chunk_before_synthesis_finishes(self):
        gate = Event()

        def speech(*args):
            yield b"ID3"
            if not gate.wait(2):
                raise AssertionError("Client could not receive first audio chunk")
            yield b"-tail"

        self.audio.stream_speech.side_effect = speech
        _, result = self.request("POST", "/api/turn", {"text": "Здравствуйте"})
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            conn.request("GET", result["audio_url"], headers={"Cookie": self.cookie})
            response = conn.getresponse()
            self.assertEqual(response.read(3), b"ID3")
            gate.set()
            self.assertEqual(response.read(), b"-tail")
        finally:
            gate.set()
            conn.close()

    def test_sessions_cannot_read_each_others_audio_or_history(self):
        _, first = self.request("POST", "/api/turn", {"text": "Здравствуйте"})
        self.cookie = ""
        self.assertEqual(self.request("GET", first["audio_url"])[0], 404)
        _, second = self.request("POST", "/api/turn", {"text": "Сәлем"})
        self.assertEqual(second["trace"]["turn"], 1)

    def test_cross_origin_requests_are_rejected_before_model(self):
        status, _ = self.request(
            "POST", "/api/turn", {"text": "test"}, {"Origin": "https://unrelated.example"}
        )
        self.assertEqual(status, 403)
        self.assertFalse(self.app.sessions)

    def test_transcription_error_removes_stale_confirmation(self):
        self.request("GET", "/api/session")
        session = next(iter(self.app.sessions.values()))
        session.manager.state.pending = PendingAction("book_appointment", {}, {})
        self.audio.transcribe.side_effect = AudioError("Не слышно речи")
        status, _ = self.request("POST", "/api/turn", {"audio": "eA==", "mime": "audio/ogg"})
        self.assertEqual(status, 502)
        self.assertIsNone(session.manager.state.pending)
        self.assertEqual(session.manager.state.turn, 0)

    def test_invalid_audio_and_oversized_requests_do_not_reach_model(self):
        for payload in (
            {"audio": "!", "mime": "audio/ogg"},
            {"audio": "eA==", "mime": "image/png"},
            {"text": "x" * 4001},
            {"audio": "eA==", "mime": "audio/webm", "duration_ms": 70000},
        ):
            with self.subTest(payload=list(payload)):
                self.assertEqual(self.request("POST", "/api/turn", payload)[0], 400)
        self.audio.transcribe.assert_not_called()

    def test_reset_discards_old_audio_and_state(self):
        _, result = self.request("POST", "/api/turn", {"text": "Здравствуйте"})
        self.assertEqual(self.request("POST", "/api/reset", {})[0], 200)
        self.assertEqual(self.request("GET", result["audio_url"])[0], 404)
        _, result = self.request("POST", "/api/turn", {"text": "Здравствуйте"})
        self.assertEqual(result["trace"]["turn"], 1)

    def test_explanation_does_not_delay_reply_or_change_the_next_turn(self):
        entered, release, completed = Event(), Event(), Event()
        self.request("GET", "/api/session")
        session = next(iter(self.app.sessions.values()))
        session.manager.llm.scenarios = ["SC25"]

        def explain(text, ids):
            entered.set()
            try:
                self.assertTrue(release.wait(2), "Text response waited for explanation")
                self.assertEqual(ids, ["SC25"])
                return {
                    "scenarios": [{"scenario_id": "SC25", "confidence": 0.9, "reason": "Полис"}],
                    "alternatives": [],
                }
            finally:
                completed.set()

        session.manager.llm.explain_route = explain
        try:
            status, response = self.request("POST", "/api/turn", {"text": "Проверьте мой полис"})
            self.assertEqual(status, 200)
            self.assertTrue(entered.wait(1))
            self.assertEqual(response["trace"]["explanation_status"], "pending")
            self.assertEqual(response["trace"]["scenarios"], ["SC25"])
            self.assertEqual(self.request("POST", "/api/reset", {})[0], 200)
        finally:
            release.set()
            self.assertTrue(completed.wait(2))
        self.assertEqual(session.manager.state.turn, 0)
        self.assertFalse(session.records)

    def test_failed_explanation_preserves_answer_and_routing(self):
        self.request("GET", "/api/session")
        session = next(iter(self.app.sessions.values()))
        session.manager.llm.scenarios = ["SC25"]
        session.manager.llm.explain_route = MagicMock(side_effect=ModelError("Invalid schema"))
        with self.assertLogs("src.web_demo", "WARNING"):
            _, result = self.request("POST", "/api/turn", {"text": "Проверьте мой полис"})
            # Семафор освобождается только после обновления статуса и метрик.
            self.assertTrue(self.app.explanation_capacity.acquire(timeout=2))
            self.assertTrue(self.app.explanation_capacity.acquire(timeout=2))
            self.app.explanation_capacity.release()
            self.app.explanation_capacity.release()
        _, trace = self.request("GET", "/api/trace/" + result["turn_id"])
        self.assertEqual(trace["scenarios"], ["SC25"])
        self.assertEqual(trace["explanation_status"], "unavailable")
        self.assertIn("телефон", result["text"])
