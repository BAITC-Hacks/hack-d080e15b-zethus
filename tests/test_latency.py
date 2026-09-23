"""Контракты оптимизаций: параллельное чтение, последовательные действия, честные метрики."""

import json
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

from test_dialog import FakeLLM

from src.agent.catalog import Catalog
from src.agent.dialog_manager import DialogManager
from src.agent.llm import DialogLLM, ModelError, Understanding
from src.main_router import build_system_prompt
from src.tools.audio import AudioError, AudioService


class ParallelTests(unittest.TestCase):
    def test_new_turn_starts_both_read_only_calls_before_either_finishes(self):
        understanding_started, routing_started = Event(), Event()

        class ParallelLLM(FakeLLM):
            def understand(self, text, state):
                understanding_started.set()
                if not routing_started.wait(2):
                    raise AssertionError("Routing was delayed until understanding finished")
                return Understanding("ru", "new", {})

            def route(self, text):
                routing_started.set()
                if not understanding_started.wait(2):
                    raise AssertionError("Understanding was delayed until routing finished")
                return ["SC25"]

        manager = DialogManager(ParallelLLM())
        result = manager.handle("Проверьте мой полис")
        self.assertEqual(result.trace["scenarios"], ["SC25"])
        self.assertEqual(result.trace["expected_slot"], "phone")
        self.assertEqual(manager.backend.events, [])
        self.assertNotIn("total", result.trace["latency_ms"])

    def test_parallel_failure_cannot_run_backend_actions(self):
        llm = FakeLLM()
        llm.route = MagicMock(side_effect=ModelError("offline"))
        manager = DialogManager(llm)
        with self.assertLogs("src.agent.dialog_manager", "WARNING"):
            result = manager.handle("Запишите к врачу")
        self.assertEqual(result.trace["event"], "model_error")
        self.assertEqual(manager.backend.events, [])
        self.assertIsNone(manager.state.pending)

    def test_serial_and_parallel_keep_same_dialog_results(self):
        results = []
        for parallel in (False, True):
            llm = FakeLLM()
            llm.scenarios = ["SC25"]
            llm.understanding = Understanding("ru", "new", {"phone": "+77010000002"})
            manager = DialogManager(llm, parallel=parallel)
            result = manager.handle("Проверьте полис, мой учебный номер +77010000002")
            results.append(
                (
                    result.text,
                    result.trace["scenarios"],
                    result.trace["slots"],
                    manager.backend.events,
                )
            )
        self.assertEqual(*results)
        self.assertEqual(results[0][2]["policy_number"], "SQ-DMS-604220")


class ExplainabilityTests(unittest.TestCase):
    def llm(self, payload):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(refusal=None, content=json.dumps(payload)),
                )
            ]
        )
        return DialogLLM(client, Catalog())

    def test_explanation_and_alternatives_come_from_model_response(self):
        payload = {
            "scenarios": [
                {
                    "scenario_id": "SC21",
                    "confidence": 0.95,
                    "reason": "Клиент просит запись к врачу.",
                }
            ],
            "alternatives": [
                {"scenario_id": "SC22", "confidence": 0.2, "reason": "Вопроса о покрытии нет."}
            ],
        }
        llm = self.llm(payload)
        self.assertEqual(llm.explain_route("Запишите к лору", ["SC21"]), payload)

    def test_invalid_explanations_fail_closed(self):
        for value in (True, -0.1, 1.1, float("nan"), "0.9"):
            with self.subTest(confidence=value):
                llm = self.llm(
                    {
                        "scenarios": [
                            {"scenario_id": "SC21", "confidence": value, "reason": "Запись"}
                        ],
                        "alternatives": [],
                    }
                )
                with self.assertRaises(ModelError):
                    llm.explain_route("Запишите к лору", ["SC21"])

    def test_selected_scenario_cannot_also_be_an_alternative(self):
        row = {"scenario_id": "SC21", "confidence": 0.9, "reason": "Запись"}
        with self.assertRaises(ModelError):
            self.llm({"scenarios": [row], "alternatives": [row]}).explain_route(
                "Запишите", ["SC21"]
            )

    def test_explanation_cannot_change_or_reorder_selected_scenarios(self):
        payload = {
            "scenarios": [{"scenario_id": "SC22", "confidence": 0.9, "reason": "Покрытие"}],
            "alternatives": [],
        }
        with self.assertRaises(ModelError):
            self.llm(payload).explain_route("Запишите к лору", ["SC21"])

    def test_router_keeps_id_only_request_and_does_not_request_explanation(self):
        llm = self.llm({"scenarios": [{"scenario_id": "SC21"}]})
        self.assertEqual(llm.route("Запишите к лору"), ["SC21"])
        llm.client.chat.completions.create.assert_called_once()
        request = llm.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertNotIn('"confidence"', request["messages"][0]["content"])

    def test_original_batch_contract_is_still_default(self):
        prompt = build_system_prompt(Catalog().routing_catalog())
        self.assertIn('Верни строго JSON формата: {"scenarios": [{"scenario_id": "ID"}]}', prompt)
        self.assertNotIn('"alternatives"', prompt)


class AudioLatencyTests(unittest.TestCase):
    def test_repeated_speech_is_cached_but_language_and_text_are_part_of_key(self):
        client = MagicMock()
        client.audio.speech.create.return_value = SimpleNamespace(content=b"audio")
        service = AudioService(client)
        self.assertEqual(service.synthesize("Здравствуйте", "ru"), b"audio")
        self.assertEqual(service.synthesize("Здравствуйте", "ru"), b"audio")
        service.synthesize("Сәлем", "kk")
        self.assertEqual(client.audio.speech.create.call_count, 2)

    def test_stream_yields_first_chunk_without_waiting_for_complete_audio(self):
        consumed = []

        def chunks():
            consumed.append(1)
            yield b"first"
            consumed.append(2)
            yield b"second"

        client = MagicMock()
        response = (
            client.audio.speech.with_streaming_response.create.return_value.__enter__.return_value
        )
        response.iter_bytes.return_value = chunks()
        stream = AudioService(client).stream_speech("Сәлем", "kk")
        self.assertEqual(next(stream), b"first")
        self.assertEqual(consumed, [1])
        self.assertEqual(list(stream), [b"second"])
        client.audio.speech.with_streaming_response.create.return_value.__exit__.assert_called_once()

    def test_empty_speech_stream_is_an_error(self):
        client = MagicMock()
        client.audio.speech.with_streaming_response.create.return_value.__enter__.return_value.iter_bytes.return_value = iter(
            ()
        )
        with self.assertRaises(AudioError):
            list(AudioService(client).stream_speech("Здравствуйте", "ru"))

    def test_browser_recording_preserves_webm_format(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="Здравствуйте")
        AudioService(client).transcribe(b"webm", filename="voice.webm")
        self.assertEqual(
            client.audio.transcriptions.create.call_args.kwargs["file"],
            ("voice.webm", b"webm", "audio/webm"),
        )
