"""Проверки маршрутизатора без сети, реального ключа и расходов на API."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src import main_router as router


def completion(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        for name in ("scenarios.json", "dev_utterances.json"):
            shutil.copyfile(router.PROJECT_ROOT / "data" / name, self.root / "data" / name)
        root_patch = patch.object(router, "PROJECT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_catalog_includes_system_intents_and_prompt_excludes_labels(self):
        scenarios, utterances = router.load_data()
        self.assertEqual(len(scenarios), 43)
        self.assertEqual(len(utterances), 104)
        prompt = router.build_system_prompt(scenarios)
        catalog = json.loads(prompt.split("Каталог сценариев:\n", 1)[1])
        self.assertEqual({s["scenario_id"] for s in catalog}, router.VALID_SCENARIO_IDS)
        for scenario in catalog:
            self.assertEqual(set(scenario), {"scenario_id", "name", "description", "not_this_if"})
        self.assertNotIn('"expected"', prompt)
        self.assertNotIn('"examples"', prompt)
        self.assertIn("SC11, SC15, SC38", prompt)

    def test_missing_input_is_reported_with_filename(self):
        (self.root / "data" / "scenarios.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "scenarios.json"):
            router.load_data()

    def test_broken_json_is_reported_with_filename(self):
        (self.root / "data" / "dev_utterances.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dev_utterances.json"):
            router.load_data()

    def test_duplicate_utterance_ids_are_rejected_before_api_calls(self):
        path = self.root / "data" / "dev_utterances.json"
        path.write_text(json.dumps({"utterances": [
            {"id": "U001", "text": "ОГПО"}, {"id": "U001", "text": "КАСКО"},
        ]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Повторяющийся ID реплики"):
            router.load_data()

    def test_missing_system_intents_are_rejected(self):
        path = self.root / "data" / "scenarios.json"
        catalog = json.loads(path.read_text(encoding="utf-8"))
        catalog["system_intents"] = []
        path.write_text(json.dumps(catalog), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "три системных намерения"):
            router.load_data()

    def test_requested_api_contract_and_multi_intent_order(self):
        client = MagicMock()
        client.chat.completions.create.return_value = completion(json.dumps({
            "scenarios": [{"scenario_id": value} for value in ("SC11", "SC27", "SC04", "SC27")],
        }))
        text = "Продлите полис, жүргізушіні қосыңыз, и сейчас попал в аварию"
        result = router.predict_intent(client, "Инструкция JSON", text)
        self.assertEqual(result, ["SC11", "SC27", "SC04"])
        client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini", temperature=0.1, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "Инструкция JSON"},
                      {"role": "user", "content": text}],
        )

    def test_system_intents_are_valid_predictions(self):
        for scenario_id in router.SYSTEM_IDS:
            with self.subTest(scenario_id=scenario_id):
                client = MagicMock()
                client.chat.completions.create.return_value = completion(json.dumps({
                    "scenarios": [{"scenario_id": scenario_id}],
                }))
                self.assertEqual(router.predict_intent(client, "JSON", "Текст"), [scenario_id])

    def test_invalid_responses_fall_back_without_partial_predictions(self):
        invalid_contents = [
            None, "", "{", "null", "[]", "{}", '{"scenarios":[]}',
            '{"scenarios":"SC01"}', '{"scenarios":["SC01"]}',
            '{"scenarios":[{}]}', '{"scenarios":[{"scenario_id":null}]}',
            '{"scenarios":[{"scenario_id":[]}]}',
            '{"scenarios":[{"scenario_id":"SC01"},{"scenario_id":"SC41"}]}',
            '{"scenarios":[{"scenario_id":"SYS_UNKNOWN"}]}',
        ]
        for content in invalid_contents:
            with self.subTest(content=content), self.assertLogs(router.LOGGER, "WARNING"):
                client = MagicMock()
                client.chat.completions.create.return_value = completion(content)
                self.assertEqual(router.predict_intent(client, "JSON", "Текст"), ["SYS_UNCLEAR"])

    def test_incomplete_refused_and_empty_completions_fall_back(self):
        content = '{"scenarios":[{"scenario_id":"SC01"}]}'
        for response in (
            completion(content, "length"), completion(content, "content_filter"),
            completion(content, refusal="refused"), SimpleNamespace(choices=[]),
        ):
            with self.subTest(response=response), self.assertLogs(router.LOGGER, "WARNING"):
                client = MagicMock()
                client.chat.completions.create.return_value = response
                self.assertEqual(router.predict_intent(client, "JSON", "Текст"), ["SYS_UNCLEAR"])

    def test_api_error_falls_back_without_logging_sensitive_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = TimeoutError("sensitive-test-value")
        with self.assertLogs(router.LOGGER, "WARNING") as logs:
            self.assertEqual(router.predict_intent(client, "JSON", "Текст"), ["SYS_UNCLEAR"])
        self.assertNotIn("sensitive-test-value", " ".join(logs.output))

    def test_blank_text_does_not_spend_api_quota(self):
        client = MagicMock()
        self.assertEqual(router.predict_intent(client, "JSON", "  \n"), ["SYS_UNCLEAR"])
        client.chat.completions.create.assert_not_called()

    def test_main_reads_dotenv_processes_all_rows_and_survives_api_errors(self):
        (self.root / ".env").write_text("OPENAI_API_KEY=test-not-a-real-key\n", encoding="utf-8")
        good = completion('{"scenarios":[{"scenario_id":"SC01"}]}')
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            good, completion("invalid json"), TimeoutError("offline test"),
        ] + [good] * 101
        output = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(router, "OpenAI") as factory,
            redirect_stdout(output),
            self.assertLogs(router.LOGGER, "WARNING"),
        ):
            factory.return_value.__enter__.return_value = client
            self.assertEqual(router.main(), 0)
            factory.assert_called_once_with(
                api_key="test-not-a-real-key", timeout=30.0, max_retries=2,
            )
        predictions = json.loads((self.root / "predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(predictions), 104)
        self.assertEqual(predictions["U001"], ["SC01"])
        self.assertEqual(predictions["U002"], ["SYS_UNCLEAR"])
        self.assertEqual(predictions["U003"], ["SYS_UNCLEAR"])
        self.assertEqual(predictions["U104"], ["SC01"])
        self.assertEqual(client.chat.completions.create.call_count, 104)
        self.assertIn("Обработка U104...", output.getvalue())
        _, utterances = router.load_data()
        for call, utterance in zip(client.chat.completions.create.call_args_list, utterances):
            self.assertEqual(call.kwargs["messages"][1], {"role": "user", "content": utterance["text"]})

    def test_missing_key_stops_before_client_creation(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            patch.object(router, "OpenAI") as factory,
            self.assertLogs(router.LOGGER, "ERROR"),
        ):
            self.assertEqual(router.main(), 1)
        factory.assert_not_called()
        self.assertFalse((self.root / "predictions.json").exists())

    def test_failed_save_preserves_previous_predictions_and_cleans_temporary_file(self):
        destination = self.root / "predictions.json"
        previous = '{"U001":["SC01"]}\n'
        destination.write_text(previous, encoding="utf-8")
        with patch.object(router.os, "replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                router._save_predictions({"U001": ["SC02"]})
        self.assertEqual(destination.read_text(encoding="utf-8"), previous)
        self.assertEqual(list(self.root.glob(".predictions-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
