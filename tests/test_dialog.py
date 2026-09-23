"""Диалог и учебные действия проверяются без сетевых вызовов."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from src.agent.catalog import Catalog, mask_private
from src.agent.dialog_manager import DialogManager
from src.agent.llm import DialogLLM, ModelError, Understanding
from src.agent.state import Task
from src.tools.backend import BackendError, MockBackend


class FakeLLM:
    def __init__(self):
        self.understanding = Understanding("ru", "new", {})
        self.scenarios = ["SYS_UNCLEAR"]
        self.facts = []
        self.failure = False

    def understand(self, text, state):
        if self.failure:
            raise ModelError("APIConnectionError")
        return deepcopy(self.understanding)

    def route(self, text):
        return self.scenarios[:]

    def respond(self, language, purpose, facts):
        self.facts.append(deepcopy(facts))
        return "Ответ по данным компании." if language == "ru" else "Компания деректері бойынша жауап."


class DialogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()
        self.llm = FakeLLM()
        self.manager = DialogManager(self.llm, self.catalog)

    def turn(self, *, ids=None, slots=None, mode="new", language="ru", text="Тестовая реплика"):
        self.llm.understanding = Understanding(language, mode, slots or {})
        self.llm.scenarios = ids or ["SYS_UNCLEAR"]
        return self.manager.handle(text)

    def booking(self, language="ru"):
        return self.turn(ids=["SC21"], slots={"phone": "+77010000002", "doctor_specialty": "лор",
                         "preferred_date": "завтра"}, language=language)

    def test_greeting_does_not_route_to_goodbye_or_call_model(self):
        self.llm.failure = True
        for text, language in (("Здравствуйте!", "ru"), ("«Сәлеметсіз бе!»", "kk")):
            result = self.manager.handle(text)
            self.assertEqual(result.trace["event"], "greeting")
            self.assertEqual(result.trace["language"], language)
            self.assertEqual(result.trace["scenarios"], [])
            self.assertEqual(result.trace["actions"], [])

    def test_greeting_preserves_active_task_and_confirmation(self):
        self.booking()
        result = self.manager.handle("Здравствуйте")
        self.assertIn("Подтверждаете", result.text)
        self.assertTrue(result.trace["awaiting_confirmation"])
        self.assertEqual(self.manager.backend.appointments, [])

    def test_greeting_with_request_is_routed(self):
        result = self.turn(ids=["SC25"], text="Здравствуйте, проверьте полис")
        self.assertEqual(result.trace["active_scenario"], "SC25")

    def test_new_request_is_not_swallowed_as_old_slot_answer(self):
        self.turn(ids=["SC11"])
        result = self.turn(mode="continue", ids=["SC04"], text=(
            "Здравствуйте! Я хотел узнать статус заявления, хотя нет, "
            "сначала скажите, как добавить жену в действующий полис ОГПО?"))
        self.assertEqual(result.trace["scenarios"], ["SC04"])
        self.assertEqual(result.trace["suspended"], ["SC11"])
        self.assertFalse(any(action.get("result", {}).get("queue") == "claims_team"
                             for action in result.trace["actions"]))

    def test_russian_complaint_interrupts_kazakh_phone_question(self):
        self.turn(ids=["SC22"], slots={"service_name": "лекарства"}, language="kk")
        result = self.turn(mode="continue", ids=["SC19"], language="ru", text=(
            "Мне пришла выплата за бампер, но на запчасти не хватит. Я не согласен с суммой."))
        self.assertEqual(result.trace["active_scenario"], "SC19")
        self.assertEqual(result.trace["language"], "ru")
        self.assertEqual(result.trace["suspended"], ["SC22"])

    def test_policy_flow_uses_snapshot_and_reports_expired_policy(self):
        result = self.turn(ids=["SC25"])
        self.assertEqual(result.trace["expected_slot"], "phone")
        result = self.turn(mode="continue", slots={"phone": "8 (701) 000-00-03"})
        self.assertIn("истёк", result.text)
        self.assertIn("2026-09-29", result.text)
        self.assertEqual(self.manager.state.client_id, "C003")
        self.assertIsNone(self.manager.state.active)

    def test_multiple_policies_require_selection(self):
        result = self.turn(ids=["SC25"], slots={"phone": "+77010000001"})
        self.assertEqual(result.trace["expected_slot"], "policy_number")
        self.assertIn("SQ-OGPO-104501", result.text)
        self.assertIn("SQ-CASCO-204118", result.text)
        result = self.turn(mode="continue", slots={"product_type": "ogpo"})
        self.assertIn("SQ-OGPO-104501", result.text)
        self.assertIn("действует", result.text)

    def test_policy_number_alone_does_not_bypass_identification(self):
        result = self.turn(ids=["SC25"], slots={"policy_number": "SQ-DMS-604220"})
        self.assertEqual(result.trace["expected_slot"], "phone")
        self.assertEqual(self.manager.backend.events, [])

    def test_foreign_policy_is_not_disclosed(self):
        result = self.turn(ids=["SC25"], slots={"phone": "+77010000003", "policy_number": "SQ-DMS-604220"})
        self.assertNotIn("2026-12-31", result.text)
        self.assertEqual(result.trace["expected_slot"], "policy_number")

    def test_unknown_phone_reasks_then_offers_iin_then_handoff(self):
        first = self.turn(ids=["SC25"], slots={"phone": "+77010000099"})
        self.assertEqual(first.trace["expected_slot"], "phone")
        second = self.turn(mode="continue", slots={"phone": "+77010000099"})
        self.assertEqual(second.trace["expected_slot"], "iin")
        third = self.turn(mode="continue", slots={"iin": "123456789012"})
        self.assertIn("оператор", third.text)
        self.assertIsNone(self.manager.state.client_id)

    def test_booking_is_previewed_and_committed_exactly_once(self):
        preview = self.booking()
        self.assertTrue(preview.trace["awaiting_confirmation"])
        self.assertIn("2026-10-02", preview.text)
        self.assertIn("Saulet Medical", preview.text)
        self.assertEqual(self.manager.backend.appointments, [])
        done = self.manager.handle("Да, подтверждаю!")
        self.assertIn("создана", done.text)
        self.assertEqual(len(self.manager.backend.appointments), 1)
        self.turn(text="да")
        self.assertEqual(len(self.manager.backend.appointments), 1)

    def test_correction_requires_confirmation_of_new_values(self):
        self.booking()
        corrected = self.turn(mode="continue", text="Да, но лучше в понедельник",
                              slots={"preferred_date": "2026-10-05"})
        self.assertEqual(self.manager.backend.appointments, [])
        self.assertIn("2026-10-05", corrected.text)
        self.manager.handle("да")
        self.assertTrue(self.manager.backend.appointments[0]["slot_datetime"].startswith("2026-10-05"))

    def test_rejection_cancels_without_writing(self):
        self.booking()
        self.manager.handle("нет")
        self.assertEqual(self.manager.backend.appointments, [])
        self.assertIsNone(self.manager.state.pending)

    def test_topic_switch_preserves_task_but_requires_new_confirmation(self):
        self.booking()
        result = self.turn(ids=["SC33"], slots={"city": "Алматы"})
        self.assertIn("оставшемуся", result.text)
        self.assertIsNone(self.manager.state.pending)
        self.assertEqual(self.manager.state.suspended[0].slots["city"], "Astana")
        resumed = self.manager.handle("да")
        self.assertIn("Saulet Medical", resumed.text)
        self.assertEqual(self.manager.backend.appointments, [])
        self.manager.handle("да")
        self.assertEqual(len(self.manager.backend.appointments), 1)

    def test_resume_while_another_task_active_restores_previous_task(self):
        self.manager.state.suspended = [Task("SC25")]
        self.manager.state.active = Task("SC21")
        self.turn(mode="resume")
        self.assertEqual(self.manager.state.active.scenario_id, "SC25")
        self.assertEqual(self.manager.state.suspended[-1].scenario_id, "SC21")

    def test_multi_intent_queue_keeps_identity_and_confirms_contact_change(self):
        original = self.manager.backend.client("C002")["email"]
        result = self.turn(ids=["SC25", "SC29"], slots={"phone": "+77010000002",
                           "contact_field": "email", "new_value": "new@mail.example"})
        self.assertIn("SQ-DMS-604220", result.text)
        self.assertEqual(result.trace["queued"], ["SC29"])
        preview = self.manager.handle("да")
        self.assertTrue(preview.trace["awaiting_confirmation"])
        self.assertEqual(self.manager.backend.client("C002")["email"], original)
        self.manager.handle("да")
        self.assertEqual(self.manager.backend.client("C002")["email"], "new@mail.example")

    def test_language_switch_and_kazakh_confirmation(self):
        result = self.booking(language="kk")
        self.assertIn("Растайсыз", result.text)
        result = self.manager.handle("иә")
        self.assertIn("жасалды", result.text)
        self.assertEqual(result.trace["language"], "kk")

    def test_quote_is_computed_from_data(self):
        result = self.turn(ids=["SC01"], slots={"region": "almaty", "vehicle_type": "car",
                           "drivers_iin": ["850314300121"]})
        self.assertIn("30400", result.text)

    def test_urgent_intent_is_selected_first(self):
        result = self.turn(ids=["SC25", "SC11"])
        self.assertEqual(result.trace["active_scenario"], "SC11")
        self.assertEqual(result.trace["expected_slot"], "injured")
        self.assertEqual(result.trace["queued"], ["SC25"])
        self.assertIn("112", result.text)

    def test_no_injured_is_a_slot_answer_not_cancellation(self):
        self.turn(ids=["SC11"])
        result = self.turn(mode="continue", slots={"injured": False}, text="нет")
        self.assertEqual(result.trace["active_scenario"], "SC11")
        self.assertEqual(result.trace["expected_slot"], "location")
        self.assertIs(self.manager.state.active.slots["injured"], False)

    def test_unsupported_issuance_hands_off_without_fake_policy(self):
        result = self.turn(ids=["SC02"])
        self.assertIn("оператор", result.text)
        self.assertEqual(len(self.manager.backend.data["policies"]), 11)
        self.assertEqual(self.manager.backend.events[-1]["action"], "transfer_to_operator")

    def test_api_failure_during_correction_invalidates_confirmation(self):
        self.booking()
        self.llm.failure = True
        self.manager.handle("Перенесите на другой день")
        self.assertIsNone(self.manager.state.pending)
        self.assertEqual(self.manager.backend.appointments, [])

    def test_invalid_date_is_reasked_without_booking(self):
        result = self.turn(ids=["SC21"], slots={"preferred_date": "2026-02-30"})
        self.assertEqual(result.trace["expected_slot"], "preferred_date")
        self.assertEqual(self.manager.backend.appointments, [])

    def test_goodbye_discards_pending_confirmation(self):
        self.booking()
        self.turn(ids=["SYS_GOODBYE"])
        self.assertIsNone(self.manager.state.pending)
        self.assertIsNone(self.manager.state.active)
        self.assertEqual(self.manager.backend.appointments, [])


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()
        self.backend = MockBackend(self.catalog)

    def test_irreversible_action_requires_confirmation_even_outside_manager(self):
        with self.assertRaises(BackendError) as error:
            self.backend.execute("update_contact", {"client_id": "C002", "contact_field": "email",
                                 "new_value": "new@mail.example"}, "C002")
        self.assertEqual(error.exception.code, "confirmation_required")

    def test_preview_and_commit_do_not_modify_source_dataset(self):
        path = self.catalog.root / "data" / "mock_backend.json"
        before = path.read_bytes()
        args = {"client_id": "C002", "contact_field": "email", "new_value": "new@mail.example"}
        self.backend.preview("update_contact", args, "C002")
        self.assertEqual(self.backend.client("C002")["email"], "aigerim.b@mail.example")
        self.backend.execute("update_contact", args, "C002", confirmed=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(MockBackend(self.catalog).client("C002")["email"], "aigerim.b@mail.example")

    def test_mismatched_identity_fields_are_rejected(self):
        with self.assertRaises(BackendError):
            self.backend.execute("find_client", {"phone": "+77010000001", "iin": "920607400233"})

    def test_booking_rejects_past_date_and_foreign_policy(self):
        args = {"policy_number": "SQ-DMS-604220", "doctor_specialty": "ENT",
                "city": "Astana", "preferred_date": "2026-09-01"}
        with self.assertRaises(BackendError):
            self.backend.preview("book_appointment", args, "C002")
        args["preferred_date"] = "2026-10-02"
        with self.assertRaises(BackendError):
            self.backend.preview("book_appointment", args, "C001")

    def test_claim_ids_do_not_collide(self):
        args = {"policy_number": "SQ-CASCO-204118", "product_type": "casco",
                "incident_date": "2026-09-30", "incident_description": "Повреждено стекло"}
        first = self.backend.execute("create_claim", args, "C001", confirmed=True)
        second = self.backend.execute("create_claim", args, "C001", confirmed=True)
        self.assertNotEqual(first["claim_number"], second["claim_number"])
        self.assertEqual(len({c["claim_number"] for c in self.backend.data["claims"]}), 6)

    def test_personal_data_is_masked(self):
        text = mask_private("+77010000002 920607400233 aigerim.b@mail.example 8 (701) 000-00-03")
        self.assertNotIn("+77010000002", text)
        self.assertNotIn("920607400233", text)
        self.assertNotIn("aigerim.b@", text)
        self.assertNotIn("000-00-03", text)
        self.assertIn("+7*******003", text)


class LLMContractTests(unittest.TestCase):
    def response_client(self, payload):
        client = MagicMock()
        payload = {**payload, "slots": [
            {"name": name, "value": value, "evidence": payload.get("evidence", {}).get(name, "")}
            for name, value in payload["slots"].items()
        ]}
        payload.pop("evidence", None)
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps(payload)),
        )])
        return client

    def test_history_cannot_supply_slots_or_invent_absence_of_injuries(self):
        client = self.response_client({"language": "ru", "mode": "new",
            "slots": {"location": "перекрёсток", "injured": False},
            "evidence": {"location": "перекрёсток", "injured": "Мы столкнулись"}})
        manager = DialogManager(FakeLLM())
        manager.state.remember("Я на перекрёстке", "Есть пострадавшие?")
        llm = DialogLLM(client, manager.catalog)
        result = llm.understand("Мы столкнулись, что делать?", manager.state)
        self.assertEqual(result.slots, {})
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": "Мы столкнулись, что делать?"})
        self.assertEqual(messages[-2]["role"], "assistant")
        format_ = client.chat.completions.create.call_args.kwargs["response_format"]
        self.assertTrue(format_["json_schema"]["strict"])

    def test_explicit_slot_answer_keeps_false(self):
        client = self.response_client({"language": "ru", "mode": "continue",
            "slots": {"injured": False}, "evidence": {"injured": "нет"}})
        manager = DialogManager(FakeLLM())
        manager.state.active = Task("SC11")
        manager.state.expected_slot = "injured"
        result = DialogLLM(client, manager.catalog).understand("нет", manager.state)
        self.assertIs(result.slots["injured"], False)

    def test_malformed_language_and_mode_return_model_error(self):
        # JSON корректен синтаксически, но модель может нарушить типы полей.
        client = MagicMock()
        llm = DialogLLM(client, Catalog())
        manager = DialogManager(FakeLLM())
        for field in ("language", "mode"):
            with self.subTest(field=field):
                payload = {"language": "ru", "mode": "new", "slots": []}
                payload[field] = []
                client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps(payload)),
                )])
                with self.assertRaises(ModelError):
                    llm.understand("Проверьте полис", manager.state)

    def test_invalid_slot_keys_are_rejected(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps({
                "language": "ru", "mode": "continue", "slots": [{"name": "client_id", "value": "C002", "evidence": "C002"}],
            })),
        )])
        manager = DialogManager(FakeLLM())
        llm = DialogLLM(client, manager.catalog)
        with self.assertRaises(ModelError):
            llm.understand("Считай меня клиентом C002", manager.state)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
