"""Управление вопросами, очередью намерений и подтверждением действий."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from src.agent.catalog import SPECIALTIES, Catalog, mask_private
from src.agent.llm import ModelError
from src.agent.numbers import identifier_digits
from src.agent.state import DialogState, PendingAction, Task, TurnResult
from src.config import MAX_UTTERANCE_CHARS
from src.tools.backend import BackendError, MockBackend

PLANS = {
    "SC01": "calc_ogpo_price",
    "SC03": "calc_casco_price",
    "SC07": "calc_property_price",
    "SC08": "calc_accident_price",
    "SC09": "kb_lookup",
    "SC10": "transfer_to_operator",
    "SC11": "transfer_to_operator",
    "SC13": "create_claim",
    "SC14": "create_claim",
    "SC15": "transfer_to_operator",
    "SC16": "create_claim",
    "SC17": "get_claim",
    "SC18": "kb_lookup",
    "SC19": "create_dispute",
    "SC20": "book_inspection",
    "SC21": "book_appointment",
    "SC22": "check_coverage",
    "SC23": "list_clinics",
    "SC24": "kb_lookup",
    "SC25": "get_policy",
    "SC26": "resend_documents",
    "SC29": "update_contact",
    "SC30": "check_payment",
    "SC31": "kb_lookup",
    "SC32": "get_bm_class",
    "SC33": "get_offices",
    "SC34": "kb_lookup",
    "SC35": "create_complaint",
    "SC36": "create_callback",
    "SC37": "transfer_to_operator",
    "SC38": "report_fraud",
    "SC39": "request_document",
    "SC40": "kb_lookup",
}
PRODUCTS = {
    "SC13": "casco",
    "SC14": "property",
    "SC15": "travel",
    "SC16": "accident",
    "SC21": "dms",
    "SC22": "dms",
    "SC24": "dms",
}
YES = {
    "да",
    "подтверждаю",
    "да подтверждаю",
    "согласен",
    "согласна",
    "верно",
    "иә",
    "ия",
    "растаймын",
    "иә растаймын",
}
NO = {"нет", "не надо", "отмена", "отмените", "жоқ", "бас тартамын"}
GREETINGS = {
    "здравствуйте": "ru",
    "здравствуй": "ru",
    "привет": "ru",
    "добрый день": "ru",
    "доброе утро": "ru",
    "добрый вечер": "ru",
    "алло": "ru",
    "сәлем": "kk",
    "сәлеметсіз бе": "kk",
    "салем": "kk",
    "салеметсиз бе": "kk",
    "қайырлы күн": "kk",
    "қайырлы таң": "kk",
    "қайырлы кеш": "kk",
}
OPERATOR_ACKS = {
    "да подключайся": "ru",
    "да подключайте": "ru",
    "подключайте": "ru",
    "да соединяйте": "ru",
    "соединяйте": "ru",
    "жду оператора": "ru",
    "иә қосыңыз": "kk",
    "қосыңыз": "kk",
    "операторды күтемін": "kk",
}


class DialogManager:
    """Ведёт один разговор и выполняет действия после проверок и подтверждения."""

    def __init__(
        self,
        llm,
        catalog: Catalog | None = None,
        backend: MockBackend | None = None,
        *,
        parallel: bool = True,
    ):
        self.catalog = catalog or Catalog()
        self.backend = backend or MockBackend(self.catalog)
        self.llm = llm
        self.state = DialogState()
        self._actions: list[dict] = []
        self._routed: list[str] = []
        self._event: str | None = None
        self.parallel = parallel
        self._timings: dict[str, int] = {}
        self._router_called = False
        self._parallel_routing_used = False
        self._completed_slots: dict = {}

    def _timed(self, stage, function, *args):
        started = time.perf_counter()
        try:
            return function(*args)
        finally:
            self._timings[stage] = self._timings.get(stage, 0) + round(
                (time.perf_counter() - started) * 1000
            )

    def _route(self, text: str) -> list[str]:
        self._router_called = True
        return self._timed("router", self.llm.route, text)

    def say(self, ru: str, kk: str) -> str:
        return kk if self.state.language == "kk" else ru

    def handle(self, text: str) -> TurnResult:
        """Обработать реплику и вернуть ответ с обезличенной трассировкой."""
        started = time.perf_counter()
        self._actions, self._routed = [], []
        self._event = None
        self._timings = {"triage": 0, "router": 0, "response": 0}
        self._router_called = False
        self._parallel_routing_used = False
        self._completed_slots = {}
        self.state.turn += 1
        try:
            reply = self._handle(text.strip())
        except ModelError as exc:
            self._event = "model_error"
            logging.getLogger(__name__).warning("Ошибка диалоговой модели: %s", exc)
            reply = self.say(
                "Не удалось разобрать ответ. Повторите, пожалуйста.",
                "Жауапты түсіне алмадым. Қайталап айтыңызшы.",
            )
        except BackendError as exc:
            reply = self._backend_error(exc)
        # Считаем именно подряд идущие неясные реплики, включая локальные ответы.
        if "SYS_UNCLEAR" not in self._routed:
            self.state.unclear_count = 0
        reply = mask_private(reply)
        if text.strip() and len(text) <= MAX_UTTERANCE_CHARS:
            self.state.remember(text, reply)
        active = self.state.active
        trace = mask_private(
            {
                "turn": self.state.turn,
                "transcript": text,
                "language": self.state.language,
                "event": self._event,
                "scenarios": self._routed or ([active.scenario_id] if active else []),
                "active_scenario": active.scenario_id if active else None,
                "client_id": self.state.client_id,
                "slots": active.slots if active else self._completed_slots,
                "router_called": self._router_called,
                "routing": None,
                "alternatives": [],
                "reason": {
                    "greeting": "Отдельное приветствие: локальный ответ без выбора бизнес-сценария.",
                    "confirmed_action": "Явное подтверждение ранее показанной операции.",
                    "slot_answer": "Заполнение ожидаемого поля; сценарий сохранён.",
                    "continuation": "Продолжение активного сценария после проверки полей.",
                    "resume": "Возврат к сохранённой задаче.",
                    "operator_handoff": "Передача оператору по правилам диалогового движка.",
                }.get(
                    self._event, "Выбор LLM по каталогу; подробное объяснение доступно в веб-демо."
                ),
                "expected_slot": self.state.expected_slot,
                "awaiting_confirmation": self.state.pending is not None,
                "awaiting_resume": self.state.awaiting_resume,
                "operator_handoff": self.state.operator_handoff,
                "queued": [t.scenario_id for t in self.state.queue],
                "suspended": [t.scenario_id for t in self.state.suspended],
                "actions": self._actions,
                "latency_ms": {
                    **self._timings,
                    "dialog": round((time.perf_counter() - started) * 1000),
                },
                "parallel_routing": self._parallel_routing_used,
            }
        )
        return TurnResult(reply, trace)

    def _handle(self, text: str) -> str:
        if not text or len(text) > MAX_UTTERANCE_CHARS:
            self.state.pending = None
            self.state.awaiting_resume = False
            return self.say(
                "Напишите коротко, чем помочь.", "Қалай көмектесе аламын? Қысқаша жазыңызшы."
            )
        answer = re.sub(r"[^\w\s]", "", text.casefold()).strip()
        answer = " ".join(answer.split())
        # Только отдельное приветствие: «Здравствуйте, проверьте полис» идёт в LLM.
        if answer in GREETINGS:
            self.state.language = GREETINGS[answer]
            self._event = "greeting"
            reply = self.say("Здравствуйте!", "Сәлеметсіз бе!")
            if self.state.operator_handoff:
                return reply + " " + self._handoff_status()
            if self.state.pending:
                return (
                    reply
                    + " "
                    + self._confirmation_text(self.state.pending.name, self.state.pending.preview)
                )
            if self.state.expected_slot:
                return reply + " " + self._ask(self.state.expected_slot)
            if not self.state.active and (self.state.queue or self.state.suspended):
                self.state.awaiting_resume = True
                return reply + self.say(
                    " Вернёмся к оставшемуся вопросу?", " Қалған сұраққа оралайық па?"
                )
            return reply + self.say(" Чем помочь?", " Қалай көмектесе аламын?")
        if answer in {"иә", "ия", "растаймын", "иә растаймын", "жоқ", "бас тартамын"}:
            self.state.language = "kk"
        elif answer in YES | NO:
            self.state.language = "ru"
        if self.state.operator_handoff and (answer in YES or answer in OPERATOR_ACKS):
            self.state.language = OPERATOR_ACKS.get(answer, self.state.language)
            return self._handoff_status()
        if answer in YES and self.state.pending:
            self._event = "confirmed_action"
            self._routed = [self.state.active.scenario_id] if self.state.active else []
            pending = self.state.pending
            # Снять подтверждение ДО выполнения: повторное «да» не повторяет действие.
            self.state.pending = None
            result = self._call(pending.name, pending.arguments, confirmed=True)
            return self._complete(self._result_text(pending.name, result))
        # «Нет» на вопрос о пострадавших — значение поля, а не отмена диалога.
        if answer in NO and (
            self.state.pending
            or not self.state.active
            or answer in {"не надо", "отмена", "отмените", "бас тартамын"}
        ):
            self.state.pending = None
            if self.state.active:
                return self._complete(
                    self.say("Хорошо, этот запрос отменён.", "Жақсы, бұл сұрау тоқтатылды.")
                )
            self.state.queue.clear()
            self.state.suspended.clear()
            self.state.awaiting_resume = False
            self.state.operator_handoff = False
            return self.say("Хорошо. Чем ещё помочь?", "Жақсы. Тағы қалай көмектесе аламын?")
        if answer in YES and not self.state.active and self.state.awaiting_resume:
            return self._resume()

        # Уже записанные цифры не отправляем модели для повторного переписывания.
        # Эта ветка заполняет ожидаемое поле, а не выбирает бизнес-сценарий.
        field = self.state.expected_slot
        if (
            self.state.active
            and field == "injured"
            and answer in {"да", "есть", "нет", "иә", "ия", "бар", "жоқ"}
        ):
            self._event = "slot_answer"
            self.state.pending = None
            self.state.language = "kk" if answer in {"иә", "ия", "бар", "жоқ"} else "ru"
            self.state.active.slots[field] = answer in {"да", "есть", "иә", "ия", "бар"}
            return self._advance()
        if (
            self.state.active
            and field == "location"
            and re.fullmatch(
                r"(?:ул(?:ица)?\.?\s*)?[a-zа-яёәғқңөұүһі]\s*[-–]?\s*\d{1,4}\s*,?\s*"
                r"(?:д(?:ом)?\.?|үй)\s*\d{1,4}(?:[/-]\d{1,4})?[a-zа-я]?[.!]?",
                text,
                re.IGNORECASE,
            )
        ):
            # Короткая запись улицы/дома заполняет только ожидаемый адрес.
            # Это сообщение клиента, а не результат проверки по карте.
            self._event = "slot_answer"
            self.state.pending = None
            self.state.active.slots[field] = text
            return self._advance()
        numeric = identifier_digits(text) if field in {"phone", "iin"} else None
        if self.state.active and numeric is not None:
            self._event = "slot_answer"
            self.state.pending = None
            try:
                value = self.catalog.normalize(field, numeric)
            except ValueError:
                return self._ask(field, invalid=True)
            self.state.active.slots[field] = value
            return self._advance()

        # Любое уточнение или смена темы требует нового чтения условий операции.
        prefetched = None
        try:
            if (
                self.parallel
                and not self.state.active
                and not (self.state.queue or self.state.suspended or self.state.operator_handoff)
            ):
                # Независимые чтения выполняются параллельно. Ни один поток
                # не меняет состояние и не вызывает бизнес-действия.
                self._parallel_routing_used = True
                with ThreadPoolExecutor(max_workers=2) as workers:
                    routing = workers.submit(self._route, text)
                    understood = self._timed("triage", self.llm.understand, text, self.state)
                    prefetched = routing.result()
            else:
                understood = self._timed("triage", self.llm.understand, text, self.state)
        finally:
            self.state.pending = None
            self.state.awaiting_resume = False
        self.state.language = understood.language
        if understood.mode == "cancel":
            return self._complete(self.say("Запрос отменён.", "Сұрау тоқтатылды."))
        if understood.mode == "resume" and (self.state.queue or self.state.suspended):
            if self.state.operator_handoff and not re.search(
                r"верн|возобнов|продолж|орала|жалғастыр", answer
            ):
                return self._handoff_status()
            return self._resume()

        normalized, invalid = {}, []
        for key, value in understood.slots.items():
            if value is None:
                continue
            try:
                normalized[key] = self.catalog.normalize(key, value)
            except (ValueError, TypeError, OverflowError):
                invalid.append(key)
        ids = prefetched
        if understood.mode == "continue" and self.state.active:
            # Проверяем длинные ответы и ответы без подходящего поля независимым
            # LLM-router: ошибочное continue не должно поглощать новую просьбу.
            expected = self.state.expected_slot
            if len(text.split()) > 8 or (expected and expected not in normalized):
                candidate_ids = self._route(text)
                business_ids = [sid for sid in candidate_ids if sid in self.catalog.scenarios]
                if business_ids and business_ids != [self.state.active.scenario_id]:
                    ids = candidate_ids
        if understood.mode == "continue" and self.state.active and ids is None:
            self._event = "continuation"
            self.state.active.slots.update(normalized)
            if invalid:
                self.state.active.slots.pop(invalid[0], None)
                return self._ask(invalid[0], invalid=True)
            return self._advance()

        ids = ids if ids is not None else self._route(text)
        self._routed = ids
        if any(s not in self.catalog.scenarios and s not in self.catalog.system for s in ids):
            raise ModelError("Unknown scenario")
        business = list(dict.fromkeys(s for s in ids if s in self.catalog.scenarios))
        if not business:
            sid = ids[0] if ids else "SYS_UNCLEAR"
            if sid == "SYS_GOODBYE":
                self.state.active = None
                self.state.queue.clear()
                self.state.suspended.clear()
                self.state.expected_slot = None
                self.state.awaiting_resume = False
                self.state.operator_handoff = False
            elif sid == "SYS_UNCLEAR":
                self.state.unclear_count += 1
                if self.state.unclear_count >= 2:
                    return self._handoff("operator_general", "Две неясные реплики подряд")
                return self.say(
                    "Что нужно: узнать условия, проверить полис или сообщить о страховом случае?",
                    "Шарттарды білу, полисті тексеру немесе сақтандыру оқиғасын хабарлау керек пе?",
                )
            else:
                self.state.unclear_count = 0
            return self.catalog.system[sid]["response"][self.state.language]
        self.state.unclear_count = 0
        self.state.operator_handoff = False
        # Стабильная сортировка сохраняет порядок просьб внутри приоритета.
        business.sort(key=lambda sid: self.catalog.scenarios[sid]["priority"] != "urgent")
        tasks = [Task(sid, deepcopy(normalized)) for sid in business]
        for task in tasks:
            # Исходная просьба нужна для информационного ответа и передачи оператору.
            task.slots.setdefault("topic", text)
        if self.state.active:
            self.state.suspended.append(self.state.active)
        self.state.active = tasks[0]
        self.state.queue = tasks[1:] + self.state.queue
        self.state.expected_slot = None
        if invalid:
            return self._ask(invalid[0], invalid=True)
        return self._advance()

    def _ask(self, field: str, *, invalid=False) -> str:
        self.state.expected_slot = field
        if field == "injured":
            # Один вопрос: «да» больше не может означать одновременно
            # «все целы» и «есть пострадавшие».
            return self.say("Есть ли пострадавшие?", "Зардап шеккендер бар ма?")
        if field == "phone" and invalid:
            return self.say(
                "Не удалось получить полный номер. Напишите его текстом: +7 и ещё 10 цифр "
                "или 10 цифр без кода страны; учебные номера доступны в /demo.",
                "Толық нөмірді ала алмадым. Мәтінмен жазыңыз: +7 және тағы 10 цифр "
                "немесе ел кодынсыз 10 цифр; оқу нөмірлері /demo командасында.",
            )
        question = self.catalog.slots[field]["prompt"][self.state.language]
        prefix = (
            self.say("Проверьте формат данных. ", "Деректердің пішімін тексеріңізші. ")
            if invalid
            else ""
        )
        return prefix + question

    def _call(self, name: str, arguments: dict, *, confirmed=False, preview=False) -> dict:
        try:
            result = (
                self.backend.preview(name, arguments, self.state.client_id)
                if preview
                else self.backend.execute(
                    name, arguments, self.state.client_id, confirmed=confirmed
                )
            )
        except BackendError as exc:
            self._actions.append(
                {"name": name, "mode": "preview" if preview else "execute", **exc.as_dict()}
            )
            raise
        self._actions.append(
            {"name": name, "mode": "preview" if preview else "execute", "result": result}
        )
        return result

    def _identify(self) -> str | None:
        task = self.state.active
        if self.state.client_id:
            return None
        credentials = {key: task.slots[key] for key in ("phone", "iin") if task.slots.get(key)}
        if not credentials:
            return self._ask("phone")
        result = self._call("find_client", credentials)
        self.state.client_id = result["client_id"]
        return None

    def _advance(self) -> str:
        task = self.state.active
        if not task:
            return self.say("Чем помочь?", "Қалай көмектесе аламын?")
        sid = task.scenario_id
        self._routed = self._routed or [sid]
        if sid == "SC21" and not task.slots.get("doctor_specialty"):
            # Иногда LLM кладёт явно названного врача в соседнее медицинское поле.
            # Принимаем только точную специальность из словаря алиасов.
            try:
                specialty = self.catalog.normalize(
                    "doctor_specialty", task.slots.get("service_name")
                )
            except (ValueError, TypeError):
                pass
            else:
                if specialty in SPECIALTIES.values():
                    task.slots["doctor_specialty"] = specialty
        scenario = self.catalog.scenarios[sid]
        action = PLANS.get(sid)
        if action is None:
            return self._handoff(
                "operator_general", "Операция пока доступна через оператора", unsupported=True
            )
        if scenario["requires_identification"]:
            question = self._identify()
            if question:
                return question
        if self.state.client_id:
            profile = self.backend.client(self.state.client_id)
            for key in ("phone", "email", "city"):
                task.slots.setdefault(key, profile[key])

        # Автоматически выбрать единственный подходящий полис, иначе уточнить.
        required = list(scenario["slots"]["required"])
        # Например, франшиза необязательна для консультации, но нужна для расчёта.
        # Спрашиваем её до вызова backend, а не после технической ошибки действия.
        required.extend(
            field
            for field in self.catalog.actions[action]["inputs"]
            if field in self.catalog.slots and field not in required
        )
        if "policy_number" in required:
            product = PRODUCTS.get(sid) or task.slots.get("product_type")
            if not task.slots.get("policy_number"):
                policies = self._call("get_policies", {"client_id": self.state.client_id})[
                    "policies"
                ]
                choices = [p for p in policies if not product or p["product"] == product]
                if len(choices) == 1:
                    task.slots["policy_number"] = choices[0]["policy_number"]
                elif not choices:
                    return self._handoff("operator_general", "Подходящий полис клиента не найден")
                else:
                    self.state.expected_slot = "policy_number"
                    items = ", ".join(p["policy_number"] for p in choices)
                    return self.say(
                        f"Найдено несколько полисов: {items}. Какой нужен?",
                        f"Бірнеше полис табылды: {items}. Қайсысы керек?",
                    )
            policy = self.backend.policy(task.slots["policy_number"], self.state.client_id)
            if product and policy["product"] != product:
                raise BackendError("invalid_input", "Нужен другой тип полиса.", "policy_number")
            task.slots["product_type"] = policy["product"]
        if (
            "claim_number" in required
            and not task.slots.get("claim_number")
            and self.state.client_id
        ):
            claims = self.backend.claims(self.state.client_id)
            if len(claims) == 1:
                task.slots["claim_number"] = claims[0]["claim_number"]
        emergency_notice = ""
        if (
            sid == "SC11"
            and task.slots.get("injured") is True
            and not task.attempts.get("injury_advised")
        ):
            task.attempts["injury_advised"] = 1
            emergency_notice = self.say(
                "Если ещё не звонили, немедленно позвоните 112. "
                "Это демо не вызывает экстренные службы. ",
                "Әлі хабарласпасаңыз, дереу 112-ге қоңырау шалыңыз. "
                "Бұл демо жедел қызметтерді шақырмайды. ",
            )
        for key in required:
            if task.slots.get(key) in (None, "", []):
                question = self._ask(key)
                if (
                    sid == "SC11"
                    and task.slots.get("injured") is not True
                    and not task.attempts.get("urgency_advised")
                ):
                    task.attempts["urgency_advised"] = 1
                    question = (
                        self.say(
                            "Если есть пострадавшие, сначала звоните 112. ",
                            "Зардап шеккендер болса, алдымен 112-ге қоңырау шалыңыз. ",
                        )
                        + question
                    )
                return emergency_notice + question
        self.state.expected_slot = None
        arguments = deepcopy(task.slots)
        arguments.update(client_id=self.state.client_id, scenario_id=sid)
        if action == "transfer_to_operator":
            arguments["queue"] = scenario.get("handoff", {}).get("queue", "operator_general")
            arguments["summary"] = self._summary()
        if action == "get_bm_class" and sid == "SC32":
            arguments["iin"] = task.slots["iin"]
        if self.catalog.actions[action]["irreversible"]:
            preview = self._call(action, arguments, preview=True)
            self.state.pending = PendingAction(action, deepcopy(arguments), deepcopy(preview))
            return self._confirmation_text(action, preview)
        result = self._call(action, arguments)
        reply = self._result_text(action, result)
        if action == "transfer_to_operator":
            return self._finish_handoff(emergency_notice + reply)
        if sid == "SC30" and any(
            p["status"] == "charged_policy_not_issued" for p in result["payments"]
        ):
            self._call(
                "transfer_to_operator", {"queue": "operator_general", "summary": self._summary()}
            )
            reply += self.say(
                " Запрос передан оператору в демо.", " Сұрау демода операторға жіберілді."
            )
            return self._finish_handoff(reply)
        if sid == "SC38":
            self._call(
                "transfer_to_operator", {"queue": "security_team", "summary": self._summary()}
            )
            reply = self.say(
                "Обращение зарегистрировано и передано службе безопасности в демо. Не сообщайте SMS-коды, CVV и PIN.",
                "Өтініш тіркеліп, демода қауіпсіздік қызметіне жіберілді. SMS кодын, CVV және PIN кодын айтпаңыз.",
            )
            return self._finish_handoff(reply)
        return self._complete(reply)

    def _confirmation_text(self, action: str, result: dict) -> str:
        question = self.say("Подтверждаете?", "Растайсыз ба?")
        if action in {"book_appointment", "book_inspection"}:
            place = result.get("clinic_name", result["address"])
            specialties = {
                "therapist": ("терапевт", "терапевт"),
                "ENT": ("ЛОР", "ЛОР"),
                "dentist": ("стоматолог", "тіс дәрігері"),
                "gynecologist": ("гинеколог", "гинеколог"),
                "cardiologist": ("кардиолог", "кардиолог"),
                "pediatrician": ("педиатр", "педиатр"),
                "lab": ("анализы", "талдаулар"),
                "ultrasound": ("УЗИ", "УДЗ"),
            }
            specialty = self.say(
                *specialties.get(result.get("doctor_specialty"), ("осмотр", "тексеру"))
            )
            text = self.say(
                f"Запись: {specialty}, {place}, {result['slot_datetime']}.",
                f"Жазылу: {specialty}, {place}, {result['slot_datetime']}.",
            )
        elif action == "update_contact":
            labels = {
                "phone": ("телефон", "телефон"),
                "email": ("почту", "пошта"),
                "address": ("адрес", "мекенжай"),
            }
            field = self.say(*labels[result["contact_field"]])
            text = self.say(
                f"Изменить {field} на {mask_private(result['new_value'])}.",
                f"{field}: {mask_private(result['new_value'])} деп өзгертемін.",
            )
        elif action == "create_claim":
            text = self.say(
                f"Зарегистрировать событие от {result['incident_date']} по полису {result['policy_number']}: {result['incident_description']}",
                f"{result['policy_number']} полисі бойынша {result['incident_date']} күнгі оқиғаны тіркеймін: {result['incident_description']}",
            )
        else:
            text = self.say(
                f"Зарегистрировать несогласие по заявлению {result['claim_number']}: {result['complaint_text']}",
                f"{result['claim_number']} өтініші бойынша келіспеушілікті тіркеймін: {result['complaint_text']}",
            )
        return text + " " + question

    def _result_text(self, action: str, result: dict) -> str:
        if action == "get_policy":
            statuses = {
                "active": ("действует", "жарамды"),
                "expired": ("истёк", "мерзімі өткен"),
                "not_started": ("ещё не вступил в силу", "әлі күшіне енген жоқ"),
            }
            status = self.say(*statuses[result["status"]])
            return self.say(
                f"Полис {result['policy_number']} {status}. Срок: {result['start_date']} — {result['end_date']}.",
                f"{result['policy_number']} полисі {status}. Мерзімі: {result['start_date']} — {result['end_date']}.",
            )
        if action.startswith("calc_"):
            return self.say(
                f"Стоимость на {result['term_months']} месяцев — {result['price']} тенге.",
                f"{result['term_months']} айға бағасы — {result['price']} теңге.",
            )
        if action in {"book_appointment", "book_inspection"}:
            place = result.get("clinic_name", result["address"])
            return self.say(
                f"Запись создана в демо: {place}, {result['slot_datetime']}.",
                f"Демода жазылу жасалды: {place}, {result['slot_datetime']}.",
            )
        if action == "update_contact":
            return self.say(
                "Контактные данные обновлены в демо.", "Демода байланыс деректері жаңартылды."
            )
        if action == "create_claim":
            return self.say(
                f"Заявление {result['claim_number']} зарегистрировано в демо.",
                f"{result['claim_number']} өтініші демода тіркелді.",
            )
        if action == "transfer_to_operator":
            return self.say(
                "Запрос и контекст переданы оператору в демо.",
                "Сұрау мен сөйлесу мәнмәтіні демода операторға жіберілді.",
            )
        if action in {"resend_documents", "request_document"}:
            return self.say(
                f"Отправка документов на {mask_private(result['sent_to'])} отмечена в демо.",
                f"Демода құжаттар {mask_private(result['sent_to'])} мекенжайына жіберілді деп белгіленді.",
            )
        if action in {"create_complaint", "create_dispute", "create_callback", "report_fraud"}:
            return self.say("Обращение зарегистрировано в демо.", "Өтініш демода тіркелді.")
        try:
            return self._timed(
                "response",
                self.llm.respond,
                self.state.language,
                f"Сообщить результат {action}",
                result,
            )
        except ModelError:
            return self.say(
                "Данные получены, но не удалось сформулировать ответ. Повторите запрос, пожалуйста.",
                "Деректер алынды, бірақ жауапты құрастыру мүмкін болмады. Сұрауды қайталаңызшы.",
            )

    def _complete(self, reply: str, *, offer_resume=True) -> str:
        if self.state.active:
            self._completed_slots = deepcopy(self.state.active.slots)
        self.state.active = None
        self.state.expected_slot = None
        self.state.pending = None
        self.state.operator_handoff = False
        self.state.awaiting_resume = offer_resume and bool(self.state.queue or self.state.suspended)
        if self.state.awaiting_resume:
            reply += self.say(" Вернёмся к оставшемуся вопросу?", " Қалған сұраққа оралайық па?")
        return reply

    def _resume(self) -> str:
        self._event = "resume"
        self.state.awaiting_resume = False
        self.state.operator_handoff = False
        if self.state.queue:
            next_task = self.state.queue.pop(0)
        elif self.state.suspended:
            next_task = self.state.suspended.pop()
        else:
            return self.say(
                "Отложенных вопросов нет. Чем помочь?",
                "Кейінге қалдырылған сұрақ жоқ. Қалай көмектесейін?",
            )
        if self.state.active:
            self.state.suspended.append(self.state.active)
        self.state.active = next_task
        self.state.pending = None
        self.state.expected_slot = None
        return self._advance()

    def _summary(self) -> dict:
        return {
            "client_id": self.state.client_id,
            "scenario": self.state.active.scenario_id if self.state.active else None,
            "slots": deepcopy(self.state.active.slots) if self.state.active else {},
            "history": deepcopy(self.state.history[-4:]),
        }

    def _handoff(self, queue: str, reason: str, *, unsupported=False) -> str:
        self._call(
            "transfer_to_operator",
            {"queue": queue, "summary": {**self._summary(), "reason": reason}},
        )
        self.state.unclear_count = 0
        prefix = (
            self.say("Эту операцию выполняет оператор. ", "Бұл әрекетті оператор орындайды. ")
            if unsupported
            else ""
        )
        return self._finish_handoff(
            prefix
            + self.say(
                "Контекст передан оператору в демо.", "Мәнмәтін демода операторға жіберілді."
            )
        )

    def _finish_handoff(self, reply: str) -> str:
        reply = self._complete(reply, offer_resume=False)
        self.state.operator_handoff = True
        self._event = "operator_handoff"
        return (
            reply
            + " "
            + self.say(
                "Живой оператор в этом демо не подключается.",
                "Бұл демода нақты оператор қосылмайды.",
            )
        )

    def _handoff_status(self) -> str:
        self._event = "operator_handoff"
        reply = self.say(
            "Передача оператору здесь учебная: живой оператор не подключается.",
            "Операторға жіберу — оқу әрекеті: нақты оператор қосылмайды.",
        )
        if self.state.queue or self.state.suspended:
            reply += self.say(
                " Чтобы продолжить с ботом, скажите «вернёмся к предыдущему вопросу».",
                "Ботпен жалғастыру үшін «алдыңғы сұраққа оралайық» деңіз.",
            )
        return reply

    def _backend_error(self, exc: BackendError) -> str:
        self.state.pending = None
        task = self.state.active
        if not task:
            return self.say(
                "Действие недоступно. Попробуйте другой запрос.",
                "Әрекет қолжетімсіз. Басқа сұрауды қолданып көріңіз.",
            )
        field = exc.field
        task.attempts[field or exc.code] = task.attempts.get(field or exc.code, 0) + 1
        if field in {"phone", "iin"} and not self.state.client_id:
            task.slots.pop(field, None)
            attempts = task.attempts.get("phone", 0) + task.attempts.get("iin", 0)
            if attempts >= 3:
                return self._handoff("operator_general", "Не удалось идентифицировать клиента")
            return self.say(
                "Клиент не найден в учебной базе; список тестовых клиентов — /demo. ",
                "Клиент оқу базасынан табылмады; оқу клиенттері — /demo. ",
            ) + self._ask("iin" if attempts >= 2 else field)
        if field in self.catalog.slots and task.attempts[field] <= 2:
            task.slots.pop(field, None)
            prefix = self.say(
                exc.message + " ", "Деректер сәйкес келмейді немесе жазба табылмады. "
            )
            return prefix + self._ask(field)
        return self._handoff("operator_general", exc.message)
