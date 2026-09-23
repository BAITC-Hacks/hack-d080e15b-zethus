"""Учебный бэкенд. Данные и изменения изолированы внутри одного сеанса."""

from copy import deepcopy
from datetime import date, datetime, time
from uuid import uuid4

from src.agent.catalog import Catalog


class BackendError(Exception):
    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.code, self.message, self.field = code, message, field

    def as_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "field": self.field}}


class MockBackend:
    SUPPORTED = frozenset({
        "find_client", "get_policies", "get_policy", "get_claim", "get_bm_class",
        "calc_ogpo_price", "calc_casco_price", "calc_property_price", "calc_accident_price",
        "list_clinics", "get_offices", "check_coverage", "book_appointment",
        "book_inspection", "create_claim", "create_dispute", "update_contact",
        "check_payment", "resend_documents", "request_document", "kb_lookup",
        "create_callback", "create_complaint", "report_fraud", "transfer_to_operator",
    })

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.data = catalog.read("mock_backend")
        self.kb = catalog.read("knowledge_base")
        self.events: list[dict] = []
        self.appointments: list[dict] = []

    def client(self, client_id: str | None) -> dict:
        for client in self.data["clients"]:
            if client["client_id"] == client_id:
                return deepcopy(client)
        raise BackendError("not_found", "Клиент не найден; нужен телефон или ИИН.", "phone")

    def policies(self, client_id: str) -> list[dict]:
        self.client(client_id)
        return [deepcopy(p) for p in self.data["policies"] if p["client_id"] == client_id]

    def claims(self, client_id: str) -> list[dict]:
        self.client(client_id)
        return [deepcopy(c) for c in self.data["claims"] if c["client_id"] == client_id]

    def policy(self, number: str, client_id: str | None, *, active=False) -> dict:
        # Знание чужого номера не даёт доступа к полису другого клиента.
        for policy in self.data["policies"]:
            if policy["policy_number"] == number and policy["client_id"] == client_id:
                result = deepcopy(policy)
                today = self.catalog.today.isoformat()
                result["status"] = ("not_started" if today < result["start_date"] else
                                    "expired" if today > result["end_date"] else "active")
                if active and result["status"] != "active":
                    raise BackendError("policy_inactive", "Полис сейчас не действует.", "policy_number")
                return result
        raise BackendError("not_found", "Полис не найден среди полисов этого клиента.", "policy_number")

    def claim(self, number: str, client_id: str | None) -> dict:
        for claim in self.data["claims"]:
            if claim["claim_number"] == number and claim["client_id"] == client_id:
                return deepcopy(claim)
        raise BackendError("not_found", "Заявление не найдено у этого клиента.", "claim_number")

    def preview(self, name: str, arguments: dict, client_id: str | None) -> dict:
        """Все проверки выполняются, но никаких изменений ещё не происходит."""
        return self._dispatch(name, deepcopy(arguments), client_id, commit=False)

    def execute(self, name: str, arguments: dict, client_id: str | None = None,
                *, confirmed: bool = False) -> dict:
        if name not in self.SUPPORTED:
            raise BackendError("service_unavailable", "Для этого действия нужен оператор.")
        if self.catalog.actions[name]["irreversible"] and not confirmed:
            raise BackendError("confirmation_required", "Сначала требуется подтверждение клиента.")
        result = self._dispatch(name, deepcopy(arguments), client_id, commit=True)
        self.events.append({"action": name, "arguments": deepcopy(arguments), "result": deepcopy(result)})
        return result

    def _dispatch(self, name: str, args: dict, client_id: str | None, *, commit: bool) -> dict:
        if name not in self.SUPPORTED:
            raise BackendError("service_unavailable", "Для этого действия нужен оператор.")
        for required in self.catalog.actions[name]["inputs"]:
            if not any(args.get(option) not in (None, "", []) for option in required.split("|")):
                raise BackendError("invalid_input", "Не хватает данных для действия.", required.split("|")[0])
        if name == "find_client":
            for client in self.data["clients"]:
                if all(args[key] == client[key] for key in ("phone", "iin") if args.get(key)):
                    return {"client_id": client["client_id"], "full_name": client["full_name"]}
            raise BackendError("not_found", "Не удалось найти клиента по указанным данным.",
                               "phone" if args.get("phone") else "iin")
        if name == "get_policies":
            if args["client_id"] != client_id:
                raise BackendError("not_found", "Клиент не найден.")
            return {"policies": self.policies(client_id)}
        if name == "get_policy":
            number = args.get("policy_number")
            if not number:
                found = [p for p in self.policies(client_id)
                         if p["details"].get("vehicle_plate") == args.get("vehicle_plate")]
                if len(found) != 1:
                    raise BackendError("invalid_input", "Уточните номер полиса.", "policy_number")
                number = found[0]["policy_number"]
            return self.policy(number, client_id)
        if name == "get_claim":
            return self.claim(args.get("claim_number"), client_id)
        if name == "get_bm_class":
            iin = self.catalog.normalize("iin", args["iin"])
            value = next((c["bm_class"] for c in self.data["clients"] if c["iin"] == iin),
                         self.data["defaults"]["unknown_iin_bm_class"])
            return {"bm_class": value}
        if name.startswith("calc_"):
            return self._quote(name, args)
        if name == "list_clinics":
            clinics = [c for c in self.kb["clinics"] if c["city"] == args["city"]]
            if args.get("doctor_specialty"):
                clinics = [c for c in clinics if args["doctor_specialty"] in c["specialties"]]
            if not clinics:
                raise BackendError("not_found", "Подходящей клиники в списке нет.", "city")
            return {"clinics": deepcopy(clinics)}
        if name == "get_offices":
            offices = [o for o in self.kb["offices"] if o["city"] == args["city"]]
            if not offices:
                raise BackendError("not_found", "Офис в этом городе не найден.", "city")
            return {"offices": deepcopy(offices)}
        if name == "check_coverage":
            policy = self.policy(args["policy_number"], client_id, active=True)
            if policy["product"] != "dms":
                raise BackendError("invalid_input", "Нужен медицинский полис ДМС.", "policy_number")
            package = policy["details"]["package"]
            return {"package": package, "service": args["service_name"],
                    "terms": deepcopy(self.kb["products"]["dms"]["packages"][package])}
        if name in {"book_appointment", "book_inspection"}:
            return self._book(name, args, client_id, commit=commit)
        if name == "create_claim":
            policy = self.policy(args.get("policy_number"), client_id)
            incident_date = self.catalog.normalize("incident_date", args["incident_date"])
            if incident_date > self.catalog.today.isoformat() or not policy["start_date"] <= incident_date <= policy["end_date"]:
                raise BackendError("policy_inactive", "Дата события вне срока действия полиса.", "incident_date")
            if policy["product"] != args["product_type"]:
                raise BackendError("invalid_input", "Тип полиса не совпадает с заявлением.", "policy_number")
            result = {"policy_number": policy["policy_number"], "incident_date": incident_date,
                      "incident_description": args["incident_description"], "claim_type": args["product_type"]}
            if commit:
                number = max(int(c["claim_number"].split("-")[1]) for c in self.data["claims"]) + 1
                result.update(claim_number=f"CL-{number:06d}", status="registered")
                self.data["claims"].append({**result, "client_id": client_id,
                    "next_step": "Submit the claim documents for review."})
            return result
        if name == "create_dispute":
            self.claim(args["claim_number"], client_id)
            return {**args, **({"ticket_id": "D-" + uuid4().hex[:10]} if commit else {})}
        if name == "update_contact":
            self.client(client_id)
            field = args["contact_field"]
            if field not in {"phone", "email", "address"}:
                raise BackendError("invalid_input", "Недопустимое контактное поле.", "contact_field")
            try:
                value = (self.catalog.normalize(field, args["new_value"]) if field != "address"
                         else self.catalog.normalize("new_value", args["new_value"]))
            except ValueError:
                raise BackendError("invalid_input", "Проверьте формат новых данных.", "new_value")
            if field == "phone" and any(c["phone"] == value and c["client_id"] != client_id for c in self.data["clients"]):
                raise BackendError("invalid_input", "Этот телефон уже привязан к другому клиенту.", "new_value")
            if commit:
                next(c for c in self.data["clients"] if c["client_id"] == client_id)[field] = value
            return {"contact_field": field, "new_value": value}
        if name == "check_payment":
            payments = [p for p in self.data["payments"] if p["client_id"] == client_id and p["date"] == args["payment_date"]]
            if not payments:
                raise BackendError("not_found", "Платёж на эту дату не найден.", "payment_date")
            return {"payments": deepcopy(payments)}
        if name in {"resend_documents", "request_document"}:
            self.policy(args["policy_number"], client_id, active=name == "resend_documents")
            destination = args.get("email") or self.client(client_id)["email"]
            return {"sent_to": destination, "simulated": True}
        if name == "kb_lookup":
            return {"facts": self.knowledge(args.get("scenario_id"), args.get("product_type")),
                    "question": args["topic"]}
        if name == "transfer_to_operator":
            queues = self.catalog.read("actions")["queues"]
            if args["queue"] not in queues:
                raise BackendError("invalid_input", "Неизвестная очередь оператора.")
            return {"queue": args["queue"], "summary": args.get("summary", {}), "simulated": True}
        if name in {"create_callback", "create_complaint", "report_fraud"}:
            return {**args, "simulated": True,
                    **({"ticket_id": "T-" + uuid4().hex[:10]} if commit else {})}
        raise BackendError("service_unavailable", "Действие пока недоступно.")

    def _quote(self, name: str, args: dict) -> dict:
        products = self.kb["products"]
        if name == "calc_ogpo_price":
            rules = products["ogpo"]["pricing"]
            classes = [self._dispatch("get_bm_class", {"iin": i}, None, commit=False)["bm_class"]
                       for i in args["drivers_iin"]]
            coef = max(rules["bm_coef"][c] for c in classes)
            price = rules["base_by_region_kzt"][args["region"]] * rules["vehicle_type_coef"][args["vehicle_type"]] * coef
        elif name == "calc_casco_price":
            rules = products["casco"]["pricing"]
            age = self.catalog.today.year - args["car_year"]
            if not 0 <= age <= rules["max_car_age"]["Standard"]:
                raise BackendError("not_eligible", "Для этого возраста автомобиля нужен оператор.")
            band = "0-3" if age <= 3 else "4-7" if age <= 7 else "8-10"
            price = args["car_value"] * rules["rate_by_car_age"][band] * rules["franchise_coef"][str(args["franchise"])]
        else:
            product = "property" if name == "calc_property_price" else "accident"
            price = products[product]["price_per_year_kzt"].get(str(args["sum_insured"]))
            if price is None:
                raise BackendError("invalid_input", "Для этого продукта нет такой страховой суммы.", "sum_insured")
            if product == "property" and args["property_type"] == "house":
                price *= products[product]["house_coef"]
        return {"price": round(price), "currency": "KZT", "term_months": 12}

    def _book(self, name: str, args: dict, client_id: str | None, *, commit: bool) -> dict:
        requested = date.fromisoformat(args["preferred_date"])
        if requested < self.catalog.today:
            raise BackendError("invalid_input", "Выберите сегодняшнюю или будущую дату.", "preferred_date")
        if name == "book_appointment":
            policy = self.policy(args["policy_number"], client_id, active=True)
            if policy["product"] != "dms":
                raise BackendError("invalid_input", "Для записи нужен полис ДМС.", "policy_number")
            if requested.isoformat() > policy["end_date"]:
                raise BackendError("policy_inactive", "Полис истекает раньше даты приёма.", "preferred_date")
            specialty = args["doctor_specialty"]
            if specialty != "therapist" and policy["details"]["package"] == "Basic":
                raise BackendError("not_covered", "В Basic для специалиста нужно направление; его проверит оператор.")
            choices = [c for c in self.kb["clinics"] if c["city"] == args["city"] and specialty in c["specialties"]]
            if not choices:
                raise BackendError("no_availability", "Подходящей клиники в этом городе нет.", "city")
            place = choices[0]
        else:
            self.claim(args["claim_number"], client_id)
            place = next((p for p in self.kb["inspection_points"] if p["city"] == args["city"]),
                         next(p for p in self.kb["inspection_points"] if p["city"] == "other"))
        # В исходных данных расписания нет: это явно учебное расписание, Пн–Пт.
        if requested.weekday() >= 5:
            raise BackendError("no_availability", "В учебном расписании доступны рабочие дни.", "preferred_date")
        hours = (10, 11, 14, 16)
        occupied = {a["slot_datetime"] for a in self.appointments if a["address"] == place["address"]}
        slot = next((datetime.combine(requested, time(hour=h)).isoformat(timespec="minutes")
                     for h in hours if datetime.combine(requested, time(hour=h)).isoformat(timespec="minutes") not in occupied), None)
        if slot is None:
            raise BackendError("no_availability", "На этот день свободных мест нет.", "preferred_date")
        result = {"slot_datetime": slot, "address": place["address"], "simulated": True}
        if name == "book_appointment":
            result.update(clinic_name=place["name"], doctor_specialty=args["doctor_specialty"])
        if commit:
            self.appointments.append({**result, "client_id": client_id, "action": name})
        return result

    def knowledge(self, scenario_id: str | None, product: str | None = None) -> dict:
        sections = {
            "SC09": "products", "SC11": "claims", "SC15": "products", "SC18": "claims",
            "SC24": "products", "SC31": "payments", "SC32": "bonus_malus", "SC34": "app_help",
            "SC38": "fraud_policy", "SC40": "products",
        }
        section = sections.get(scenario_id, "company")
        facts = self.kb[section]
        if section == "products":
            selected = "dms" if scenario_id in {"SC09", "SC24"} else "travel" if scenario_id == "SC15" else product
            if selected in facts:
                facts = facts[selected]
        return deepcopy(facts)
