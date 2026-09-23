"""Каталог, нормализация и проверка полей из slots.json."""

import json
import re
from datetime import date, timedelta
from pathlib import Path

from src.agent.numbers import identifier_digits
from src.config import PROJECT_ROOT

CITY_ALIASES = {
    "алматы": "Almaty",
    "алмата": "Almaty",
    "астана": "Astana",
    "шымкент": "Shymkent",
    "шимкент": "Shymkent",
    "караганда": "Karaganda",
    "қарағанды": "Karaganda",
    "актобе": "Aktobe",
    "ақтөбе": "Aktobe",
    "атырау": "Atyrau",
    "павлодар": "Pavlodar",
    "өскемен": "Oskemen",
    "усть-каменогорск": "Oskemen",
}
SPECIALTIES = {
    "лор": "ENT",
    "отоларинголог": "ENT",
    "ent": "ENT",
    "терапевт": "therapist",
    "гинеколог": "gynecologist",
    "кардиолог": "cardiologist",
    "стоматолог": "dentist",
    "педиатр": "pediatrician",
    "узи": "ultrasound",
    "анализы": "lab",
}


class Catalog:
    """Каталог сценариев и полей с проверкой значений по исходным данным."""

    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = Path(root)
        self.raw = self.read("scenarios")
        self.scenarios = {s["scenario_id"]: s for s in self.raw["scenarios"]}
        self.system = {s["id"]: s for s in self.raw["system_intents"]}
        self.slots = {s["name"]: s for s in self.read("slots")["slots"]}
        self.actions = {a["name"]: a for a in self.read("actions")["actions"]}
        self.today = date.fromisoformat(self.raw["meta"]["as_of_date"])

    def read(self, name: str) -> dict:
        path = self.root / "data" / f"{name}.json"
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def routing_catalog(self) -> list[dict]:
        return list(self.scenarios.values()) + [
            {
                "scenario_id": s["id"],
                "name": s["id"],
                "description": s["description"],
                "not_this_if": [],
            }
            for s in self.system.values()
        ]

    def demo_help(self) -> str:
        """Только явно учебные профили из поставляемого датасета, без авторизации."""
        clients = {c["client_id"]: c for c in self.read("mock_backend")["clients"]}
        examples = (
            ("C002", "ДМС: запись к врачу / дәрігерге жазылу"),
            ("C001", "ОГПО и КАСКО / ОГПО және КАСКО"),
            ("C003", "Просроченный ОГПО / мерзімі өткен ОГПО"),
        )
        lines = ["Учебные клиенты / Оқу клиенттері:"]
        lines.extend(f"{label}: {clients[cid]['phone']}" for cid, label in examples)
        lines.append(
            "Отправьте запрос, а затем выбранный номер при вопросе о телефоне. "
            "Ваш личный номер может отсутствовать в учебной базе. /reset — новый разговор."
        )
        return "\n".join(lines)

    def normalize(self, name: str, value):
        """Модель извлекает значения; окончательная проверка остаётся в Python."""
        spec = self.slots[name]
        if isinstance(value, str):
            value = value.strip()
        if name in {"phone", "iin"} and isinstance(value, str):
            value = identifier_digits(value) or value
        if name == "phone" and isinstance(value, str):
            digits = re.sub(r"[\s()+\-–—]", "", value)
            if (
                len(digits) == 10
                and digits.isascii()
                and digits.isdigit()
                and not value.startswith("+")
            ):
                value = "+7" + digits
            if len(digits) == 11 and digits[0] in "78":
                value = "+7" + digits[1:]
        if name in {
            "policy_number",
            "claim_number",
            "vehicle_plate",
            "culprit_vehicle_plate",
        } and isinstance(value, str):
            value = value.upper().replace(" ", "")
        if name == "city" and isinstance(value, str):
            value = CITY_ALIASES.get(value.casefold(), value)
        if name == "doctor_specialty" and isinstance(value, str):
            value = SPECIALTIES.get(value.casefold(), value)
        if spec["type"] == "date":
            offsets = {
                "сегодня": 0,
                "бүгін": 0,
                "завтра": 1,
                "ертең": 1,
                "вчера": -1,
                "кеше": -1,
                "послезавтра": 2,
                "бүрсігүні": 2,
            }
            if isinstance(value, str) and value.casefold() in offsets:
                value = (self.today + timedelta(days=offsets[value.casefold()])).isoformat()
            if not isinstance(value, str):
                raise ValueError(name)
            value = date.fromisoformat(value).isoformat()
        elif spec["type"] == "integer":
            if isinstance(value, bool) or not str(value).isdigit():
                raise ValueError(name)
            value = int(value)
            if value <= 0:
                raise ValueError(name)
        elif spec["type"] == "boolean":
            if type(value) is not bool:
                raise ValueError(name)
        elif spec["type"] == "list":
            if not isinstance(value, list) or not value:
                raise ValueError(name)
            if any(not isinstance(v, str) or not re.fullmatch(spec["pattern"], v) for v in value):
                raise ValueError(name)
        elif spec["type"] == "enum":
            for option in spec["values"]:
                if str(option).casefold() == str(value).casefold():
                    value = option
                    break
            if value not in spec["values"] or isinstance(value, bool):
                raise ValueError(name)
        elif not isinstance(value, str) or not value or len(value) > 2000:
            raise ValueError(name)
        if spec.get("pattern") and spec["type"] != "list":
            if not re.fullmatch(spec["pattern"], str(value)):
                raise ValueError(name)
        return value


def mask_private(value):
    """Маскировать телефон, ИИН и почту при показе трассировки и ответов."""
    if isinstance(value, dict):
        return {key: mask_private(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_private(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?<!\d)\d{12}(?!\d)", lambda m: "********" + m[0][-4:], value)
        # В исходной реплике номер может содержать скобки, пробелы и дефисы.
        value = re.sub(
            r"(?<![\w+])(?:\+7|[78])(?:[ ()\-–—]*\d){10}(?!\d)",
            lambda m: "+7*******" + re.sub(r"\D", "", m[0])[-3:],
            value,
        )
        value = re.sub(
            r"(?<![\w+])7(?:[ ()\-–—]*\d){9}(?!\d)",
            lambda m: "*******" + re.sub(r"\D", "", m[0])[-3:],
            value,
        )
        value = re.sub(r"([\w.+-])[^@\s]*@([^\s]+)", r"\1***@\2", value)
    return value
