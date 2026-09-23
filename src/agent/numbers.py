"""Цифры и группы числительных RU/KK для ожидаемого телефона или ИИН."""

import re


UNITS = {
    "ноль": 0, "нуль": 0, "нөл": 0, "нол": 0, "один": 1, "одна": 1, "бір": 1,
    "два": 2, "две": 2, "екі": 2, "три": 3, "үш": 3, "четыре": 4, "төрт": 4,
    "пять": 5, "бес": 5, "шесть": 6, "алты": 6, "семь": 7, "жеті": 7,
    "восемь": 8, "сегіз": 8, "девять": 9, "тоғыз": 9,
}
TENS = {
    "десять": 10, "он": 10, "двадцать": 20, "жиырма": 20, "тридцать": 30, "отыз": 30,
    "сорок": 40, "қырық": 40, "пятьдесят": 50, "елу": 50, "шестьдесят": 60, "алпыс": 60,
    "семьдесят": 70, "жетпіс": 70, "восемьдесят": 80, "сексен": 80, "девяносто": 90, "тоқсан": 90,
}
TEENS = dict(zip(("одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                  "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"), range(11, 20)))
HUNDREDS = dict(zip(("сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
                     "семьсот", "восемьсот", "девятьсот"), range(100, 1000, 100)))


def identifier_digits(text: str) -> str | None:
    """Сохранить цифры и нули; вернуть None при словах вне числовой записи.

    Это разбор формата поля, а не классификатор намерений. Не добавляет недостающие
    цифры. Запятые/паузы разделяют группы: «семь, семьсот один, ноль, ноль...».
    """
    if not isinstance(text, str) or not text.strip():
        return None
    tokens = re.findall(r"\d+|[^\W\d_]+|[^\w\s]", text.casefold())
    parts, prefix, current, kind = [], "", None, None

    def flush():
        nonlocal current, kind
        if current is not None:
            parts.append(str(current))
        current, kind = None, None

    for token in tokens:
        if token in {"+", "плюс"}:
            if prefix or parts or current is not None:
                return None
            prefix = "+"
        elif token in {",", ".", ";", "-", "–", "—", "(", ")", "!", ":", "«", "»", '"'}:
            flush()
        elif token.isascii() and token.isdigit():
            flush()
            parts.append(token)
        elif token in HUNDREDS:
            flush()
            current, kind = HUNDREDS[token], "hundred"
        elif token == "жүз":
            if current is None:
                current = 100
            elif kind == "unit" and 1 <= current <= 9:
                current *= 100
            else:
                return None
            kind = "hundred"
        elif token in TENS or token in TEENS:
            value = TENS.get(token, TEENS.get(token))
            if kind != "hundred":
                flush()
            current = (current or 0) + value
            kind = "tens" if token in TENS and token != "десять" else "unit"
        elif token in UNITS:
            value = UNITS[token]
            if value == 0:
                flush()
                parts.append("0")
            else:
                if kind not in {"hundred", "tens"}:
                    flush()
                current = (current or 0) + value
                kind = "unit"
        else:
            return None
    flush()
    return prefix + "".join(parts) if parts else None
