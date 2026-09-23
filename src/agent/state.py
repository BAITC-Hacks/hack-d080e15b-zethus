"""Состояние одного разговора; операции подтверждаются только внутри него."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    scenario_id: str
    slots: dict[str, Any] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)


@dataclass
class PendingAction:
    name: str
    arguments: dict[str, Any]
    preview: dict[str, Any]


@dataclass
class DialogState:
    language: str = "ru"
    client_id: str | None = None
    active: Task | None = None
    queue: list[Task] = field(default_factory=list)
    suspended: list[Task] = field(default_factory=list)
    expected_slot: str | None = None
    pending: PendingAction | None = None
    awaiting_resume: bool = False
    operator_handoff: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    unclear_count: int = 0
    turn: int = 0

    def remember(self, text: str, reply: str) -> None:
        self.history.extend([{"role": "user", "content": text},
                             {"role": "assistant", "content": reply}])
        self.history = self.history[-12:]


@dataclass
class TurnResult:
    text: str
    trace: dict[str, Any]
