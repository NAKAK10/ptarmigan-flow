from __future__ import annotations

from ptarmigan_flow.application.use_cases.llm_runtime import LLMAskSessionMemory
from ptarmigan_flow.config import LLMCorrectionMode


def _prompt(answers: list[bool], calls: list[int]):
    def prompt() -> bool:
        calls.append(1)
        return answers.pop(0)

    return prompt


def test_ask_prompts_once_and_reuses_answer() -> None:
    memory = LLMAskSessionMemory()
    calls: list[int] = []
    prompt = _prompt([True], calls)

    assert memory.resolve("ask", prompt) is True
    assert memory.resolve("ask", prompt) is True
    assert len(calls) == 1


def test_ask_remembers_dont_use_answer() -> None:
    memory = LLMAskSessionMemory()
    calls: list[int] = []
    prompt = _prompt([False], calls)

    assert memory.resolve(LLMCorrectionMode.ASK, prompt) is False
    assert memory.resolve(LLMCorrectionMode.ASK, prompt) is False
    assert len(calls) == 1


def test_always_and_never_return_no_override_without_prompting() -> None:
    memory = LLMAskSessionMemory()
    calls: list[int] = []
    prompt = _prompt([], calls)

    assert memory.resolve("always", prompt) is None
    assert memory.resolve(LLMCorrectionMode.NEVER, prompt) is None
    assert calls == []


def test_leaving_ask_mode_forgets_the_answer() -> None:
    memory = LLMAskSessionMemory()
    calls: list[int] = []
    prompt = _prompt([True, False], calls)

    assert memory.resolve("ask", prompt) is True
    memory.observe_mode("always")
    assert memory.answer is None
    assert memory.resolve(" ASK ", prompt) is False
    assert len(calls) == 2


def test_observing_ask_keeps_the_answer() -> None:
    memory = LLMAskSessionMemory()
    memory.resolve("ask", lambda: True)

    memory.observe_mode(LLMCorrectionMode.ASK)

    assert memory.answer is True
