"""Use cases for LLM runtime decision and post-processing wiring."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from ptarmigan_flow.launchd import LAUNCH_AGENT_LABEL, LAUNCHD_LLM_ENABLED_ENV
from ptarmigan_flow.text_processing.interfaces import ChainedTextPostProcessor, TextPostProcessor
from ptarmigan_flow.text_processing.llm import (
    LLMClientError,
    LLMCorrectionSettings,
    LLMPostProcessor,
)

LOGGER = logging.getLogger(__name__)


def normalize_optional_secret(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_bool_token(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def launchd_llm_enabled_override_from_payload(payload: object) -> bool | None:
    if not isinstance(payload, dict):
        return None
    environment = payload.get("EnvironmentVariables")
    if not isinstance(environment, dict):
        return None
    return parse_bool_token(environment.get(LAUNCHD_LLM_ENABLED_ENV))


def launchd_llm_enabled_override_from_env() -> bool | None:
    if LAUNCHD_LLM_ENABLED_ENV not in os.environ:
        return None
    raw = os.environ.get(LAUNCHD_LLM_ENABLED_ENV)
    parsed = parse_bool_token(raw)
    if parsed is None:
        LOGGER.warning(
            "Ignoring invalid %s=%r (expected 1/0/true/false)",
            LAUNCHD_LLM_ENABLED_ENV,
            raw,
        )
    return parsed


def runtime_language_from_config(config: object) -> str:
    language = getattr(config, "language", "en")
    if not isinstance(language, str):
        return "en"
    normalized = language.strip()
    if not normalized or normalized.lower() == "auto":
        return "en"
    return normalized


def build_llm_settings_from_config(
    config: object,
    llm_cfg: object,
) -> LLMCorrectionSettings | None:
    provider = str(getattr(llm_cfg, "provider", "")).strip().lower()
    base_url = str(getattr(llm_cfg, "base_url", "")).strip()
    model = str(getattr(llm_cfg, "model", "")).strip()
    if not base_url or not model:
        return None
    return LLMCorrectionSettings(
        provider=provider,
        base_url=base_url,
        model=model,
        timeout_seconds=float(getattr(llm_cfg, "timeout_seconds", 5.0)),
        max_input_chars=int(getattr(llm_cfg, "max_input_chars", 500)),
        api_key=normalize_optional_secret(getattr(llm_cfg, "api_key", None)),
        enabled_tools=bool(getattr(llm_cfg, "enabled_tools", False)),
        language=runtime_language_from_config(config),
        glossary=[
            str(term).strip()
            for term in (getattr(llm_cfg, "glossary", None) or [])
            if str(term).strip()
        ],
        think=normalize_optional_secret(getattr(llm_cfg, "think", None)),
    )


def should_enable_llm_correction_for_this_run(
    llm_cfg: object,
    *,
    is_interactive_session: Callable[[], bool],
    prompt_llm_for_this_run: Callable[[], bool],
) -> bool:
    in_launchd_context = os.environ.get("XPC_SERVICE_NAME") == LAUNCH_AGENT_LABEL
    if in_launchd_context:
        launchd_override = launchd_llm_enabled_override_from_env()
        if launchd_override is not None:
            return launchd_override

    mode = str(getattr(llm_cfg, "mode", "never")).strip().lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "ask":
        if not is_interactive_session():
            LOGGER.info(
                "LLM correction mode=ask in non-interactive session; "
                "disabling LLM correction for this run"
            )
            return False
        return prompt_llm_for_this_run()
    LOGGER.warning("Unknown LLM correction mode '%s'; disabling LLM correction", mode)
    return False


def llm_enabled_for_this_run(
    config: object,
    *,
    is_interactive_session: Callable[[], bool],
    prompt_llm_for_this_run: Callable[[], bool],
) -> bool:
    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    if llm_cfg is None:
        return False
    return should_enable_llm_correction_for_this_run(
        llm_cfg,
        is_interactive_session=is_interactive_session,
        prompt_llm_for_this_run=prompt_llm_for_this_run,
    )


def build_runtime_post_processor(
    config: object,
    *,
    base_processor: TextPostProcessor,
    llm_enabled_override: bool | None = None,
    is_interactive_session: Callable[[], bool],
    prompt_llm_for_this_run: Callable[[], bool],
    llm_processor_factory: Callable[[LLMCorrectionSettings], TextPostProcessor] | None = None,
) -> TextPostProcessor:
    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    llm_enabled = llm_enabled_override
    if llm_enabled is None:
        llm_enabled = llm_cfg is not None and should_enable_llm_correction_for_this_run(
            llm_cfg,
            is_interactive_session=is_interactive_session,
            prompt_llm_for_this_run=prompt_llm_for_this_run,
        )
    if llm_cfg is None or not llm_enabled:
        return base_processor

    settings = build_llm_settings_from_config(config, llm_cfg)
    if settings is None:
        LOGGER.warning(
            "LLM correction is enabled but base_url/model is missing; "
            "continuing without LLM correction"
        )
        return base_processor

    factory = llm_processor_factory or LLMPostProcessor
    try:
        llm_processor = factory(settings)
    except Exception as exc:
        LOGGER.warning("Failed to initialize LLM correction; continuing without it (%s)", exc)
        return base_processor

    try:
        llm_processor.preflight()
    except LLMClientError as exc:
        LOGGER.warning("LLM correction preflight warning: %s", exc)
    except Exception as exc:
        LOGGER.warning("Unexpected LLM preflight failure: %s", exc)

    LOGGER.info(
        "LLM correction enabled: provider=%s base_url=%s model=%s timeout=%.2fs max_input_chars=%d",
        settings.provider,
        settings.base_url,
        settings.model,
        settings.timeout_seconds,
        settings.max_input_chars,
    )
    return ChainedTextPostProcessor([base_processor, llm_processor])


def normalize_llm_mode(mode: object) -> str:
    return str(getattr(mode, "value", mode)).strip().lower()


class LLMAskSessionMemory:
    """Remember the GUI app's answer to ``mode = "ask"`` for one app session.

    The app asks once (with a modal) and reuses the answer for later daemon
    restarts. Seeing any mode other than ``ask`` forgets the answer, so
    switching away from ``ask`` and back asks again.
    """

    def __init__(self) -> None:
        self._answer: bool | None = None

    @property
    def answer(self) -> bool | None:
        return self._answer

    def observe_mode(self, mode: object) -> None:
        if normalize_llm_mode(mode) != "ask":
            self._answer = None

    def resolve(self, mode: object, prompt: Callable[[], bool]) -> bool | None:
        """Return the LLM override for this daemon start.

        ``None`` means "no override" (``always``/``never`` decide as usual).
        In ``ask`` mode the remembered answer is reused, otherwise ``prompt``
        is called once and its answer remembered.
        """
        self.observe_mode(mode)
        if normalize_llm_mode(mode) != "ask":
            return None
        if self._answer is None:
            self._answer = bool(prompt())
        return self._answer
