"""LLM provider switch.

Pick GPT or Claude with `LLM_PROVIDER` in `.env`:

    LLM_PROVIDER=gpt          # default — uses OPENAI_API_KEY + OPENAI_MODEL
    LLM_PROVIDER=claude       # uses ANTHROPIC_API_KEY + CLAUDE_MODEL

Other env knobs:
    OPENAI_API_KEY=...        (required when LLM_PROVIDER=gpt)
    OPENAI_MODEL=gpt-5.4      (optional override)
    ANTHROPIC_API_KEY=...     (required when LLM_PROVIDER=claude)
    CLAUDE_MODEL=claude-opus-4-7   (optional override)
    LLM_TEMPERATURE=0.2       (optional override; rule_agent default 0.2)
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _selected_provider() -> str:
    raw = (os.getenv("LLM_PROVIDER") or "gpt").strip().lower()
    if raw in {"claude", "anthropic"}:
        return "claude"
    return "gpt"


def _make_llm(default_temperature: float = 0.2):
    """Build the configured chat model. Imports the SDK lazily so unused providers
    don't have to be installed.

    Note on temperature:
    - GPT: pass through as before.
    - Claude Opus 4.7 and newer: the API rejects `temperature` ("deprecated for
      this model"). We omit it by default and only forward it when the user
      explicitly opts in with LLM_TEMPERATURE for older Claude models that still
      accept it.
    """
    provider = _selected_provider()
    explicit_temp = os.getenv("LLM_TEMPERATURE")
    temperature = float(explicit_temp) if explicit_temp else default_temperature

    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER=claude but langchain-anthropic is not installed. "
                "Add `langchain-anthropic` to requirements.txt and re-install."
            ) from e

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set in .env"
            )

        kwargs = {"model": os.getenv("CLAUDE_MODEL", "claude-opus-4-7")}
        # Only forward temperature when the user explicitly set LLM_TEMPERATURE,
        # so models that deprecate it (Opus 4.7+) work out of the box.
        if explicit_temp is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)

    # Default: OpenAI/GPT
    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "LLM_PROVIDER=gpt but OPENAI_API_KEY is not set in .env"
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        temperature=temperature,
    )


llm = _make_llm()
