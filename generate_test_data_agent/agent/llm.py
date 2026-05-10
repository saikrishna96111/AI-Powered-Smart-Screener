"""LLM provider switch (shared shape with rule_agent/agent/llm.py).

Pick GPT or Claude with `LLM_PROVIDER` in `.env`:

    LLM_PROVIDER=gpt          # default — uses OPENAI_API_KEY + OPENAI_MODEL
    LLM_PROVIDER=claude       # uses ANTHROPIC_API_KEY + CLAUDE_MODEL
"""

import os

from dotenv import load_dotenv, find_dotenv

_here = os.path.dirname(__file__)
_agent_root = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))

load_dotenv(find_dotenv())
load_dotenv(os.path.join(_agent_root, ".env"))
load_dotenv(os.path.join(_repo_root, "rule_agent", ".env"))

_AGENT_DEFAULT_TEMPERATURE = 0.35
_llm = None


def _selected_provider() -> str:
    raw = (os.getenv("LLM_PROVIDER") or "gpt").strip().lower()
    if raw in {"claude", "anthropic"}:
        return "claude"
    return "gpt"


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    provider = _selected_provider()
    explicit_temp = os.getenv("LLM_TEMPERATURE")
    temperature = (
        float(explicit_temp) if explicit_temp else _AGENT_DEFAULT_TEMPERATURE
    )

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
        # Claude Opus 4.7+ rejects `temperature` ("deprecated"). Only forward when
        # explicitly set so older Claude models that still accept it keep working.
        if explicit_temp is not None:
            kwargs["temperature"] = temperature
        _llm = ChatAnthropic(**kwargs)
        return _llm

    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "LLM_PROVIDER=gpt but OPENAI_API_KEY is not set in .env"
        )

    _llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        temperature=temperature,
    )
    return _llm
