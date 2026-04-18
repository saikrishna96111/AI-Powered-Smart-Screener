import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

_here = os.path.dirname(__file__)
_agent_root = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))

load_dotenv(find_dotenv())
load_dotenv(os.path.join(_agent_root, ".env"))
load_dotenv(os.path.join(_repo_root, "rule_agent", ".env"))

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
            temperature=0.1,
        )
    return _llm
