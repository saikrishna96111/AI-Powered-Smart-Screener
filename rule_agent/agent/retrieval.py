"""Read-side retrieval for the rule-agent.

The CDS examples + guidelines + SAP Help portal pages are ingested by the
error_handling agent's pipeline (``error_handling_agent/scripts/build_index.py``)
into a persistent Chroma store. This module is the **read** path used by the
rule-agent during CDS generation: it opens the same Chroma directory, runs
similarity search, and formats the top hits as a prompt-ready block.

**Guidelines are always injected** from ``cds_guardrails.txt`` on disk (full
file) whenever the rule agent retrieves context, so guardrails are present
even if the vector index is stale or similarity search ranks them low.

Path resolution (in order of precedence):
  1. ``$AISS_VECTOR_STORE_DIR`` environment variable
  2. ``<repo>/error_handling_agent/vector_store/cds_docs``

The module degrades gracefully: if the store is missing (e.g. the user
hasn't run ``build_index.py`` yet) or chromadb / sentence-transformers
isn't installed, ``retrieve_reference_examples`` still returns guidelines
from disk when available.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REPO_DEFAULT_STORE = (
    _THIS_DIR.parent.parent / "error_handling_agent" / "vector_store" / "cds_docs"
)
_GUIDELINES_PATH = (
    _THIS_DIR.parent.parent
    / "error_handling_agent"
    / "data"
    / "sources"
    / "guidelines"
    / "cds_guardrails.txt"
)

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION_NAME = "cds_docs"


def get_persist_dir() -> Path:
    env = os.environ.get("AISS_VECTOR_STORE_DIR")
    return Path(env).expanduser().resolve() if env else _REPO_DEFAULT_STORE


@lru_cache(maxsize=1)
def _get_vector_store():
    """Open the shared Chroma store (read-only intent). Returns None on failure."""
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        return None

    persist_dir = get_persist_dir()
    if not persist_dir.exists():
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=_COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def _safe_collection_size() -> int:
    vs = _get_vector_store()
    if vs is None:
        return 0
    try:
        return vs._collection.count()
    except Exception:
        return 0


def load_cds_guardrails_text() -> str:
    """Return the full CDS guardrails document (always read from disk for rule-agent)."""
    try:
        if _GUIDELINES_PATH.is_file():
            return _GUIDELINES_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


def _format_retrieval_block(docs: list[Any], *, guidelines_text: str) -> str:
    """Build the prompt block: mandatory guardrails first, then examples + excerpts."""
    parts: list[str] = []

    if guidelines_text:
        parts.append("=== CDS Guardrails (S/4HANA 2025 — mandatory) ===")
        parts.append(guidelines_text)
        parts.append("")

    example_block = _format_example_block(docs)
    if example_block and not example_block.startswith("(no reference"):
        parts.append("=== Reference CDS examples and supporting excerpts ===")
        parts.append(example_block)
    elif not guidelines_text:
        parts.append(example_block)

    return "\n".join(parts).strip() or "(no reference material available)"


def _format_example_block(docs: list[Any]) -> str:
    """Render retrieved Documents into a single prompt-ready text block.

    Reference examples are shown FIRST and in full (because they're the gold
    template the LLM should mirror), other doc-type hits follow as compact
    excerpts.
    """
    if not docs:
        return "(no reference examples available — falling back to model knowledge only)"

    examples: list = []
    others: list = []
    for d in docs:
        st = (d.metadata or {}).get("source_type", "")
        if st == "example":
            examples.append(d)
        else:
            others.append(d)

    parts: list[str] = []

    for i, d in enumerate(examples, start=1):
        meta = d.metadata or {}
        name = meta.get("source_name") or meta.get("source") or f"example_{i}"
        parts.append(f"--- Example {i}: {name} (source_type=example) ---")
        parts.append((d.page_content or "").strip())
        parts.append("")

    for i, d in enumerate(others, start=1):
        meta = d.metadata or {}
        name = meta.get("source_name") or meta.get("source") or f"doc_{i}"
        st = meta.get("source_type", "?")
        snippet = (d.page_content or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200].rstrip() + " …"
        parts.append(f"--- Supporting excerpt {i}: {name} (source_type={st}) ---")
        parts.append(snippet)
        parts.append("")

    return "\n".join(parts).strip()


def retrieve_reference_examples(
    query: str,
    *,
    k_examples: int = 2,
    k_other: int = 2,
    k_guidelines: int = 2,
) -> tuple[list[Any], str]:
    """Pull guardrails (always) + gold examples + supporting excerpts.

    Returns ``(docs, prompt_text)``. ``docs`` is the raw list of LangChain
    Document objects (useful if a caller wants its own formatting);
    ``prompt_text`` is the ready-to-inject block for the cds.yaml template.

    Strategy:
      0. Always prepend the full ``cds_guardrails.txt`` from disk.
      1. Filter Chroma for ``source_type == "example"`` and take the
         top-``k_examples`` (full gold CDS views).
      2. Filter for ``source_type == "guideline"`` (RAG index supplement).
      3. Filter for OTHER source_types (web, pdf) as supporting evidence.
      4. Concatenate and format.
    """
    guidelines_text = load_cds_guardrails_text()
    vs = _get_vector_store()
    if vs is None:
        return [], _format_retrieval_block([], guidelines_text=guidelines_text)

    query = (query or "").strip() or "ABAP CDS view template S/4HANA 2025 guardrails"

    docs: list = []

    # Step 1 — gold examples (Chroma metadata filter).
    try:
        ex = vs.similarity_search(
            query,
            k=max(1, k_examples),
            filter={"source_type": "example"},
        )
        docs.extend(ex)
    except Exception:
        pass

    # Step 2 — guideline chunks from the vector index (supplement disk file).
    if k_guidelines > 0:
        try:
            gl = vs.similarity_search(
                query,
                k=max(1, k_guidelines),
                filter={"source_type": "guideline"},
            )
            for d in gl:
                if d not in docs:
                    docs.append(d)
        except Exception:
            pass

    # Step 3 — other supporting context (web, pdf, …).
    if k_other > 0:
        try:
            other = vs.similarity_search(
                query,
                k=max(1, k_other),
                filter={"source_type": {"$nin": ["example", "guideline"]}},
            )
            docs.extend(other)
        except Exception:
            pass

    return docs, _format_retrieval_block(docs, guidelines_text=guidelines_text)


def index_status() -> dict:
    """Lightweight status dict for diagnostic logging."""
    persist_dir = get_persist_dir()
    return {
        "persist_dir": str(persist_dir),
        "exists": persist_dir.exists(),
        "chunks": _safe_collection_size(),
    }
