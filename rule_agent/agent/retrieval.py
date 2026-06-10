"""Read-side retrieval for the rule-agent.

The CDS examples + guidelines + SAP Help portal pages are ingested by the
error-handling agent's pipeline (``error_handling_agent/scripts/build_index.py``)
into a persistent Chroma store. This module is the **read** path used by the
rule-agent during CDS generation: it opens the same Chroma directory, runs
similarity search, and formats the top hits as a prompt-ready block.

Path resolution (in order of precedence):
  1. ``$AISS_VECTOR_STORE_DIR`` environment variable
  2. ``<repo>/error_handling_agent/vector_store/cds_docs``

The module degrades gracefully: if the store is missing (e.g. the user
hasn't run ``build_index.py`` yet) or chromadb / sentence-transformers
isn't installed, ``retrieve_reference_examples`` returns an empty list.
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
) -> tuple[list[Any], str]:
    """Pull the most relevant gold examples (+ a few supporting excerpts).

    Returns ``(docs, prompt_text)``. ``docs`` is the raw list of LangChain
    Document objects (useful if a caller wants its own formatting);
    ``prompt_text`` is the ready-to-inject block for the cds.yaml template.

    Strategy:
      1. Filter Chroma for ``source_type == "example"`` and take the
         top-``k_examples`` (full gold CDS views).
      2. Filter for the OTHER source_types (guideline, web, pdf) and take
         the top-``k_other`` chunks as supporting evidence.
      3. Concatenate and format.
    """
    vs = _get_vector_store()
    if vs is None:
        return [], _format_example_block([])

    query = (query or "").strip() or "ABAP CDS view template"

    docs: list = []
    # Step 1 — gold examples first (Chroma metadata filter).
    try:
        ex = vs.similarity_search(
            query,
            k=max(1, k_examples),
            filter={"source_type": "example"},
        )
        docs.extend(ex)
    except Exception:
        ex = []

    # Step 2 — supporting context (anything not "example").
    if k_other > 0:
        try:
            other = vs.similarity_search(
                query,
                k=max(1, k_other),
                filter={"source_type": {"$ne": "example"}},
            )
            docs.extend(other)
        except Exception:
            pass

    return docs, _format_example_block(docs)


def index_status() -> dict:
    """Lightweight status dict for diagnostic logging."""
    persist_dir = get_persist_dir()
    return {
        "persist_dir": str(persist_dir),
        "exists": persist_dir.exists(),
        "chunks": _safe_collection_size(),
    }
