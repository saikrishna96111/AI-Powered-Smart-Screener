"""Persistent Chroma vector store wrapped behind a tiny module-level API.

- Embeddings: ``sentence-transformers/all-MiniLM-L6-v2`` (local, 384-dim, CPU friendly).
- Persistence directory: ``<error_handling_agent>/vector_store/cds_docs/``.

The first call to :func:`get_embeddings` downloads the model into the local
HuggingFace cache (~90 MB) and reuses it forever after. The vector store
object is cached so repeated calls in the same Python process don't reopen
Chroma's on-disk segment.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default location of the persistent Chroma collection (kept inside the agent
# folder so it ships with the project and is easy to delete / rebuild).
_AGENT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_PERSIST_DIR = _AGENT_DIR / "vector_store" / "cds_docs"
DEFAULT_COLLECTION = "cds_docs"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_store = None


def _resolved_persist_dir(persist_dir: str | Path | None) -> Path:
    pd = Path(persist_dir or os.getenv("RAG_PERSIST_DIR") or DEFAULT_PERSIST_DIR)
    pd.mkdir(parents=True, exist_ok=True)
    return pd


def get_embeddings():
    """Return a cached :class:`HuggingFaceEmbeddings` instance (MiniLM-L6-v2)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    # Local import keeps the agent importable when the RAG extras aren't installed
    # (e.g. someone running just ``run.py`` without the new packages yet).
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embeddings


def get_vector_store(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
):
    """Open / create the persistent Chroma collection (cached per process)."""
    global _store
    if _store is not None:
        return _store

    from langchain_chroma import Chroma

    pd = _resolved_persist_dir(persist_dir)
    name = collection_name or os.getenv("RAG_COLLECTION", DEFAULT_COLLECTION)

    _store = Chroma(
        collection_name=name,
        embedding_function=get_embeddings(),
        persist_directory=str(pd),
    )
    return _store


def reset_cache() -> None:
    """Force the next call to rebuild the in-process Chroma / embeddings handles.

    Useful right after :func:`build_index` so the same Python process picks up
    the freshly written collection segments.
    """
    global _store, _embeddings
    _store = None
    _embeddings = None


def collection_size() -> int:
    """Best-effort row count of the underlying Chroma collection (0 if empty)."""
    store = get_vector_store()
    try:
        col = store._collection  # langchain-chroma exposes the raw chromadb collection
        return int(col.count())
    except Exception:
        return 0


def get_retriever(k: int = 6):
    """Return a basic similarity retriever fetching the top-``k`` chunks."""
    return get_vector_store().as_retriever(search_kwargs={"k": int(k)})
