"""LangGraph node that grounds the fix prompt with retrieved doc excerpts.

The node:

1. Builds a focused retrieval query from (error_text + CDS lines near the
   line / column the error mentions, or the first 30 lines if the error has
   no positional hint).
2. Pulls the top-K chunks from the persistent Chroma collection.
3. Stores them on the agent state under ``references`` (structured, useful
   for citations) and ``references_text`` (pre-rendered block ready to drop
   into the fix prompt).

Failures (e.g. nobody has run ``build_index.py`` yet) are non-fatal: the
node writes an empty references block and the fix prompt falls back to its
ungrounded behaviour with a visible "(no reference excerpts retrieved)" note.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .store import collection_size, get_retriever


_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "6"))


def _build_query(state: dict) -> str:
    err = (state.get("error_text") or "").strip()
    cds = (state.get("cds_source") or "").strip()
    cds_lines = cds.splitlines()

    line_nums: list[int] = []
    for raw in re.findall(r"\b(\d{1,5})\b", err):
        try:
            ln = int(raw)
        except ValueError:
            continue
        if 1 <= ln <= len(cds_lines):
            line_nums.append(ln)

    snippet_lines: list[str] = []
    if line_nums and cds_lines:
        seen = set()
        for ln in line_nums[:3]:
            for j in range(max(0, ln - 4), min(len(cds_lines), ln + 4)):
                if j in seen:
                    continue
                seen.add(j)
                snippet_lines.append(cds_lines[j])
    else:
        snippet_lines = cds_lines[:30]

    snippet = "\n".join(snippet_lines).strip()
    return (
        "Find ABAP CDS documentation chunks that explain the rule(s) violated "
        "by the following error and snippet.\n\n"
        f"ERROR:\n{err}\n\n"
        f"CDS SNIPPET:\n{snippet}"
    )


def _ref_id(meta: dict) -> str:
    slug = (meta or {}).get("ref_slug") or ""
    if not slug:
        src = (meta or {}).get("source_name") or (meta or {}).get("source") or "doc"
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(src).split("/")[-1].split("\\")[-1])
    page = (meta or {}).get("page")
    chunk = (meta or {}).get("chunk_index", 0)
    suffix = f"p{page}" if page is not None else f"#{chunk}"
    return f"{slug}::{suffix}"


def _format_reference_block(ref: dict, max_chars: int = 1200) -> str:
    text = ref.get("text") or ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    src = ref.get("source") or "(unknown source)"
    return f"### [{ref['ref_id']}] {ref.get('source_type', '?')} — {src}\n{text}"


def retrieve_docs_node(state: dict) -> dict:
    cds = (state.get("cds_source") or "").strip()
    err = (state.get("error_text") or "").strip()
    if not cds or not err:
        return {"references": [], "references_text": ""}

    try:
        # Cheap sanity check first — if the index hasn't been built yet,
        # emit a helpful hint instead of failing silently.
        if collection_size() == 0:
            hint = (
                "(RAG index is empty — run "
                "`python scripts/build_index.py` once to populate it.)"
            )
            return {"references": [], "references_text": hint}

        retriever = get_retriever(k=_DEFAULT_TOP_K)
        results = retriever.invoke(_build_query(state))
    except Exception as exc:  # vector store unavailable, model failed to load, ...
        return {
            "references": [],
            "references_text": f"(Reference retrieval unavailable: {type(exc).__name__}: {exc})",
        }

    refs: list[dict] = []
    seen_ids: set[str] = set()
    for d in results or []:
        meta = dict(d.metadata or {})
        rid = _ref_id(meta)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        refs.append(
            {
                "ref_id": rid,
                "source": meta.get("source", ""),
                "source_type": meta.get("source_type", ""),
                "page": meta.get("page"),
                "text": d.page_content or "",
            }
        )

    references_text = "\n\n".join(_format_reference_block(r) for r in refs)
    return {"references": refs, "references_text": references_text}
