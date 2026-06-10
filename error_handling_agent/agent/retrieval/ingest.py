"""Ingestion pipeline.

Pulls source documents (PDFs + crawled SAP Help pages), splits them into
overlapping chunks tuned for CDS / ABAP examples, embeds them with MiniLM,
and upserts into the persistent Chroma collection.

Run via ``python scripts/build_index.py``. Idempotent: chunks have stable
SHA-1-derived IDs so re-running only re-embeds content that actually changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .store import (
    DEFAULT_PERSIST_DIR,
    get_vector_store,
    reset_cache,
)

_AGENT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOURCES_DIR = _AGENT_DIR / "data" / "sources"
DEFAULT_EXAMPLES_DIR = _AGENT_DIR / "data" / "sources" / "examples"
DEFAULT_GUIDELINES_DIR = _AGENT_DIR / "data" / "sources" / "guidelines"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "crawler_config.yaml"
DEFAULT_MANIFEST_PATH = _AGENT_DIR / "data" / "INDEX_MANIFEST.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha1_id(source: str, idx: int, content_head: str) -> str:
    payload = f"{source}|{idx}|{content_head[:200]}".encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:16]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).stem)[:48].strip("_") or "doc"
    return f"{stem}-{idx}-{h}"


def _slug_from_url_or_path(value: str) -> str:
    tail = value.replace("\\", "/").rstrip("/").split("/")[-1] or "doc"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tail)[:80].strip("_") or "doc"


def _clean_html(html: str) -> str:
    """Strip nav / chrome and return readable text for the chunker."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for sel in (
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "form",
        "noscript",
        "aside",
    ):
        for el in soup.select(sel):
            el.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_text_files(
    directory: Path,
    *,
    source_type: str,
    glob_patterns: tuple[str, ...] = ("*.txt", "*.cds", "*.ddls", "*.md"),
) -> list:
    """Load every text file in ``directory`` as ONE Document per file.

    Used for the curated `examples/` and `guidelines/` folders so each gold
    CDS view / rule sheet is retrievable as a coherent unit (the chunker can
    still split them further, but with code-aware separators they stay
    mostly intact).
    """
    from langchain_core.documents import Document

    out: list = []
    if not directory.exists():
        return out
    seen: set[Path] = set()
    for pat in glob_patterns:
        for f in sorted(directory.glob(pat)):
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            out.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(f),
                        "source_type": source_type,
                        "source_name": f.stem,
                    },
                )
            )
    return out


def load_examples(examples_dir: Path | None = None) -> list:
    """Curated gold-standard, working CDS views (one Document per file)."""
    return _load_text_files(
        Path(examples_dir or DEFAULT_EXAMPLES_DIR),
        source_type="example",
    )


def load_guidelines(guidelines_dir: Path | None = None) -> list:
    """Curated CDS authoring rules / hard guardrails (one Document per file)."""
    return _load_text_files(
        Path(guidelines_dir or DEFAULT_GUIDELINES_DIR),
        source_type="guideline",
    )


def load_pdfs(sources_dir: Path | None = None) -> list:
    """Load every ``*.pdf`` under ``sources_dir`` with PyMuPDF (good for code blocks)."""
    from langchain_community.document_loaders import PyMuPDFLoader

    sd = Path(sources_dir or DEFAULT_SOURCES_DIR)
    docs: list = []
    if not sd.exists():
        return docs
    for pdf in sorted(sd.glob("*.pdf")):
        try:
            loader = PyMuPDFLoader(str(pdf))
            for d in loader.load():
                meta = dict(d.metadata or {})
                meta.setdefault("source", str(pdf))
                meta.setdefault("source_type", "pdf")
                meta.setdefault("source_name", pdf.stem)
                d.metadata = meta
                docs.append(d)
        except Exception as exc:  # bad PDF -> skip, keep building
            print(f"  ! Skipping {pdf.name}: {type(exc).__name__}: {exc}")
    return docs


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def load_urls(config_path: Path | None = None) -> list:
    """Crawl the configured community / docs URLs.

    Notes:
        * A browser-style User-Agent is sent on every request — without it
          community.sap.com returns 403.
        * The classic SAP Help portal (help.sap.com/doc/abapdocu_* and
          help.sap.com/docs/abap-cloud/*) is now a Vue SPA that only renders
          content client-side, so plain HTTP returns a ~1 KB shell. Those URLs
          are intentionally NOT in the seed list — see crawler_config.yaml.
    """
    from langchain_community.document_loaders import RecursiveUrlLoader

    cp = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(cp)
    max_depth = int(cfg.get("max_depth", 2))
    max_pages = int(cfg.get("max_pages", 200))
    timeout = int(cfg.get("request_timeout", 30))
    allow_prefixes = tuple(cfg.get("allow_prefixes", ()))
    deny_substrings = tuple(cfg.get("deny_substrings", ()))
    seeds = list(cfg.get("seeds", []))
    user_agent = str(cfg.get("user_agent") or _DEFAULT_UA)
    headers = {"User-Agent": user_agent}

    def keep(url: str) -> bool:
        if not url:
            return False
        if allow_prefixes and not any(url.startswith(p) for p in allow_prefixes):
            return False
        if any(s in url for s in deny_substrings):
            return False
        return True

    out: list = []
    seen_urls: set[str] = set()
    count = 0

    for seed in seeds:
        if count >= max_pages:
            break
        print(f"  - crawling: {seed}")
        try:
            loader = RecursiveUrlLoader(
                url=seed,
                max_depth=max_depth,
                extractor=_clean_html,
                check_response_status=True,
                use_async=False,
                prevent_outside=True,
                timeout=timeout,
                headers=headers,
            )
            for d in loader.load():
                if count >= max_pages:
                    break
                url = (d.metadata or {}).get("source") or ""
                if url in seen_urls or not keep(url):
                    continue
                seen_urls.add(url)
                content = (d.page_content or "").strip()
                if len(content) < 200:
                    continue  # near-empty pages (TOC stubs) -> useless for retrieval
                meta = dict(d.metadata or {})
                meta["source_type"] = "web"
                meta.setdefault("source_name", url)
                d.metadata = meta
                d.page_content = content
                out.append(d)
                count += 1
        except Exception as exc:
            print(f"    ! crawl failed for {seed}: {type(exc).__name__}: {exc}")
    return out


def load_urls_js(config_path: Path | None = None) -> list:
    """Crawl JS-rendered SAP Help portal pages with a headless Chromium browser.

    These pages (``help.sap.com/doc/abapdocu_*`` and
    ``help.sap.com/docs/abap-cloud/*``) are Vue SPAs — the navigation tree and
    body text only exist after JavaScript executes, so we drive a real browser
    and follow same-product links breadth-first. Configured via the
    ``js_seeds`` / ``js_allow_prefixes`` / ``js_max_depth`` / ``js_max_pages``
    keys in ``crawler_config.yaml``.
    """
    cp = Path(config_path or DEFAULT_CONFIG_PATH)
    cfg = _load_yaml(cp)
    js_seeds = list(cfg.get("js_seeds") or [])
    if not js_seeds:
        return []

    try:
        from .playwright_crawler import crawl_js_pages
    except ImportError as exc:
        print(
            f"  ! Playwright not available ({exc}); skipping JS-rendered URLs.\n"
            f"    Install with:  pip install playwright  &&  playwright install chromium"
        )
        return []

    js_max_depth = int(cfg.get("js_max_depth", 3))
    js_max_pages = int(cfg.get("js_max_pages", 250))
    page_timeout_ms = int(cfg.get("request_timeout", 30)) * 1000
    settle_ms = int(cfg.get("js_settle_ms", 1500))
    min_chars = int(cfg.get("js_min_chars", 250))
    allow_prefixes = tuple(cfg.get("js_allow_prefixes") or cfg.get("allow_prefixes", ()))
    deny_substrings = tuple(cfg.get("deny_substrings", ()))
    user_agent = str(cfg.get("user_agent") or _DEFAULT_UA)

    print(
        f"  - JS crawl: {len(js_seeds)} seed(s), max_depth={js_max_depth}, "
        f"max_pages={js_max_pages}"
    )

    def _progress(stored_so_far: int, visited: int, url: str, status: str) -> None:
        flag = "" if status == "ok" else f"  [{status}]"
        print(f"    [{visited:>3}|kept~{stored_so_far - 1:>3}] {url}{flag}", flush=True)

    pages = crawl_js_pages(
        seeds=js_seeds,
        allow_prefixes=allow_prefixes,
        deny_substrings=deny_substrings,
        user_agent=user_agent,
        max_depth=js_max_depth,
        max_pages=js_max_pages,
        page_timeout_ms=page_timeout_ms,
        settle_ms=settle_ms,
        min_chars=min_chars,
        on_progress=_progress,
    )

    from langchain_core.documents import Document

    out: list = []
    for p in pages:
        out.append(
            Document(
                page_content=p["text"],
                metadata={
                    "source": p["url"],
                    "source_type": "web",
                    "source_name": p.get("title") or p["url"],
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Chunking + indexing
# ---------------------------------------------------------------------------


def chunk(docs: Iterable) -> list:
    """Split docs with separators biased toward CDS / annotation / SQL structure."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=[
            "\n@",
            "\ndefine view",
            "\ndefine view entity",
            "\nselect from",
            "\nwith parameters",
            "\n// ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )
    raw = list(docs)
    if not raw:
        return []
    split = splitter.split_documents(raw)
    for i, c in enumerate(split):
        meta = dict(c.metadata or {})
        meta["chunk_index"] = i
        src = meta.get("source") or meta.get("source_name") or "unknown"
        meta["ref_slug"] = _slug_from_url_or_path(str(src))
        c.metadata = meta
    return split


def store_chunks(chunks: list, persist_dir: Path | None = None) -> int:
    """Upsert chunks into Chroma keyed by stable IDs; returns count actually stored."""
    if not chunks:
        return 0
    pd = Path(persist_dir or DEFAULT_PERSIST_DIR)
    pd.mkdir(parents=True, exist_ok=True)
    store = get_vector_store(persist_dir=pd)

    ids: list[str] = []
    for c in chunks:
        meta = c.metadata or {}
        sid = _sha1_id(
            meta.get("source", "?"),
            int(meta.get("chunk_index", 0)),
            (c.page_content or "")[:200],
        )
        meta["doc_id"] = sid
        c.metadata = meta
        ids.append(sid)

    # Chroma's add_documents is upsert-by-id when IDs are passed (recent versions).
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def write_manifest(pdf_docs, url_docs, total_chunks) -> dict:
    DEFAULT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "last_synced_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "counts": {
            "pdf_documents": len(pdf_docs),
            "web_documents": len(url_docs),
            "chunks_stored": int(total_chunks),
        },
        "pdf_sources": sorted({(d.metadata or {}).get("source", "") for d in pdf_docs}),
        "web_sources_sample": sorted(
            {(d.metadata or {}).get("source", "") for d in url_docs}
        )[:50],
    }
    DEFAULT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def build_index(
    *,
    sources_dir: Path | None = None,
    config_path: Path | None = None,
    persist_dir: Path | None = None,
    skip_urls: bool = False,
    skip_pdfs: bool = False,
    skip_js: bool = False,
    skip_examples: bool = False,
    skip_guidelines: bool = False,
) -> dict:
    """Run the full ingest pipeline. Returns the manifest dict."""
    print(">>> Loading PDFs ...", flush=True)
    pdf_docs = [] if skip_pdfs else load_pdfs(sources_dir)
    print(f"    {len(pdf_docs)} page-docs from PDFs", flush=True)

    print(">>> Loading gold CDS examples ...", flush=True)
    example_docs = [] if skip_examples else load_examples()
    print(f"    {len(example_docs)} example file(s)", flush=True)
    for d in example_docs:
        print(f"      - {(d.metadata or {}).get('source_name')}", flush=True)

    print(">>> Loading CDS guidelines ...", flush=True)
    guideline_docs = [] if skip_guidelines else load_guidelines()
    print(f"    {len(guideline_docs)} guideline file(s)", flush=True)
    for d in guideline_docs:
        print(f"      - {(d.metadata or {}).get('source_name')}", flush=True)

    print(">>> Crawling URLs (static HTML) ...", flush=True)
    url_docs = [] if skip_urls else load_urls(config_path)
    print(f"    {len(url_docs)} page-docs from static crawl", flush=True)

    print(">>> Crawling URLs (JS-rendered, headless Chromium) ...", flush=True)
    js_docs = [] if skip_js else load_urls_js(config_path)
    print(f"    {len(js_docs)} page-docs from JS crawl", flush=True)

    all_web_docs = url_docs + js_docs

    print(">>> Chunking ...", flush=True)
    chunks = chunk(pdf_docs + example_docs + guideline_docs + all_web_docs)
    print(f"    {len(chunks)} chunks", flush=True)

    print(">>> Embedding + persisting (this is the slow step on first run) ...", flush=True)
    n = store_chunks(chunks, persist_dir=persist_dir)
    print(f"    Stored {n} chunks into Chroma at {persist_dir or DEFAULT_PERSIST_DIR}", flush=True)

    reset_cache()  # next get_vector_store() will reopen with the fresh state
    manifest = write_manifest(pdf_docs, all_web_docs, n)
    manifest["counts"]["example_files"] = len(example_docs)
    manifest["counts"]["guideline_files"] = len(guideline_docs)
    manifest["example_files"] = sorted(
        (d.metadata or {}).get("source_name", "") for d in example_docs
    )
    manifest["guideline_files"] = sorted(
        (d.metadata or {}).get("source_name", "") for d in guideline_docs
    )
    # Rewrite manifest with the enriched info.
    DEFAULT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
