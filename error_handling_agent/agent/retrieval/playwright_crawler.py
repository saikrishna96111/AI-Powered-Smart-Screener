"""Headless-Chromium crawler for JavaScript-rendered SAP Help portal pages.

The SAP Help portal serves a Vue.js single-page app — a plain HTTP fetch
returns a ~1 KB shell and no navigation links. By driving Chromium via
Playwright we can:

1. Wait for the page's content area to render,
2. Extract the visible text inside ``<main>`` / ``<article>``,
3. Read every ``<a href>`` (which only exists after JS executes),
4. Follow same-product links breadth-first, respecting an allow-prefix
   filter and a hard page cap.

Returns a list of dicts: ``{"url", "title", "text"}`` for each page that
yielded enough content to be useful for retrieval.
"""

from __future__ import annotations

from collections import deque
from urllib.parse import urldefrag


_EXTRACT_TEXT_JS = """() => {
    const SEL = 'main, article, [role="main"], .topic-content, .content-area, #main-content';
    const main = document.querySelector(SEL);
    const el = main || document.body;
    const clone = el.cloneNode(true);
    for (const tag of ['script','style','nav','header','footer','noscript','aside']) {
        for (const n of clone.querySelectorAll(tag)) n.remove();
    }
    return (clone.innerText || '').trim();
}"""

_EXTRACT_LINKS_JS = """() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)"""


def _normalize(url: str) -> str:
    """Drop the fragment so /page#section1 and /page#section2 are one URL."""
    if not url:
        return ""
    base, _frag = urldefrag(url)
    return base.rstrip("/")


def _keep(url: str, allow_prefixes: tuple[str, ...], deny_substrings: tuple[str, ...]) -> bool:
    if not url:
        return False
    if allow_prefixes and not any(url.startswith(p) for p in allow_prefixes):
        return False
    if any(s in url for s in deny_substrings):
        return False
    return True


def crawl_js_pages(
    seeds: list[str],
    allow_prefixes: tuple[str, ...] = (),
    deny_substrings: tuple[str, ...] = (),
    user_agent: str = "",
    max_depth: int = 3,
    max_pages: int = 300,
    page_timeout_ms: int = 30000,
    selector_timeout_ms: int = 10000,
    settle_ms: int = 1500,
    wait_selector: str = "main, article, [role='main'], .topic-content",
    min_chars: int = 250,
    headless: bool = True,
    on_progress=None,
) -> list[dict]:
    """Breadth-first crawl of JS-rendered pages.

    ``on_progress`` (optional callable) gets ``(index, total_seen, url, status)``
    on every page so the build script can show live output.
    """
    from playwright.sync_api import sync_playwright

    if not seeds:
        return []

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque((_normalize(s), 0) for s in seeds)
    out: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=user_agent or None)
        page = ctx.new_page()
        try:
            while queue and len(out) < max_pages:
                url, depth = queue.popleft()
                if not url or url in visited:
                    continue
                visited.add(url)
                if not _keep(url, allow_prefixes, deny_substrings):
                    continue
                if depth > max_depth:
                    continue

                status = "ok"
                title = ""
                text = ""
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=page_timeout_ms)
                    try:
                        page.wait_for_selector(wait_selector, timeout=selector_timeout_ms)
                    except Exception:
                        # SPA may not have any of these selectors; carry on.
                        pass
                    if settle_ms > 0:
                        page.wait_for_timeout(settle_ms)
                    text = page.evaluate(_EXTRACT_TEXT_JS) or ""
                    title = page.title() or ""
                except Exception as exc:
                    status = f"goto-fail: {type(exc).__name__}"
                    text = ""

                if on_progress:
                    on_progress(len(out) + 1, len(visited), url, status)

                if status == "ok" and text and len(text) >= min_chars:
                    out.append({"url": url, "title": title, "text": text})

                # Enqueue links (even if THIS page had too little content — its
                # children may still be useful).
                if depth + 1 > max_depth:
                    continue
                try:
                    raw_links = page.evaluate(_EXTRACT_LINKS_JS) or []
                except Exception:
                    raw_links = []

                added = 0
                seen_in_page: set[str] = set()
                for href in raw_links:
                    norm = _normalize(href)
                    if not norm or norm in seen_in_page:
                        continue
                    seen_in_page.add(norm)
                    if norm in visited:
                        continue
                    if not _keep(norm, allow_prefixes, deny_substrings):
                        continue
                    queue.append((norm, depth + 1))
                    added += 1

        finally:
            try:
                browser.close()
            except Exception:
                pass

    return out
