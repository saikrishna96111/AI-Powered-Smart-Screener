"""Build / refresh the RAG index for the error-handling agent.

Run once after dropping the source PDFs into ``data/sources/`` and whenever
the SAP Help pages change:

    python scripts/build_index.py                    # everything
    python scripts/build_index.py --no-urls --no-js  # offline (PDFs + examples + guidelines)
    python scripts/build_index.py --examples-only    # only refresh the curated CDS examples

Skip flags (all may be combined):
    --no-pdfs         Skip PDFs in data/sources/*.pdf
    --no-examples     Skip CDS examples in data/sources/examples/
    --no-guidelines   Skip guideline files in data/sources/guidelines/
    --no-urls         Skip the static HTTP crawl (community.sap.com)
    --no-js           Skip the Playwright-rendered SAP Help portal crawl

The script writes ``data/INDEX_MANIFEST.json`` with counts and timestamps
so you can sanity-check what's currently in the vector store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_importable() -> None:
    here = Path(__file__).resolve().parent
    agent_root = here.parent
    sys.path.insert(0, str(agent_root))


def main() -> int:
    _ensure_importable()
    from agent.retrieval.ingest import build_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdfs", action="store_true", help="Skip PDF ingestion")
    parser.add_argument("--no-examples", action="store_true", help="Skip curated CDS examples")
    parser.add_argument("--no-guidelines", action="store_true", help="Skip curated guideline files")
    parser.add_argument("--no-urls", action="store_true", help="Skip the static HTTP crawl")
    parser.add_argument("--no-js", action="store_true", help="Skip the Playwright JS-rendered crawl")
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Quick-refresh: skip everything EXCEPT the curated CDS examples + guidelines",
    )
    parser.add_argument(
        "--sources-dir",
        default=None,
        help="Override the directory holding the source PDFs",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override the crawler_config.yaml path",
    )
    args = parser.parse_args()

    if args.examples_only:
        args.no_pdfs = True
        args.no_urls = True
        args.no_js = True

    manifest = build_index(
        sources_dir=Path(args.sources_dir) if args.sources_dir else None,
        config_path=Path(args.config) if args.config else None,
        skip_pdfs=args.no_pdfs,
        skip_urls=args.no_urls,
        skip_js=args.no_js,
        skip_examples=args.no_examples,
        skip_guidelines=args.no_guidelines,
    )

    print("\n=== Index built ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
