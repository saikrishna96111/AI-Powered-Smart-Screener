"""Demo-day CDS fixtures: known-good syntax for the three manager scenarios.

When the user's intent matches one of these scenarios, the rule agent returns
the fixture CDS verbatim (no LLM rewrite) so demos stay syntax-error free.

Disable with env ``RULE_AGENT_DEMO_SCENARIOS=0``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# Each entry: id, fixture filename, human label, match predicate over lowercased text.
_SCENARIOS: list[dict] = [
    {
        "id": "duplicate_invoice",
        "file": "01_duplicate_invoice.cds",
        "label": "Duplicate Invoice Detection",
        "view": "ZAI_DUPL_INV1",
    },
    {
        "id": "po_creator_approver",
        "file": "02_po_creator_approver.cds",
        "label": "PO Creator and Approver Same Person",
        "view": "ZAI_PO_CREATOR_APPR",
    },
    {
        "id": "vendor_bank_change",
        "file": "03_vendor_bank_change.cds",
        "label": "Vendor Bank Change Before Payment",
        "view": "ZI_VEND_BANK_CHG_PRE_PAY",
    },
]


def demo_scenarios_enabled() -> bool:
    """On by default; set RULE_AGENT_DEMO_SCENARIOS=0 to force LLM generation."""
    val = os.getenv("RULE_AGENT_DEMO_SCENARIOS", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _match_duplicate(text: str) -> bool:
    if "zai_dupl_inv" in text or "zv_dup_inv" in text:
        return True
    if "duplicate" in text and "invoice" in text:
        return True
    if "duplicate" in text and "vendor" in text and (
        "invoice" in text or "detection" in text
    ):
        return True
    return False


def _match_po_creator_approver(text: str) -> bool:
    if "zai_po_creator" in text or "zv_po_cr_app" in text:
        return True
    if ("creator" in text and "approver" in text) or (
        "created by" in text and "approver" in text
    ):
        return True
    if "sod" in text and ("po" in text or "purchase order" in text):
        return True
    if "same person" in text and (
        "po" in text or "purchase" in text or "creator" in text
    ):
        return True
    if "release" in text and "creator" in text and (
        "approver" in text or "same" in text
    ):
        return True
    return False


def _match_vendor_bank(text: str) -> bool:
    if "vend_bank" in text or "vb_chg_pay" in text or "zi_vend_bank" in text:
        return True
    if "bank" in text and ("change" in text or "changed" in text):
        if "vendor" in text or "payment" in text or "pay" in text:
            return True
    if "bec" in text and ("bank" in text or "vendor" in text):
        return True
    return False


_MATCHERS = {
    "duplicate_invoice": _match_duplicate,
    "po_creator_approver": _match_po_creator_approver,
    "vendor_bank_change": _match_vendor_bank,
}


def _load_fixture(filename: str) -> str:
    path = _DIR / filename
    raw = path.read_text(encoding="utf-8")
    # Drop leftover @OData.publish annotation lines (comments mentioning it are fine).
    raw = re.sub(r"(?m)^\s*@OData\.publish\s*:[^\n]*\n?", "", raw)
    return raw.strip() + "\n"


def match_demo_scenario(intent: str, user_text: str) -> dict | None:
    """Return ``{id, label, view, cds_code}`` if intent/messages match a demo scenario."""
    if not demo_scenarios_enabled():
        return None

    haystack = f"{intent or ''}\n{user_text or ''}".lower()
    if not haystack.strip():
        return None

    for scenario in _SCENARIOS:
        matcher = _MATCHERS[scenario["id"]]
        if matcher(haystack):
            return {
                "id": scenario["id"],
                "label": scenario["label"],
                "view": scenario["view"],
                "cds_code": _load_fixture(scenario["file"]),
            }
    return None
