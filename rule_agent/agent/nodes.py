# agent/nodes.py

import json
import os
import re
import unicodedata
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import llm
from .prompt_loader import load_prompt
from .schemas import (
    ExtractBatchSchema,
    IntentSchema,
    RequiredFieldsPlan,
    SyntaxReviewSchema,
)
from .utils import extract_text

# Verbs that signal the user wants something built (includes gerunds: "creating", "making").
_CDS_ACTION_WORDS = (
    "generate",
    "generating",
    "create",
    "creating",
    "creates",
    "build",
    "building",
    "make",
    "making",
    "draft",
    "drafting",
    "prepare",
    "preparing",
    "define",
    "defining",
    "design",
    "designing",
    "produce",
    "producing",
)
_CDS_ACTION_RE = (
    r"(?:generate|generating|create|creating|creates|build|building|make|making|"
    r"draft|drafting|prepare|preparing|define|defining|design|designing|produce|producing)"
)


def _is_cds_request(text: str) -> bool:
    t = (text or "").lower()
    has_domain = any(
        k in t
        for k in (
            "cds",
            "wrapper",
            "view",
            "exception view",
            "monitoring view",
            "cds rule",
            "dedicated rule",
        )
    )
    has_action = any(k in t for k in _CDS_ACTION_WORDS)
    # "generate a rule", "creating a rule", "rule ... generate", etc.
    has_rule_with_action = bool(
        re.search(
            r"\b" + _CDS_ACTION_RE + r"\b(?:\s+\w+){0,12}\brule\b",
            t,
        )
    ) or bool(
        re.search(
            r"\brule\b(?:\s+\w+){0,12}\b" + _CDS_ACTION_RE + r"\b",
            t,
        )
    )
    # "Can you help me creating a rule ...", "help us build a rule"
    has_help_then_rule = bool(
        re.search(
            r"\bhelp\b.{0,72}\b" + _CDS_ACTION_RE + r"\b.{0,72}\brule\b",
            t,
        )
    )
    cds_with_action = "cds" in t and bool(re.search(r"\b" + _CDS_ACTION_RE + r"\b", t))
    return cds_with_action or (
        has_domain and has_action
    ) or (has_action and has_rule_with_action) or has_help_then_rule


def _conversation_reply(user_text: str) -> str:
    """Natural chat reply before CDS flow starts."""
    lowered = (user_text or "").strip().lower()
    if lowered in {"hi", "hello", "hey", "hii", "hola"}:
        return "Hi, how can I help you?"

    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a friendly conversational assistant for an SAP CDS wrapper builder. "
                    "Reply naturally to the user's message in 1-2 short sentences. "
                    "Do not ask for technical fields yet unless the user asked to create/generate/build a CDS wrapper/view/rule. "
                    "If asked casual chat (like 'how are you'), answer naturally."
                )
            ),
            HumanMessage(content=user_text or ""),
        ]
    )
    text = extract_text(resp).strip()
    if text:
        return text
    return "I am here and ready to help. Tell me what you want to do."


def conversation_gate_node(state: dict):

    if state.get("cds_flow_started"):
        return {}

    last_user_msg = _last_human_message_content(state.get("messages", []))
    if _is_cds_request(last_user_msg):
        return {"cds_flow_started": True}

    return {
        "messages": [
            AIMessage(
                content=_conversation_reply(last_user_msg)
            )
        ]
    }


def _message_content_as_text(content: Any) -> str:
    """Normalize LC message content (string or multimodal blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts).strip()
    return str(content).strip()


def _all_user_text(messages) -> str:
    parts = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            t = _message_content_as_text(m.content)
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def _normalize_turn_text(s: str) -> str:
    """Strip BOM/ZWSP and normalize Unicode so terminals/copy-paste don't break 'yes' detection."""
    t = (s or "").replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "")
    return unicodedata.normalize("NFKC", t).strip()


def _last_human_message_content(messages) -> str:
    """Latest Human turn text (LangGraph often orders messages as [Human, AIMessage])."""
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return _normalize_turn_text(_message_content_as_text(m.content))
    return ""


def _is_table_slot_key(name: str) -> bool:
    """Planner synonyms for 'which SAP tables' — should map to a single key_tables slot."""
    low = (name or "").strip().lower()
    if not low:
        return False
    if low == "key_tables":
        return True
    if "table" in low or low.endswith("_tables") or low.startswith("tables"):
        return True
    return False


def _normalize_required_fields(fields: list[str]) -> list[str]:
    """Keep stable order; merge duplicate table-driving keys into one key_tables entry."""
    out: list[str] = []
    seen: set[str] = set()
    tables_merged = False
    for raw in fields:
        f = (raw or "").strip()
        if not f:
            continue
        if _is_table_slot_key(f):
            if not tables_merged:
                out.append("key_tables")
                seen.add("key_tables")
                tables_merged = True
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


_OUTPUT_GRAIN_KEYS = frozenset(
    {
        "output_grain",
        "result_level",
        "grain",
        "output_level",
        "result_grain",
        "review_grain",
        "business_level",
        "exception_level",
        "record_level",
        "result_layout",
        "output_layout",
        "exception_layout",
        "output_record",
        "exception_record",
        "exception_output",
        "result_grain_level",
        "review_level",
        "report_level",
        "row_level",
        "row_grain",
    }
)


def _is_grain_slot_key(name: str) -> bool:
    """Heuristic: planner aliases for 'how each exception row is shown'."""
    low = (name or "").strip().lower()
    if not low:
        return False
    if low in _OUTPUT_GRAIN_KEYS:
        return True
    # *_level / *_layout / *_grain / *_record_layout / business_* level-ish keys
    if "grain" in low:
        return True
    if low.endswith("_level") or low == "level":
        return True
    if low.endswith("_layout") or low == "layout":
        return True
    return False


def _latest_ai_message_with_kv_example(messages) -> str:
    """Last assistant bubble that contains a key=value example (fallback if ordering confuses pairing)."""
    for m in reversed(messages or []):
        if not isinstance(m, AIMessage):
            continue
        body = _message_content_as_text(m.content)
        if body and _extract_kv_pairs_ordered(body):
            return body
    return ""


def _assistant_clarification_for_last_human(messages) -> str:
    """
    AIMessage the user is replying to.
    Supports [Human, AIMessage] (common here) and [AIMessage, Human]; if several AIMessages follow
    the last Human, use the last one (newest assistant bubble).
    """
    msgs = messages or []
    last_hi = None
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            last_hi = i
            break
    if last_hi is None:
        return ""
    tail = msgs[last_hi + 1 :]
    tail_ai = [m for m in tail if isinstance(m, AIMessage)]
    if tail_ai:
        return _message_content_as_text(tail_ai[-1].content)
    for m in reversed(msgs[:last_hi]):
        if isinstance(m, AIMessage):
            return _message_content_as_text(m.content)
    return ""


_PURE_AFFIRMATION_RE = re.compile(
    r"^\s*(?:"
    r"yes(?:\s+please)?|yep|yeah|yea|"
    r"ok(?:ay)?(?:\s+thanks)?|"
    r"sure|agreed|correct|fine|right|absolutely|definitely|"
    r"sounds\s+good|looks\s+good|that\s+works|"
    r"thank\s+you|thanks|"
    r"that'?s?\s+fine|that'?s?\s+good"
    r")[\s.,!?]*$",
    re.IGNORECASE | re.VERBOSE,
)


def _is_pure_affirmation(text: str) -> bool:
    """True when the user only confirms (e.g. agrees with suggested example options)."""
    raw = _normalize_turn_text(text or "")
    if not raw or len(raw) > 120:
        return False
    return bool(_PURE_AFFIRMATION_RE.match(raw))


def _parse_suggested_key_values_from_assistant(text: str) -> dict[str, str]:
    """
    Extract snake_case_key=value or key: value from the assistant's clarification message.
    Catches examples in backticks and plain lines (e.g. key_tables=BKPF,BSEG).
    """
    pairs = _extract_kv_pairs_ordered(text)
    out: dict[str, str] = {}
    for key, val in pairs:
        out[key] = val
    return out


def _clean_kv_value(raw: str) -> str:
    v = (raw or "").strip().rstrip("`").strip().rstrip(",").strip()
    if not v:
        return ""
    # Markdown emphasis (`**`, `__`, `*`, `_`) wrapping key/value should not survive
    # into the stored value (e.g. "**exclusions:** Ignore..." -> "Ignore...").
    v = re.sub(r"^[*_`]+\s*", "", v)
    v = re.sub(r"\s*[*_`]+$", "", v)
    # Stop at em-dash / mdash prose tails: "...RBKP,RSEG` — reply yes"
    v = re.split(r"\s+[—\-]\s+", v, maxsplit=1)[0].strip()
    # Strip trailing "reply …" / "you may reply …" / "if that works …" fragments
    # the LLM appends after the example text (these never belong in the value).
    v = re.split(
        r"(?:^|\s+)(?:reply|accept|if\s+that|you\s+may\s+reply)\b",
        v,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    # Normalize multi-line assistant examples into one stored line
    v = re.sub(r"\s+", " ", v).strip()
    # Drop leftover trailing punctuation that was just sentence break
    v = v.rstrip(",;").strip()
    return v


def _is_placeholder_kv_value(val: str) -> bool:
    v = (val or "").strip().lower().strip("`").strip()
    return not v or v in ("...", "…", "key_tables=...", "tbd", "n/a", "na")


def _extract_kv_pairs_ordered(text: str) -> list[tuple[str, str]]:
    """All key=value / key: value pairs in document order; supports multiline inside `` `...` ``."""
    pairs, _backticked = _extract_kv_pairs_with_origin(text)
    return pairs


def _extract_kv_pairs_with_origin(
    text: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Returns (all_pairs_ordered, backticked_pairs_only).

    Backticked pairs are the canonical 'example' the user is being asked to
    accept with `yes` — always treat them as higher trust than free-text matches
    like 'shown: one row per ...' that may appear in prose.
    """
    if not text:
        return [], []
    scratch: list[tuple[str, str]] = []
    backticked: list[tuple[str, str]] = []

    # 1) Full backtick segments (handles `` `output_grain: line1\nline2` `` — old regex stopped at \\n)
    for m in re.finditer(r"`([\s\S]*?)`", text):
        inner = (m.group(1) or "").strip()
        km = re.match(
            r"^([a-z][a-z0-9_]*)\s*[:=]\s*(.+)$",
            inner,
            re.IGNORECASE | re.DOTALL,
        )
        if not km:
            continue
        key = km.group(1).strip().lower()
        val = _clean_kv_value(km.group(2))
        if not key or _is_placeholder_kv_value(val):
            continue
        scratch.append((key, val))
        backticked.append((key, val))

    # 2) Inline single-line pairs outside backticks only.
    #    IMPORTANT: only consume SAME-LINE whitespace around `:`/`=`. If \s* could
    #    span newlines, `use:` followed by a newline would swallow the next line's
    #    real `key: value` pair as its value (this is the exclusions: bug).
    text_wo_ticks = re.sub(r"`[\s\S]*?`", " ", text)
    kv_line = re.compile(
        r"\b([a-z][a-z0-9_]*)[ \t]*[:=][ \t]*([^\n`]+)",
        re.IGNORECASE,
    )
    for m in kv_line.finditer(text_wo_ticks):
        key = m.group(1).strip().lower()
        val = _clean_kv_value(m.group(2))
        if not key or _is_placeholder_kv_value(val):
            continue
        scratch.append((key, val))

    # Last occurrence per key wins; output keys sorted by last appearance index
    last_i: dict[str, int] = {}
    by_k: dict[str, str] = {}
    for i, (k, v) in enumerate(scratch):
        last_i[k] = i
        by_k[k] = v
    keys_sorted = sorted(by_k.keys(), key=lambda x: last_i[x])
    ordered = [(k, by_k[k]) for k in keys_sorted]

    # Same dedup for backticked-only set.
    bt_last: dict[str, int] = {}
    bt_val: dict[str, str] = {}
    for i, (k, v) in enumerate(backticked):
        bt_last[k] = i
        bt_val[k] = v
    bt_keys_sorted = sorted(bt_val.keys(), key=lambda x: bt_last[x])
    backticked_ordered = [(k, bt_val[k]) for k in bt_keys_sorted]

    return ordered, backticked_ordered


def _suggestion_value_for_missing_field(
    missing_key: str, pairs_ordered: list[tuple[str, str]]
) -> str | None:
    """Map planner snake_case key ↔ assistant example key (often key_tables=…)."""
    if not pairs_ordered:
        return None
    sk = missing_key.strip().lower()

    for pk, pv in pairs_ordered:
        if pk == sk:
            return pv

    sk_fold = sk.replace("_", "")
    for pk, pv in pairs_ordered:
        if pk.replace("_", "") == sk_fold:
            return pv

    # Planner uses driver_tables / sap_tables; assistant almost always prints key_tables=
    table_synonyms = (
        "key_tables",
        "tables",
        "sap_tables",
        "main_tables",
        "driver_tables",
        "source_tables",
        "involved_tables",
    )
    if any(t in sk for t in ("table", "tables")):
        for pk, pv in reversed(pairs_ordered):
            if pk in table_synonyms:
                return pv

    if _is_grain_slot_key(sk):
        for pk, pv in reversed(pairs_ordered):
            if _is_grain_slot_key(pk):
                return pv

    if len(sk) >= 6:
        for pk, pv in pairs_ordered:
            if len(pk) >= 6 and (pk in sk or sk in pk):
                return pv

    return None


def _apply_yes_means_suggested_options(
    updated_fields: dict,
    missing: list,
    allowed: set,
    prior_question: str,
    last_user_msg: str,
) -> None:
    """If user only says yes/ok, fill missing keys from key=value examples in the prior assistant turn.

    Strict semantic mapping: a backticked example (e.g. `output_grain: ...`) only fills a
    SAME-CATEGORY missing slot — never an unrelated one like `exclusions` or
    `amount_threshold`. We treat backticked pairs as the canonical suggestion and fall
    back to free-text pairs only for direct/fuzzy hits.
    """
    if not _is_pure_affirmation(last_user_msg):
        return
    pairs_ordered, backticked_pairs = _extract_kv_pairs_with_origin(prior_question)
    if not pairs_ordered and not backticked_pairs:
        return

    suggested = dict(pairs_ordered)

    for mk in missing:
        current = str(updated_fields.get(mk, "") or "").strip()
        if current:
            continue
        sk = mk.strip().lower()
        raw = suggested.get(sk)
        if raw:
            updated_fields[mk] = raw
            continue
        fuzzy = _suggestion_value_for_missing_field(mk, pairs_ordered)
        if fuzzy:
            updated_fields[mk] = fuzzy

    # Canonical-example rescue: assistant typically asks ONE question per turn and
    # shows ONE backticked example. If that example's key is an alias the planner
    # didn't use verbatim (e.g. `output_grain` shown for a `business_level` slot),
    # find a *category-compatible* still-unfilled missing slot and fill it.
    if backticked_pairs:
        # Use the last backticked pair (most recently shown, usually the question's example).
        bt_key, bt_value = backticked_pairs[-1]
        if bt_value:
            _fill_first_compatible_missing_slot(
                updated_fields, missing, bt_key, bt_value
            )

    # Planner sometimes emits several table-like keys; one "yes" should satisfy all of them together.
    table_missing = [
        mk
        for mk in missing
        if _is_table_slot_key(mk)
        and not str(updated_fields.get(mk, "") or "").strip()
    ]
    if table_missing and pairs_ordered:
        resolved = None
        for mk in missing:
            v = str(updated_fields.get(mk, "") or "").strip()
            if v and _is_table_slot_key(mk):
                resolved = v
                break
        if resolved is None:
            for pk, pv in reversed(pairs_ordered):
                if pk == "key_tables" or "table" in pk:
                    resolved = pv
                    break
        if resolved is not None:
            for mk in table_missing:
                updated_fields[mk] = resolved

    grain_missing = [
        mk
        for mk in missing
        if _is_grain_slot_key(mk)
        and not str(updated_fields.get(mk, "") or "").strip()
    ]
    if grain_missing:
        resolved = None
        for mk in missing:
            v = str(updated_fields.get(mk, "") or "").strip()
            if v and _is_grain_slot_key(mk):
                resolved = v
                break
        if resolved is None:
            for pk, pv in reversed(pairs_ordered):
                if _is_grain_slot_key(pk):
                    resolved = pv
                    break
        if resolved is None and backticked_pairs:
            for pk, pv in reversed(backticked_pairs):
                if _is_grain_slot_key(pk):
                    resolved = pv
                    break
        if resolved is not None:
            for mk in grain_missing:
                updated_fields[mk] = resolved


def _fill_first_compatible_missing_slot(
    updated_fields: dict,
    missing: list,
    suggestion_key: str,
    suggestion_value: str,
) -> None:
    """Find a still-empty missing slot whose category matches the example's key and fill it.

    Categories we currently understand: tables, grain. Anything else only fills if the
    planner key itself is a substring/superstring match of the suggestion key — keeps
    `output_grain` from poisoning slots like `exclusions` or `amount_threshold`.
    """
    sk_low = (suggestion_key or "").strip().lower()
    if not sk_low or not suggestion_value:
        return

    is_grain = _is_grain_slot_key(sk_low)
    is_table = _is_table_slot_key(sk_low)

    for mk in missing:
        if str(updated_fields.get(mk, "") or "").strip():
            continue
        mk_low = (mk or "").strip().lower()
        same_category = (
            (is_grain and _is_grain_slot_key(mk_low))
            or (is_table and _is_table_slot_key(mk_low))
        )
        if same_category:
            updated_fields[mk] = suggestion_value
            return

    # Fallback: only when the planner used the *exact* same key as the suggestion.
    if sk_low in {(m or "").strip().lower() for m in missing}:
        for mk in missing:
            if (mk or "").strip().lower() == sk_low and not str(
                updated_fields.get(mk, "") or ""
            ).strip():
                updated_fields[mk] = suggestion_value
                return


def _format_params(collected: dict) -> str:
    return json.dumps(collected or {}, indent=2, ensure_ascii=False)


# -----------------------------
# CDS PARAMETERS COLLECTION HELPERS
# -----------------------------


_NEGATIVE_PARAM_RESPONSES = {
    "no",
    "nope",
    "nah",
    "none",
    "no thanks",
    "no, thanks",
    "skip",
    "n/a",
    "na",
    "not now",
    "no additional",
    "nothing",
    "nothing else",
    "no more",
    "no more params",
    "no more parameters",
    "no parameters",
}

_DATE_PARAMETER_EXAMPLE = "posting date"
_ADDITIONAL_PARAMETERS_EXAMPLE = "company code, fiscal year, amount threshold"


def _strip_inline_keys(text: str, keys: tuple[str, ...]) -> str:
    """Drop a leading ``key=`` / ``key:`` prefix the user typed when answering."""
    cleaned = (text or "").strip().strip("`").strip()
    for key in keys:
        m = re.match(rf"^{re.escape(key)}\s*[:=]\s*(.+)$", cleaned, re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()
            break
    return cleaned.strip()


def _parse_date_parameter_response(text: str) -> str:
    """Capture the user's reply to the mandatory date-parameter question.

    Accepts: a plain phrase ("creation date"), a `key=value` line, or a pure
    affirmation (which means "use the example as-is").
    """
    raw = _normalize_turn_text(text or "")
    if not raw:
        return _DATE_PARAMETER_EXAMPLE
    if _is_pure_affirmation(raw):
        return _DATE_PARAMETER_EXAMPLE
    pairs = _extract_kv_pairs_ordered(raw)
    for k, v in pairs:
        if "date" in k or k in {"date_parameter", "mandatory_date", "p_date"}:
            return v
    return _strip_inline_keys(raw, ("date_parameter", "date"))


def _parse_additional_parameters_response(text: str) -> str:
    """Capture the user's reply to the optional additional-parameters question.

    Empty string means "no additional parameters". Otherwise the value is a
    free-form list (comma / "and" / semicolon separated) that the CDS prompt
    and the downstream parameter parser will both interpret.
    """
    raw = _normalize_turn_text(text or "")
    if not raw:
        return ""
    low = raw.lower().rstrip(".!? ").strip()
    if low in _NEGATIVE_PARAM_RESPONSES:
        return ""
    if _is_pure_affirmation(raw):
        return _ADDITIONAL_PARAMETERS_EXAMPLE
    pairs = _extract_kv_pairs_ordered(raw)
    for k, v in pairs:
        if "additional" in k or k in {"parameters", "extra_parameters", "more_parameters"}:
            return v
    return _strip_inline_keys(raw, ("additional_parameters", "parameters", "more"))


def _split_parameter_list(raw: str) -> list[str]:
    """Split a user list like ``company code, fiscal year and amount threshold`` into items."""
    if not raw:
        return []
    text = re.sub(r"\s+\band\b\s+", ",", raw, flags=re.IGNORECASE)
    parts = re.split(r"[;,\n]+", text)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        item = p.strip().strip("`").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# Keyword aliases that let us decide which clarifying questions are SUPERSEDED
# by the user's CDS parameter list. If any keyword for a planner field appears
# inside the user's additional_parameters answer, that field is dropped from
# required_fields (so we don't ask for a hardcoded value the user already said
# is a runtime parameter).
_FIELD_PARAM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "company_code_scope": (
        "company code",
        "company codes",
        "bukrs",
    ),
    "amount_threshold": (
        "amount threshold",
        "minimum amount",
        "minimum invoice amount",
        "threshold amount",
        "min amount",
        "min invoice amount",
        "wrbtr threshold",
    ),
    "tolerance_percent": (
        "tolerance percent",
        "tolerance %",
        "match tolerance",
        "amount tolerance",
        "tolerance",
    ),
    "time_window_days": (
        "time window",
        "lookback",
        "look back",
        "look-back",
        "rolling window",
        "date window",
        "time scope",
    ),
    "exclusions": (
        "exclusion",  # matches "exclusion" and "exclusions"
    ),
}


# These fields are ALWAYS superseded by the mandatory date parameter:
# its from/to range already defines the window the view operates on, so we
# never need a separate "how many days to look back" clarification.
_FIELDS_COVERED_BY_DATE_PARAMETER = frozenset(
    {
        "time_window_days",
        "time_window",
        "time_scope",
        "date_window",
        "date_range",
        "lookback_days",
        "lookback_period",
        "lookback_window",
    }
)


def _normalize_param_text(text: str) -> str:
    return (text or "").strip().lower()


def _field_covered_by_parameter(field_name: str, additional_text: str) -> bool:
    """True if any keyword for ``field_name`` appears in the user's parameter answer."""
    if not field_name or not additional_text:
        return False
    keywords = _FIELD_PARAM_KEYWORDS.get(field_name.strip().lower())
    if not keywords:
        return False
    haystack = _normalize_param_text(additional_text)
    return any(kw in haystack for kw in keywords)


def _filter_required_fields_by_parameters(
    fields: list[str], cds_parameter_inputs: dict | None
) -> list[str]:
    """Drop required_fields entries that are already covered by the user's CDS parameters.

    Two kinds of coverage:
      1. The mandatory date parameter ALWAYS covers any time-window / lookback field
         (its from/to range parameterizes the window — no separate lookback question).
      2. Anything in additional_parameters matching the keyword map above.
    """
    if not fields:
        return list(fields or [])
    inputs = cds_parameter_inputs or {}
    additional = _normalize_param_text(inputs.get("additional_parameters") or "")
    date_param_present = bool(
        _normalize_param_text(inputs.get("date_parameter") or "")
    )
    kept: list[str] = []
    for f in fields:
        low = (f or "").strip().lower()
        if date_param_present and low in _FIELDS_COVERED_BY_DATE_PARAMETER:
            continue
        if additional and _field_covered_by_parameter(f, additional):
            continue
        kept.append(f)
    return kept


# -----------------------------
# PARAMETERS NODE
# -----------------------------


DATE_PARAMETER_QUESTION_TEMPLATE = """Great, the business design for **{intent}** is captured. \
Now let's define the **CDS view parameters** the backend will pass at runtime (so values aren't hardcoded).

1\u20e3 **What is the mandatory date parameter?**
*(The date field this view will be parameterized on \u2014 e.g. creation date, posting date, document date, clearing date.)*

Reply in your own words (e.g. "creation date"), or accept this example:

`date_parameter={example}`

Reply **yes** to use that example as-is."""


ADDITIONAL_PARAMETERS_QUESTION_TEMPLATE = """Got it \u2014 date parameter captured (**{date_param}**).

1\u20e3 **Do you want to add any additional parameters?**
*(They will also become CDS view parameters your backend can pass values for \u2014 e.g. company code, fiscal year, amount threshold, vendor account group.)*

- Reply **no** to skip (date parameter only).
- Or list them comma-separated, for example: `additional_parameters={example}`
- Or reply **yes** to accept the example list as-is."""


def parameters_node(state: dict):
    """State machine: ask for the mandatory date parameter, then for optional extras.

    Each turn either emits one question and ends the turn (waiting for the user
    to reply) or, when both answers are in, sets parameters_collection_done so
    the graph falls through to ``explain``.
    """

    if state.get("parameters_collection_done"):
        return {}

    phase = state.get("params_phase")
    intent_txt = (state.get("intent") or "your SAP monitoring control").strip()

    if phase is None:
        body = DATE_PARAMETER_QUESTION_TEMPLATE.format(
            intent=intent_txt, example=_DATE_PARAMETER_EXAMPLE
        )
        return {
            "params_phase": "ask_date",
            "messages": [AIMessage(content=body)],
        }

    last_user_msg = _last_human_message_content(state.get("messages", []))
    inputs = dict(state.get("cds_parameter_inputs") or {})

    if phase == "ask_date":
        date_value = _parse_date_parameter_response(last_user_msg) or _DATE_PARAMETER_EXAMPLE
        inputs["date_parameter"] = date_value
        body = ADDITIONAL_PARAMETERS_QUESTION_TEMPLATE.format(
            date_param=date_value, example=_ADDITIONAL_PARAMETERS_EXAMPLE
        )
        return {
            "params_phase": "ask_more",
            "cds_parameter_inputs": inputs,
            "messages": [AIMessage(content=body)],
        }

    if phase == "ask_more":
        more_value = _parse_additional_parameters_response(last_user_msg)
        inputs["additional_parameters"] = more_value
        return {
            "params_phase": "done",
            "cds_parameter_inputs": inputs,
            "parameters_collection_done": True,
        }

    return {"parameters_collection_done": True}


def _fields_still_needed(required: list[str], collected: dict) -> list[str]:
    """Keys from required_fields that are absent or empty in collected_fields."""
    out: list[str] = []
    for f in required:
        if f not in collected:
            out.append(f)
            continue
        if not str(collected.get(f, "") or "").strip():
            out.append(f)
    return out


# -----------------------------
# INTENT NODE
# -----------------------------


_INTENT_FALLBACK_BAD_TOKENS = (
    "from user message",
    "from the user",
    "name the",
    "short phrase",
    "one short phrase",
    "monitoring control from",
    "exception or monitoring",
)


def _looks_like_prompt_echo(intent_text: str) -> bool:
    """LLM occasionally returns the schema description verbatim instead of a real label."""
    low = (intent_text or "").strip().lower()
    if not low or len(low) < 3:
        return True
    if any(tok in low for tok in _INTENT_FALLBACK_BAD_TOKENS):
        return True
    return False


def intent_node(state: dict):

    # Once requirements are planned, never re-run intent from short replies like "yes"
    # (would corrupt intent and downstream prompts).
    if state.get("required_fields") is not None:
        return {}
    if state.get("intent"):
        return {}

    last_user_msg = _last_human_message_content(state.get("messages", []))

    structured_llm = llm.with_structured_output(
        IntentSchema,
        method="function_calling",
    )

    result = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You label the SAP S/4HANA exception/monitoring control the user wants to build. "
                    "Reply ONLY with a short business label (2-6 words) that names the control type "
                    "(e.g. 'Duplicate vendor invoice check', 'GR/IR clearing exception', "
                    "'3-way match tolerance breach'). Do NOT repeat the instruction text, "
                    "do NOT include phrases like 'name the control' or 'from user message'."
                )
            ),
            HumanMessage(content=last_user_msg or ""),
        ]
    )

    intent_value = (result.intent or "").strip()
    if _looks_like_prompt_echo(intent_value):
        # Safe fallback: use the user's own phrasing rather than the leaked prompt.
        intent_value = (last_user_msg or "your SAP monitoring control").strip()
        # Cap to a reasonable label length so downstream templates render cleanly.
        if len(intent_value) > 80:
            intent_value = intent_value[:77].rstrip() + "..."

    return {
        "intent": intent_value,
    }


# -----------------------------
# REQUIREMENTS NODE
# -----------------------------


def requirements_node(state: dict):

    if state.get("required_fields") is not None:
        return {}

    user_text = _all_user_text(state.get("messages", []))
    intent = state.get("intent") or ""
    cds_params = dict(state.get("cds_parameter_inputs") or {})
    additional_params_text = (cds_params.get("additional_parameters") or "").strip()
    date_param_text = (cds_params.get("date_parameter") or "").strip()

    parameters_summary = (
        f"- Mandatory date parameter (already a CDS parameter): {date_param_text or '(none)'}\n"
        f"- Additional CDS parameters declared by the user: "
        f"{additional_params_text or '(none)'}"
    )

    structured_llm = llm.with_structured_output(
        RequiredFieldsPlan,
        method="function_calling",
    )

    result = structured_llm.invoke(
        f"""Plan clarifying snake_case fields to gather before generating a SAP CDS exception view.

Control intent: {intent}

User message(s):
{user_text}

CDS view parameters already declared (the backend will pass these at runtime):
{parameters_summary}

Return required_fields: 3–7 short names for facts still needed. Prefer these exact names when applicable:
key_tables (tables/sources), exception_or_match_logic, time_window_days (or time_scope),
amount_threshold, tolerance_percent, company_code_scope, output_grain, exclusions.

Always use the name key_tables for “which SAP tables drive this control” so confirmation examples match extraction.
Include **at most one** tables-related field in required_fields — never list multiple synonyms (only key_tables).

CRITICAL — do NOT plan a clarifying field for anything the user has already declared as a CDS view parameter:
- The mandatory date_parameter ALWAYS supersedes any lookback / time-window question.
  Its from/to range is what the backend will pass at runtime, so NEVER include
  time_window_days, time_scope, lookback_days, date_window or similar — even if the
  additional_parameters list does not mention them.
- If the user's additional CDS parameters mention "company code(s)" or "BUKRS" → DO NOT include company_code_scope.
- If they mention "amount threshold" / "minimum amount" → DO NOT include amount_threshold.
- If they mention "tolerance" / "tolerance percent" → DO NOT include tolerance_percent.
- If they mention "exclusion(s)" → DO NOT include exclusions.
The values for these will come from the backend at runtime, so we must NOT ask the user to give a hardcoded value.

If the user already gave enough detail on involved tables, how to detect the exception,
time scope, and thresholds/tolerances when relevant, return an empty list.

Do not ask about tools, transport, or non-SAP configuration."""
    )

    planned = _normalize_required_fields(list(result.required_fields or []))
    filtered = _filter_required_fields_by_parameters(planned, cds_params)
    return {"required_fields": filtered}


# -----------------------------
# EXTRACT NODE
# -----------------------------


def extract_node(state: dict):

    required = state.get("required_fields") or []
    collected_so_far = dict(state.get("collected_fields", {}))
    # Never trust stale state["missing_fields"] — LangGraph partial merges can leave it [] forever.
    missing = _fields_still_needed(required, collected_so_far)
    if not missing:
        return {}

    last_user_msg = _last_human_message_content(state.get("messages", []))
    allowed = set(required)
    if not allowed:
        allowed = set(missing)

    prior_question = _assistant_clarification_for_last_human(state.get("messages", []))
    if not prior_question.strip():
        prior_question = _latest_ai_message_with_kv_example(state.get("messages", []))

    updated_fields = dict(collected_so_far)

    # Pure confirmations: deterministic parse only — LLM structured output is unreliable here.
    if _is_pure_affirmation(last_user_msg):
        _apply_yes_means_suggested_options(
            updated_fields,
            missing,
            allowed,
            prior_question,
            last_user_msg or "",
        )
    else:
        structured_llm = llm.with_structured_output(
            ExtractBatchSchema,
            method="function_calling",
        )

        result = structured_llm.invoke(
            f"""You extract structured parameters for an SAP CDS exception-wrapper builder.

Allowed keys (exact snake_case field names only — use these as field_name): {sorted(allowed)}

Keys we still need this turn (prioritize filling these if the user answered): {missing}

Already collected (user may revise any of these): {json.dumps(collected_so_far, ensure_ascii=False)}

Rules:
- Read the user's latest message in plain language. Map their meaning to the correct key(s).
  Example: "one row per vendor + reference + amount" → output_grain with that text as value.
- key=value or key: value lines still work, but conversational answers are expected.
- If the user corrects an earlier answer (e.g. "actually use last 30 days"), return an updated pair
  for the matching allowed key even if it was not listed under missing this turn.
- If the message is only a question or chitchat with no factual answer for any allowed key, return no pairs.
- Value must be concise prose suitable to store as the parameter (no keys prefix in the value).

Assistant's prior question (for confirmation / context):
{prior_question or "(none)"}

User's latest message:
{last_user_msg}
"""
        )

        for p in result.pairs or []:
            name = (p.field_name or "").strip()
            val = (p.value or "").strip()
            if name in allowed and val:
                updated_fields[name] = val

    return {
        "collected_fields": updated_fields,
    }


# -----------------------------
# MISSING NODE
# -----------------------------


def missing_node(state: dict):

    required = state.get("required_fields") or []
    collected = state.get("collected_fields", {})

    missing = _fields_still_needed(required, collected)

    return {
        "missing_fields": missing,
    }


# -----------------------------
# QUESTION NODE
# -----------------------------


KEY_TABLES_QUESTION_TEMPLATE = """You're building a CDS exception wrapper for: **{intent}**

1️⃣ **Which SAP tables should this check read from?**
*(Choose the minimum tables that hold the data you need. For duplicate invoices, teams often use MM invoices (**RBKP/RSEG**) and/or FI postings (**BKPF/BSEG**) depending on how invoices are posted.)*

Reply in your own words with table names, or accept this example:

`key_tables=BKPF,BSEG,RBKP,RSEG`

Reply **yes** to use that example as-is."""

EXCEPTION_OR_MATCH_LOGIC_TEMPLATE = """You're defining the exception logic for: **{intent}**

1️⃣ **What should count as an exception or mismatch?**
*(Say what belongs on the exception list — duplicate invoices, threshold breaches, missing links, etc.)*

Reply in your own words first. You can also accept this concrete example:

`{field_key}=Vendor invoice documents flagged when the same vendor (LIFNR), company code (BUKRS), invoice/reference field you use for matching (e.g. XBLNR or BELNR), currency (WAERS), and gross invoice amount (WRBTR or equivalent) occur together more than once within the posting time window`

Reply **yes** to use that example as-is."""

_EXCEPTION_LOGIC_KEYS = frozenset(
    {"exception_or_match_logic", "exception_logic", "match_logic", "exception_rule"}
)

OUTPUT_GRAIN_QUESTION_TEMPLATE = """You're shaping the result layout for: **{intent}**

1️⃣ **Result level (how each exception shows up)**
*(Pick one business grain — company code, document, line item, vendor + invoice, etc.)*

Reply in your own words first, or accept this example:

`{field_key}=One row per accounting document line item`

Reply **yes** to use that example as-is."""


def question_node(state: dict):

    if not state["missing_fields"]:
        return {}

    next_key = state["missing_fields"][0]
    # Deterministic copy avoids LLM drift ("payments without PO", vague intros) and keeps key_tables= parsable.
    if next_key == "key_tables":
        intent_txt = (state.get("intent") or "your SAP monitoring control").strip()
        body = KEY_TABLES_QUESTION_TEMPLATE.format(intent=intent_txt)
        return {"messages": [AIMessage(content=body)]}

    if next_key in _EXCEPTION_LOGIC_KEYS:
        intent_txt = (state.get("intent") or "your SAP monitoring control").strip()
        # Example line must use the same snake_case key as the planner (yes-extraction matches this).
        body = EXCEPTION_OR_MATCH_LOGIC_TEMPLATE.format(
            intent=intent_txt, field_key=next_key
        )
        return {"messages": [AIMessage(content=body)]}

    if _is_grain_slot_key(next_key):
        intent_txt = (state.get("intent") or "your SAP monitoring control").strip()
        body = OUTPUT_GRAIN_QUESTION_TEMPLATE.format(
            intent=intent_txt, field_key=next_key
        )
        return {"messages": [AIMessage(content=body)]}

    data = load_prompt("question")
    lines = f"1. {next_key}"
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{missing_fields_lines}}", lines)
        .replace(
            "{{missing_fields_json}}",
            json.dumps([next_key], ensure_ascii=False),
        )
        .replace("{{user_so_far}}", _all_user_text(state.get("messages", [])))
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    text = extract_text(resp).strip()

    return {
        "messages": [AIMessage(content=text)],
    }


# -----------------------------
# EXPLAIN NODE
# -----------------------------


def _human_message_count(messages) -> int:
    return sum(1 for m in (messages or []) if isinstance(m, HumanMessage))


def explain_node(state: dict):

    if state.get("explained"):
        return {}

    data = load_prompt("explain")
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{params}}", _format_params(state.get("collected_fields", {})))
        .replace(
            "{{cds_parameter_inputs}}",
            _format_params(state.get("cds_parameter_inputs", {})),
        )
        .replace("{{description}}", _all_user_text(state.get("messages", [])))
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    text = extract_text(resp).strip()

    approval_line = "Reply **yes** to generate the CDS view, or tell me what to change."
    if approval_line not in text:
        # Belt-and-braces: ensure the user sees a clear approval prompt even if the
        # LLM dropped the final line.
        text = text.rstrip() + "\n\n" + approval_line

    return {
        "messages": [AIMessage(content=text)],
        "explained": True,
        # Snapshot the human-message count so approval_node can tell whether the
        # user has actually replied to this recap (i.e. h_count > snapshot).
        "summary_human_count": _human_message_count(state.get("messages", [])),
    }


# -----------------------------
# APPROVAL NODE
# -----------------------------


def approval_node(state: dict):

    if not state.get("explained"):
        return {}

    # Critical: when the recap was just sent in THIS same invoke (because the user's
    # last 'yes' merely closed the final clarification), do not consume that 'yes' as
    # CDS approval. The graph routes to END and waits for the user's NEXT reply.
    current_h_count = _human_message_count(state.get("messages", []))
    summary_h_count = state.get("summary_human_count") or 0
    if current_h_count <= summary_h_count and not state.get("cds_delivered"):
        return {}

    last_user_msg = _last_human_message_content(state.get("messages", [])).lower().strip()

    if state.get("cds_delivered") and last_user_msg in (
        "no",
        "nope",
        "nothing",
        "exit",
        "quit",
        "done",
        "goodbye",
        "that's all",
    ):
        return {
            "messages": [AIMessage(content="Thanks. Goodbye!")],
            "approved": False,
            "session_ended": True,
        }

    approval_phrases = (
        "approve",
        "approved",
        "yes",
        "ok",
        "okay",
        "go ahead",
        "proceed",
        "looks good",
        "i am satisfied",
        "i'm satisfied",
        "generate",
        "create",
        "build",
    )
    cds_request_phrases = (
        "cds",
        "view",
        "wrapper",
    )

    if any(
        phrase in last_user_msg
        for phrase in ["not approve", "don't approve", "reject"]
    ):
        return {}

    is_approved = any(word in last_user_msg for word in approval_phrases)
    asks_for_cds = any(word in last_user_msg for word in cds_request_phrases)

    # Accept natural approvals like:
    # "ok, i am satisfied, generate a cds view for this"
    if not is_approved and not asks_for_cds:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "If you're happy with this, say 'yes' or 'generate CDS view' "
                        "and I will create it."
                    )
                )
            ],
            "approved": False,
        }

    if state.get("cds_delivered"):
        return {
            "messages": [
                AIMessage(content="Rule was already generated. Anything else?")
            ],
            "approved": False,
        }

    return {"approved": True}


# -----------------------------
# CDS COMPANION ARTIFACT HELPERS
# (baseinfo JSON + abapGit DDLS XML written next to every generated view)
# -----------------------------


def _strip_cds_comments(cds_code: str) -> str:
    """Drop // and /* */ comments so FROM/JOIN parsing isn't tricked by examples in comments."""
    if not cds_code:
        return ""
    no_block = re.sub(r"/\*[\s\S]*?\*/", " ", cds_code)
    no_line = re.sub(r"//[^\n]*", " ", no_block)
    return no_line


def _extract_ddl_name(cds_code: str) -> str | None:
    """Return the identifier after 'define view [entity]' (e.g. ZAI_DUPL_INV1)."""
    if not cds_code:
        return None
    m = re.search(
        r"\bdefine\s+(?:root\s+)?view(?:\s+entity)?\s+([A-Za-z_][A-Za-z0-9_]*)",
        cds_code,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def _extract_from_tables(cds_code: str) -> list[str]:
    """List of unique uppercase SAP table names referenced by FROM/JOIN clauses, in source order."""
    if not cds_code:
        return []
    cleaned = _strip_cds_comments(cds_code)
    seen: set[str] = set()
    out: list[str] = []
    pattern = re.compile(
        r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(cleaned):
        name = m.group(1).upper()
        # Skip identifiers that aren't real table-like tokens (CDS keywords, aliases)
        if name in {"SELECT", "DISTINCT", "AS"}:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _extract_ddtext(cds_code: str) -> str | None:
    """Pull the human label from @EndUserText.label: '...' (used as DDTEXT)."""
    if not cds_code:
        return None
    m = re.search(
        r"@EndUserText\.label\s*:\s*'([^']{1,120})'",
        cds_code,
    )
    return m.group(1).strip() if m else None


_PARAM_RESERVED_WORDS = frozenset(
    {
        "with",
        "parameters",
        "as",
        "select",
        "from",
        "join",
        "where",
        "group",
        "having",
        "key",
        "association",
        "to",
        "on",
    }
)


def _humanize_parameter_name(param_name: str) -> str:
    """``p_date_from`` -> ``Date From``; ``p_company_code`` -> ``Company Code``."""
    cleaned = re.sub(r"^p_", "", (param_name or "").strip(), flags=re.IGNORECASE)
    parts = [p for p in re.split(r"[_\s]+", cleaned) if p]
    if not parts:
        return param_name or ""
    return " ".join(p.capitalize() for p in parts)


def _extract_cds_parameters(cds_code: str) -> list[dict]:
    """Parse the ``with parameters ... as select`` block of a CDS DDL.

    Returns a list of dicts: ``[{"name", "type", "label"}]``. Empty list if the
    view declares no parameters. Annotation lines (``@EndUserText.label: '...'``)
    immediately preceding a parameter line are picked up as the label.
    """
    if not cds_code:
        return []
    cleaned = _strip_cds_comments(cds_code)
    block_match = re.search(
        r"\bwith\s+parameters\b(.*?)\bas\s+(?:select|projection|with|join)\b",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if not block_match:
        return []
    block = block_match.group(1)

    pending_label: str | None = None
    out: list[dict] = []
    seen: set[str] = set()

    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip(",").strip()
        if not line:
            continue
        if line.startswith("@"):
            m = re.match(
                r"@EndUserText\.label\s*:\s*'([^']{1,120})'",
                line,
            )
            if m:
                pending_label = m.group(1).strip()
            continue
        m = re.match(
            r"^([a-zA-Z_][\w]*)\s*:\s*([a-zA-Z_][\w.]*(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?)",
            line,
        )
        if not m:
            continue
        name = m.group(1).strip()
        if name.lower() in _PARAM_RESERVED_WORDS:
            continue
        type_str = re.sub(r"\s+", "", m.group(2))
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": name,
                "type": type_str,
                "label": (pending_label or _humanize_parameter_name(name)),
            }
        )
        pending_label = None

    return out


def _build_baseinfo_json(tables: list[str]) -> str:
    """{ "BASEINFO": { "FROM": [TABLE1, TABLE2, ...] } } — pretty-printed."""
    return json.dumps(
        {"BASEINFO": {"FROM": list(tables)}}, indent=2, ensure_ascii=False
    )


def _build_parameters_json(parameters: list[dict], ddl_name: str) -> str:
    """JSON payload the backend reads to know which values to send to the CDS view.

    Shape mirrors baseinfo for consistency:
        { "PARAMETERS": { "VIEW": "ZAI_...", "LIST": [ {name,type,label}, ... ] } }
    """
    return json.dumps(
        {
            "PARAMETERS": {
                "VIEW": ddl_name or "",
                "LIST": list(parameters or []),
            }
        },
        indent=2,
        ensure_ascii=False,
    )


def _xml_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _build_abapgit_ddls_xml(ddl_name: str, ddtext: str) -> str:
    """abapGit DDLS serializer XML — same shape as the user's example file."""
    safe_name = _xml_escape(ddl_name or "ZAI_GENERATED")
    safe_text = _xml_escape((ddtext or "Generated CDS view")[:120])
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<abapGit version="v1.0.0" serializer="LCL_OBJECT_DDLS" serializer_version="v1.0.0">\n'
        ' <asx:abap xmlns:asx="http://www.sap.com/abapxml" version="1.0">\n'
        '  <asx:values>\n'
        '   <DDLS>\n'
        f'    <DDLNAME>{safe_name}</DDLNAME>\n'
        '    <DDLANGUAGE>E</DDLANGUAGE>\n'
        f'    <DDTEXT>{safe_text}</DDTEXT>\n'
        '    <SOURCE_TYPE>V</SOURCE_TYPE>\n'
        '   </DDLS>\n'
        '  </asx:values>\n'
        ' </asx:abap>\n'
        '</abapGit>\n'
    )


def _write_cds_artifacts(
    ddl_name: str,
    cds_code: str,
    baseinfo_text: str,
    xml_text: str,
    parameters_text: str,
) -> str | None:
    """Write the four files into ./generated_cds/<ddl_name>/. Returns the dir, or None on error.

    File names are always lower-cased per the S/4HANA 2025 guardrails
    (zai_dupl_inv1.ddls, etc.). The folder name preserves the original case
    of the DDL identifier so it lines up with how it appears in ADT.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", ddl_name or "ZAI_GENERATED").strip("_") or "ZAI_GENERATED"
    file_stem = safe_name.lower()
    base_root = os.path.abspath(
        os.path.join(os.getcwd(), "generated_cds")
    )
    out_dir = os.path.join(base_root, safe_name)
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(
            os.path.join(out_dir, f"{file_stem}.ddls.asddls"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(cds_code or "")
        with open(
            os.path.join(out_dir, f"{file_stem}.baseinfo"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(baseinfo_text or "")
        with open(
            os.path.join(out_dir, f"{file_stem}.ddls.xml"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(xml_text or "")
        with open(
            os.path.join(out_dir, f"{file_stem}.parameters"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(parameters_text or "")
    except OSError:
        return None
    return out_dir


def _build_cds_artifacts(
    cds_code: str, intent: str
) -> tuple[str, list[str], str, str, str, list[dict]]:
    """Returns (ddl_name, tables, baseinfo_text, xml_text, parameters_text, parameters).

    parameters is the structured list parsed out of the ``with parameters``
    clause; parameters_text is the JSON written to the ``.parameters`` file.
    """
    ddl_name = _extract_ddl_name(cds_code) or "ZAI_GeneratedView"
    tables = _extract_from_tables(cds_code)
    ddtext = _extract_ddtext(cds_code) or (intent or "Generated CDS view")
    baseinfo_text = _build_baseinfo_json(tables)
    xml_text = _build_abapgit_ddls_xml(ddl_name, ddtext)
    parameters = _extract_cds_parameters(cds_code)
    parameters_text = _build_parameters_json(parameters, ddl_name)
    return ddl_name, tables, baseinfo_text, xml_text, parameters_text, parameters


# -----------------------------
# RETRIEVE REFERENCE EXAMPLES
# Pulls the closest gold CDS view(s) + supporting excerpts from the shared
# Chroma store. Runs right before cds_node so the LLM sees a working
# template that matches the user's intent.
# -----------------------------


def retrieve_examples_node(state: dict):
    if not state.get("approved"):
        return {}
    if state.get("reference_examples_text"):
        # Already retrieved in a previous turn (e.g. a re-run after editing).
        return {}

    try:
        from .retrieval import retrieve_reference_examples, index_status
    except Exception as exc:
        print(f"[retrieve_examples_node] retrieval import failed: {exc}")
        return {
            "reference_examples_text": "(retrieval module unavailable — see logs)",
            "reference_examples_meta": [],
        }

    status = index_status()
    if not status.get("exists") or status.get("chunks", 0) <= 0:
        print(
            f"[retrieve_examples_node] vector store empty at {status.get('persist_dir')}; "
            f"skipping retrieval. Run error_handling_agent/scripts/build_index.py to populate."
        )
        return {
            "reference_examples_text": (
                "(no reference examples available — "
                "run error_handling_agent/scripts/build_index.py to populate the index)"
            ),
            "reference_examples_meta": [],
        }

    intent_text = (state.get("intent") or "").strip()
    description = _all_user_text(state.get("messages", []))
    query = (
        f"Generate an ABAP CDS view for: {intent_text}.\n"
        f"Business brief:\n{description[:1500]}"
    )

    try:
        docs, text_block = retrieve_reference_examples(
            query,
            k_examples=2,
            k_other=2,
        )
    except Exception as exc:
        print(f"[retrieve_examples_node] retrieval failed: {exc}")
        return {
            "reference_examples_text": "(retrieval failed at runtime — see logs)",
            "reference_examples_meta": [],
        }

    meta_summary: list[dict] = []
    for d in docs:
        m = d.metadata or {}
        meta_summary.append(
            {
                "source_name": m.get("source_name", ""),
                "source_type": m.get("source_type", ""),
                "source": m.get("source", ""),
            }
        )

    print(
        f"[retrieve_examples_node] retrieved {len(docs)} doc(s); "
        f"examples={sum(1 for m in meta_summary if m['source_type'] == 'example')}, "
        f"others={sum(1 for m in meta_summary if m['source_type'] != 'example')}"
    )

    return {
        "reference_examples_text": text_block,
        "reference_examples_meta": meta_summary,
    }


# -----------------------------
# CDS GENERATION NODE
# -----------------------------


def cds_node(state: dict):

    if not state.get("approved"):
        return {}

    data = load_prompt("cds")
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{params}}", _format_params(state.get("collected_fields", {})))
        .replace(
            "{{cds_parameter_inputs}}",
            _format_params(state.get("cds_parameter_inputs", {})),
        )
        .replace("{{description}}", _all_user_text(state.get("messages", [])))
        .replace(
            "{{reference_examples}}",
            (state.get("reference_examples_text") or "(no reference examples available)"),
        )
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    body = extract_text(resp).strip()

    fence = re.search(r"```(?:abap|cds)?\s*\n([\s\S]*?)```", body, re.IGNORECASE)
    cds_code = fence.group(1).strip() if fence else body

    (
        ddl_name,
        _tables,
        baseinfo_text,
        xml_text,
        parameters_text,
        parameters,
    ) = _build_cds_artifacts(cds_code, state.get("intent") or "")
    out_dir = _write_cds_artifacts(
        ddl_name, cds_code, baseinfo_text, xml_text, parameters_text
    )

    artifacts_block = (
        f"\n\n---\n\n**Companion artifacts** (also saved to `{out_dir}`)"
        if out_dir
        else "\n\n---\n\n**Companion artifacts** (could not write to disk — copy manually)"
    )
    artifacts_block += (
        f"\n\n`{ddl_name}.baseinfo`\n\n```json\n{baseinfo_text}\n```\n\n"
        f"`{ddl_name}.ddls.xml`\n\n```xml\n{xml_text}```\n\n"
        f"`{ddl_name}.parameters`\n\n```json\n{parameters_text}\n```"
    )

    confirmation = (
        "Rule approved. Here is your CDS view:\n\n" + body + artifacts_block
    )

    return {
        "messages": [AIMessage(content=confirmation)],
        "cds_delivered": True,
        "cds_code": cds_code,
        "cds_review_done": False,
        "cds_ddl_name": ddl_name,
        "cds_baseinfo": baseinfo_text,
        "cds_xml": xml_text,
        "cds_parameters_text": parameters_text,
        "cds_parameters": parameters,
        "cds_artifacts_dir": out_dir,
    }


# -----------------------------
# SYNTAX REVIEW NODE (auto-healing loop)
# Runs BEFORE cds_review_node so we ensure the DDL is syntactically valid
# ABAP CDS before any performance/architecture review.
# -----------------------------


MAX_SYNTAX_FIX_RETRIES = 3


def _strip_cds_fences(text: str) -> str:
    """If the LLM wrapped the corrected_cds in ```abap … ``` despite the prompt rule,
    pull out the code anyway."""
    if not text:
        return ""
    fence = re.search(r"```(?:abap|cds|sql)?\s*\n?([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _format_issue_lines(issues: list[str]) -> str:
    if not issues:
        return "(none)"
    return "\n".join(f"- {ln.strip().lstrip('- ').strip()}" for ln in issues if ln and ln.strip())


def syntax_review_node(state: dict):
    """Validate that the generated CDS uses only ABAP-CDS-supported syntax. If not,
    iteratively ask the LLM to rewrite using supported constructs (up to
    MAX_SYNTAX_FIX_RETRIES). Updates `cds_code` + companion artifacts in place."""

    # Skip switch (handy for offline testing or when the user just wants the raw draft).
    if os.getenv("RULE_AGENT_SKIP_CDS_SYNTAX_REVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return {
            "cds_syntax_status": "SKIPPED",
            "cds_syntax_issues": [],
            "cds_syntax_retries": 0,
            "cds_syntax_review_done": True,
        }

    # Don't re-run on subsequent turns.
    if state.get("cds_syntax_review_done"):
        return {}

    cds_code = state.get("cds_code") or ""
    if not cds_code or not state.get("cds_delivered"):
        return {}

    data = load_prompt("cds_syntax_review")
    intent = state.get("intent") or ""
    params_json = _format_params(state.get("collected_fields", {}))

    structured_llm = llm.with_structured_output(
        SyntaxReviewSchema,
        method="function_calling",
    )

    issues_so_far: list[str] = []
    final_status = "FAILED"
    retries_used = 0
    last_error: str | None = None

    for attempt in range(1, MAX_SYNTAX_FIX_RETRIES + 1):
        prior_issues = _format_issue_lines(issues_so_far)
        user_block = (
            data["user"]
            .replace("{{intent}}", intent)
            .replace("{{params}}", params_json)
            .replace("{{cds_code}}", cds_code)
            .replace("{{prior_issues}}", prior_issues)
            .replace("{{attempt}}", str(attempt))
            .replace("{{max_attempts}}", str(MAX_SYNTAX_FIX_RETRIES))
        )

        try:
            result = structured_llm.invoke(
                [
                    SystemMessage(content=data["system"]),
                    HumanMessage(content=user_block),
                ]
            )
        except Exception as exc:  # network / SDK / schema errors must not crash the run
            last_error = f"{type(exc).__name__}: {exc}"
            issues_so_far.append(f"Validator call failed: {last_error}")
            break

        status = (result.syntax_status or "").strip().upper()
        new_issues = [s for s in (result.issues or []) if s and s.strip()]
        retries_used = attempt

        if status == "PASSED":
            final_status = "PASSED"
            # On a clean pass with no rewrites, leave issues_so_far as accumulated history
            break

        # FAILED → adopt the corrected DDL and loop again
        rewritten = _strip_cds_fences(result.corrected_cds or "")
        if not rewritten or len(rewritten) < 60:
            # Validator said FAILED but didn't return a usable rewrite — stop trying.
            issues_so_far.extend(new_issues)
            issues_so_far.append(
                "Validator returned FAILED but provided no usable corrected_cds — stopping."
            )
            final_status = "FAILED"
            break

        issues_so_far.extend(new_issues)
        cds_code = rewritten
        # Loop continues with the rewritten CDS

    # Recompute companion artifacts for whatever cds_code we finished with
    (
        ddl_name,
        _tables,
        baseinfo_text,
        xml_text,
        parameters_text,
        parameters,
    ) = _build_cds_artifacts(cds_code, intent)
    out_dir = _write_cds_artifacts(
        ddl_name, cds_code, baseinfo_text, xml_text, parameters_text
    )

    # Build the user-facing summary
    if final_status == "PASSED" and retries_used == 1:
        header = (
            "**Syntax review** — passed on first attempt. "
            "The CDS uses only ABAP-CDS-supported syntax."
        )
    elif final_status == "PASSED":
        header = (
            f"**Syntax review** — passed after {retries_used - 1} fix attempt"
            f"{'s' if retries_used - 1 != 1 else ''}. "
            "The CDS now uses only ABAP-CDS-supported syntax."
        )
    elif last_error:
        header = (
            f"**Syntax review** — could not run (validator error after "
            f"{retries_used} attempt(s)). Continuing with the original DDL."
        )
    else:
        header = (
            f"**Syntax review** — still has issues after "
            f"{MAX_SYNTAX_FIX_RETRIES} attempts. Continuing, but please "
            "double-check before activating in ADT."
        )

    issues_block = ""
    if issues_so_far:
        issues_block = (
            "\n\nIssues addressed during the auto-fix loop:\n"
            + _format_issue_lines(issues_so_far)
        )

    code_changed = cds_code != (state.get("cds_code") or "")
    revised_block = ""
    if code_changed:
        revised_block += "\n\n**Updated CDS view (after syntax fixes):**\n\n```abap\n"
        revised_block += cds_code
        revised_block += "\n```"
        revised_block += (
            f"\n\n**Updated companion artifacts** "
            + (f"(rewritten in `{out_dir}`)" if out_dir else "(disk write failed)")
            + f"\n\n`{ddl_name}.baseinfo`\n\n```json\n{baseinfo_text}\n```\n\n"
            f"`{ddl_name}.ddls.xml`\n\n```xml\n{xml_text}```\n\n"
            f"`{ddl_name}.parameters`\n\n```json\n{parameters_text}\n```"
        )

    message = header + issues_block + revised_block

    out: dict = {
        "messages": [AIMessage(content=message)],
        "cds_syntax_status": final_status,
        "cds_syntax_issues": issues_so_far,
        "cds_syntax_retries": max(0, retries_used - 1) if final_status == "PASSED" else retries_used,
        "cds_syntax_review_done": True,
    }

    if code_changed:
        out["cds_code"] = cds_code
        out["cds_ddl_name"] = ddl_name
        out["cds_baseinfo"] = baseinfo_text
        out["cds_xml"] = xml_text
        out["cds_parameters_text"] = parameters_text
        out["cds_parameters"] = parameters
        out["cds_artifacts_dir"] = out_dir

    return out


# -----------------------------
# CDS REVIEW NODE (post-generation QA)
# -----------------------------


def _last_abap_fence(text: str) -> str | None:
    matches = list(
        re.finditer(r"```(?:abap|cds)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    )
    if not matches:
        return None
    return matches[-1].group(1).strip()


def cds_review_node(state: dict):
    """Second pass: checklist review; optional revised DDL. Skip with RULE_AGENT_SKIP_CDS_REVIEW=1."""

    if os.getenv("RULE_AGENT_SKIP_CDS_REVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return {"cds_review_done": True}

    if not state.get("cds_code") or not state.get("cds_delivered"):
        return {}

    if state.get("cds_review_done"):
        return {}

    data = load_prompt("cds_review")
    cds_code = state["cds_code"] or ""
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{params}}", _format_params(state.get("collected_fields", {})))
        .replace("{{description}}", _all_user_text(state.get("messages", [])))
        .replace("{{cds_code}}", cds_code)
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    body = extract_text(resp).strip()

    review_message = "**Engineering review**\n\n" + body
    out: dict = {
        "cds_review_done": True,
    }

    refined: str | None = None
    if "revised cds" in body.lower():
        candidate = _last_abap_fence(body)
        if candidate and len(candidate) > 80:
            refined = candidate
            out["cds_code"] = candidate

    if refined:
        # Re-generate the companion files so baseinfo/xml/parameters track the revised DDL.
        (
            ddl_name,
            _tables,
            baseinfo_text,
            xml_text,
            parameters_text,
            parameters,
        ) = _build_cds_artifacts(refined, state.get("intent") or "")
        out_dir = _write_cds_artifacts(
            ddl_name, refined, baseinfo_text, xml_text, parameters_text
        )
        out["cds_ddl_name"] = ddl_name
        out["cds_baseinfo"] = baseinfo_text
        out["cds_xml"] = xml_text
        out["cds_parameters_text"] = parameters_text
        out["cds_parameters"] = parameters
        out["cds_artifacts_dir"] = out_dir
        review_message += (
            f"\n\n---\n\n**Revised companion artifacts** "
            + (f"(updated in `{out_dir}`)" if out_dir else "(disk write failed)")
            + f"\n\n`{ddl_name}.baseinfo`\n\n```json\n{baseinfo_text}\n```\n\n"
            f"`{ddl_name}.ddls.xml`\n\n```xml\n{xml_text}```\n\n"
            f"`{ddl_name}.parameters`\n\n```json\n{parameters_text}\n```"
        )

    out["messages"] = [AIMessage(content=review_message)]
    return out
