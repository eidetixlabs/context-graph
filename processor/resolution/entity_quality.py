"""
processor/resolution/entity_quality.py — Shared entity mention quality filters.

Aligns server-side enforcement with identity_v2 prompt reject rules.
"""

from __future__ import annotations

import re

from processor.config import config

MIN_MENTION_CONFIDENCE: float = config.ner_min_confidence
MIN_NER_BACKFILL_CONFIDENCE: float = config.ner_min_confidence + 0.1
MAX_LLM_ENTITIES_PER_CHUNK = 20
PHASE3_MIN_CONFIDENCE = 0.85

_GENERIC_TOKENS = frozenset({
    "ai", "content", "manual", "complete", "strategic", "http", "dns",
    "pdf", "arr", "mrr", "roi", "kpi", "nps", "okr", "gtm", "plg",
    "ux", "ui", "sdk", "api", "cli", "cta", "sla", "soc", "si",
    "gdpr", "faq", "dr", "seo", "crm", "erp", "saas", "b2b", "b2c",
    "ongoing", "community", "attribution", "experimentation", "scale",
    "intel", "git", "w",
})

_GEO_ACRONYMS = frozenset({
    "uk", "us", "eu", "dach", "apac", "emea", "latam", "usd", "gbp", "eur",
    "non-us", "q4",
})

_SCHEDULE_WORDS = frozenset({
    "ongoing", "bi-weekly", "biweekly", "weekly", "monthly", "quarterly",
    "phase 1", "phase 2", "phase 3", "phase 4",
    "q1", "q2", "q3", "q4",
})

_METRIC_PHRASES = frozenset({
    "domain rating", "core web vitals", "conversion rate", "click-through rate",
    "customer acquisition cost", "lifetime value", "net promoter score",
})

_HEADING_PHRASES = frozenset({
    "executive summary", "technical overview", "introduction", "conclusion",
    "appendix", "overview", "summary", "background", "methodology",
    "references", "table of contents", "icp refinement",
})

_TEAM_FUNCTION_PHRASES = frozenset({
    "vp marketing", "vp engineering", "growth marketing", "product marketing",
    "brand studio", "platform engineering", "channel zero",
})

_JUNK_ORDINAL = re.compile(r"^(?:tier|level)\s+\d+[:\s]*$", re.IGNORECASE)
_HEADING_PREFIX = re.compile(
    r"^(?:[IVXLCDM]+|\d+)\.\s+",
    re.IGNORECASE,
)
_TEAM_HEADING_PREFIX = re.compile(
    r"^(?:vp|head of|director of|chief)\s+",
    re.IGNORECASE,
)
_VERSUS_PHRASE = re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE)
_COMPOSITE_COMMA = re.compile(r"^[^,]+,\s+[^,]+$")
_COMPOSITE_AMPERSAND = re.compile(r"\s&\s")
_NAMED_RISK_SUFFIX = re.compile(
    r"\b(degradation|dissipation|blocker|bottleneck)\b",
    re.IGNORECASE,
)
_PARENTHETICAL_MENTION = re.compile(r"^(?P<left>[^()]+)\((?P<right>[^()]+)\)$")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
_ROLE_HINT_TOKENS = frozenset({
    "advisor", "analyst", "architect", "captain", "ceo", "cfo", "chair",
    "chief", "cofounder", "coordinator", "cto", "designer", "director",
    "engineer", "founder", "head", "lead", "manager", "officer", "owner",
    "partner", "president", "producer", "specialist", "strategist", "supervisor",
    "vp",
})
_ROLE_CONTEXT_TOKENS = frozenset({
    "business", "commercial", "content", "customer", "data", "design",
    "engineering", "finance", "growth", "hr", "legal", "market", "marketing",
    "operations", "ops", "people", "platform", "pod", "product", "readiness",
    "research", "sales", "science", "security", "success", "support", "team",
})


def _normalized_alpha_tokens(value: str) -> list[str]:
    return [re.sub(r"[^a-z]", "", token.lower()) for token in value.split() if token]


def _looks_like_person_name(value: str) -> bool:
    tokens = [token.strip().strip(".") for token in value.split()]
    if not 1 <= len(tokens) <= 4:
        return False
    if not all(tokens):
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", token) for token in tokens):
        return False
    if not all(token[0].isupper() for token in tokens):
        return False
    normalized = _normalized_alpha_tokens(value)
    return not any(token in _ROLE_HINT_TOKENS for token in normalized)


def _looks_like_role_phrase(value: str) -> bool:
    normalized = [token for token in _normalized_alpha_tokens(value) if token]
    if not normalized:
        return False
    if any(token in _ROLE_HINT_TOKENS for token in normalized):
        return True
    if any(token in _ROLE_CONTEXT_TOKENS for token in normalized):
        return True
    return False


def is_subsumed_fragment(
    mention: str,
    known_mentions: set[str] | frozenset[str] | None = None,
) -> bool:
    """True when a single-token mention is a prefix of a longer known name."""
    if not known_mentions:
        return False
    text = mention.strip()
    tokens = text.split()
    if len(tokens) != 1:
        return False
    low = text.lower()
    for other in known_mentions:
        other_low = other.lower()
        if other_low == low:
            continue
        other_tokens = other.split()
        if len(other_tokens) < 2:
            continue
        if other_low.startswith(low + " ") or other_tokens[0].lower() == low:
            return True
    return False


def is_junk_mention(
    mention: str,
    known_mentions: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return True if mention should be rejected per identity_v2 quality rules."""
    text = mention.strip()
    if not text:
        return True

    if len(text) == 1:
        return True

    if known_mentions:
        if text.endswith("'s"):
            base = text[:-2]
            if base in known_mentions:
                return True
            base_lower = base.lower()
            if any(m.lower() == base_lower for m in known_mentions):
                return True

    if is_subsumed_fragment(text, known_mentions):
        return True

    if "\n" in text or _EMOJI_RE.search(text):
        return True

    if _COMPOSITE_AMPERSAND.search(text):
        return True

    if " + " in text or " / " in text:
        return True

    if _COMPOSITE_COMMA.match(text):
        return True

    if _VERSUS_PHRASE.search(text):
        return True

    parenthetical = _PARENTHETICAL_MENTION.match(text)
    if parenthetical:
        left = parenthetical.group("left").strip()
        right = parenthetical.group("right").strip()
        if (
            _looks_like_person_name(left) and _looks_like_role_phrase(right)
        ) or (
            _looks_like_role_phrase(left) and _looks_like_person_name(right)
        ):
            return True

    lower = text.lower()
    if lower in _SCHEDULE_WORDS or lower in _METRIC_PHRASES:
        return True
    if lower in _TEAM_FUNCTION_PHRASES:
        return True

    if _JUNK_ORDINAL.match(text):
        return True

    stripped_heading = _HEADING_PREFIX.sub("", lower).strip()
    if stripped_heading in _HEADING_PHRASES:
        return True

    if _TEAM_HEADING_PREFIX.match(text):
        return True

    if _NAMED_RISK_SUFFIX.search(text) and len(text.split()) >= 3:
        return True

    words = text.split()
    if len(words) >= 4:
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if capitalized <= 1:
            return True

    tokens = text.split()
    if len(tokens) == 1:
        token = re.sub(r"[^a-z0-9-]", "", tokens[0].lower())
        if token in _GENERIC_TOKENS or token in _GEO_ACRONYMS:
            return True

    return False


def passes_quality_gate(
    mention: str,
    confidence: float,
    known_mentions: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return True if mention should proceed to resolution / persistence."""
    if confidence < MIN_MENTION_CONFIDENCE:
        return False
    return not is_junk_mention(mention, known_mentions)


def derive_person_alias_base(mention: str) -> str | None:
    """Return the bare person name from a role-decorated single-person mention.

    Handles the two decoration shapes that ``is_junk_mention`` rejects:
      - parenthetical role:  "Riya (Data)"  -> "Riya";  "Launch Captain (Riya)" -> "Riya"
      - trailing comma role: "Riya Kapoor, Data Lead" -> "Riya Kapoor"

    Returns ``None`` for plain names and for true composites ("A & B", "A / B",
    "A + B") that join two distinct entities — those must stay dropped, never
    collapsed into one alias.
    """
    text = mention.strip()
    if not text:
        return None

    parenthetical = _PARENTHETICAL_MENTION.match(text)
    if parenthetical:
        left = parenthetical.group("left").strip()
        right = parenthetical.group("right").strip()
        if _looks_like_person_name(left) and _looks_like_role_phrase(right):
            return left
        if _looks_like_role_phrase(left) and _looks_like_person_name(right):
            return right
        return None

    if _COMPOSITE_COMMA.match(text):
        left, _, right = text.partition(",")
        left = left.strip()
        right = right.strip()
        if _looks_like_person_name(left) and _looks_like_role_phrase(right):
            return left

    return None
