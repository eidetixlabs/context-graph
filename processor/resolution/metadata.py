"""
processor/resolution/metadata.py — Document-level metadata computed from mentions + keywords.

Used as Phase 2 context input only — not persisted to any store.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from processor.resolution.models import MentionCandidate
from processor.state_machine import DocumentRecord


@dataclass
class DocumentMetadata:
    author: str | None
    person_most_count: str | None
    topic_most_count: str | None


def compute_document_metadata(
    doc: DocumentRecord,
    mentions: list[MentionCandidate],
    keywords: list[str],
) -> DocumentMetadata:
    """
    Derive lightweight document metadata from Phase 1 outputs.

    Args:
        doc:      Document record (may have author_employee_id).
        mentions: Resolved mention candidates from Phase 1.
        keywords: Aggregated keywords from LLM enrichment across all chunks.

    Returns:
        DocumentMetadata with author, most-frequent person, and primary topic.
    """
    person_counts: Counter[str] = Counter(
        m.mention_text for m in mentions if m.mention_type == "person"
    )
    person_most = person_counts.most_common(1)[0][0] if person_counts else None

    keyword_counts: Counter[str] = Counter(keywords)
    topic_most = keyword_counts.most_common(1)[0][0] if keyword_counts else None

    author = person_most

    return DocumentMetadata(
        author=author,
        person_most_count=person_most,
        topic_most_count=topic_most,
    )
