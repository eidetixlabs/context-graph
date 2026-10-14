"""Unit tests for entity mention quality filters and resolution noise controls."""

from __future__ import annotations

from processor.resolution.entity_quality import (
    MAX_LLM_ENTITIES_PER_CHUNK,
    MIN_MENTION_CONFIDENCE,
    is_junk_mention,
    is_subsumed_fragment,
    passes_quality_gate,
)
from processor.resolution.models import (
    MentionCandidate,
    ResolutionMap,
    ResolutionMapEntry,
    ResolvedEntity,
)
from processor.resolution.resolver import build_resolved_entities
from processor.schemas.identity import IdentityResolutionResult, ResolvedIdentity
from processor.agents.extraction.phase2_context import (
    _format_resolved_entities,
    _prefetch_to_context_result,
)
from processor.agents.extraction.pipeline import _resolution_map_to_identity_result
from processor.schemas.scoring import ScoringResult
from processor.schemas.context import TopicContext

PASSING_CONFIDENCE = min(1.0, MIN_MENTION_CONFIDENCE + 0.05)
BELOW_THRESHOLD_CONFIDENCE = MIN_MENTION_CONFIDENCE - 0.01


def test_is_junk_mention_rejects_generic_tokens() -> None:
    assert is_junk_mention("AI")
    assert is_junk_mention("GTM")
    assert is_junk_mention("API")
    assert is_junk_mention("FAQ")
    assert not is_junk_mention("Alex Kim")


def test_is_junk_mention_rejects_possessive_when_base_exists() -> None:
    known = {"Alex Kim", "Datadog"}
    assert is_junk_mention("Alex Kim's", known)
    assert not is_junk_mention("Alex Kim's", set())


def test_is_junk_mention_rejects_headings_and_schedules() -> None:
    assert is_junk_mention("Executive Summary")
    assert is_junk_mention("I. Introduction")
    assert is_junk_mention("Phase 1")
    assert is_junk_mention("Bi-weekly")


def test_is_junk_mention_rejects_v17_noise() -> None:
    assert is_junk_mention("Community")
    assert is_junk_mention("Attribution")
    assert is_junk_mention("VP Marketing")
    assert is_junk_mention("BigQuery versus Snowflake")
    assert is_junk_mention("Jira, Linear")
    assert is_junk_mention("UK")
    assert is_junk_mention("standalone brand campaign dollars")


def test_is_junk_mention_rejects_role_decorated_composites() -> None:
    assert is_junk_mention("Naman + Squad Lead")
    assert is_junk_mention("Naman / Squad Lead")
    assert is_junk_mention("Naman Kabra (Founder)")
    assert is_junk_mention("Launch Captain (Naman)")
    assert is_junk_mention("Riya (Data)")
    assert is_junk_mention("Market Readiness Pod (Riya)")
    assert not is_junk_mention("Naman Kabra")


def test_is_junk_mention_allows_non_role_parentheticals() -> None:
    assert not is_junk_mention("Riya (Paris)")
    assert not is_junk_mention("Acme Ventures (Riya)")
    assert not is_junk_mention("Project Beacon (Pilot)")


def test_is_subsumed_fragment_rejects_single_name_prefix() -> None:
    known = {"Alex Kim", "Jordan Reyes", "Maya Singh"}
    assert is_subsumed_fragment("Alex", known)
    assert is_subsumed_fragment("Jordan", known)
    assert not is_subsumed_fragment("Alex Kim", known)


def test_possessive_rejected_with_chunk_name_context() -> None:
    chunk_names = {"Alex Kim", "Maya Singh", "Sam Okafor"}
    assert is_junk_mention("Alex Kim's", chunk_names)
    assert not is_junk_mention("Alex Kim", chunk_names)


def test_passes_quality_gate() -> None:
    assert passes_quality_gate("Alex Kim", PASSING_CONFIDENCE)
    assert not passes_quality_gate("GTM", PASSING_CONFIDENCE)
    assert not passes_quality_gate("Alex Kim", BELOW_THRESHOLD_CONFIDENCE)


def test_passes_quality_gate_accepts_exact_threshold() -> None:
    assert passes_quality_gate("Alex Kim", MIN_MENTION_CONFIDENCE)
    assert not passes_quality_gate("Alex Kim", BELOW_THRESHOLD_CONFIDENCE)


def test_build_resolved_entities_propagates_mention_confidence() -> None:
    resolution_map = ResolutionMap(
        doc_id="doc:test",
        org_id="org:test-org",
        entries={
            "Alex Kim": ResolutionMapEntry(
                mention_text="Alex Kim",
                entity_id=None,
                resolution_method="new",
                confidence=PASSING_CONFIDENCE,
            ),
        },
    )
    mentions = [
        MentionCandidate(
            mention_text="Alex Kim",
            mention_type="person",
            chunk_index=0,
            confidence=PASSING_CONFIDENCE,
        ),
    ]

    build_resolved_entities(resolution_map, mentions, "org:test-org")

    entity = resolution_map.resolved_entities[0]
    assert entity.is_new is True
    assert entity.confidence == PASSING_CONFIDENCE
    assert resolution_map.entries["Alex Kim"].entity_id == entity.entity_id
    assert resolution_map.entries["Alex Kim"].confidence == PASSING_CONFIDENCE


def test_build_resolved_entities_skips_junk() -> None:
    resolution_map = ResolutionMap(
        doc_id="doc:test",
        org_id="org:test-org",
        entries={
            "Alex Kim": ResolutionMapEntry(
                mention_text="Alex Kim",
                entity_id=None,
                resolution_method="new",
                confidence=PASSING_CONFIDENCE,
            ),
            "GTM": ResolutionMapEntry(
                mention_text="GTM",
                entity_id=None,
                resolution_method="new",
                confidence=PASSING_CONFIDENCE,
            ),
        },
    )
    mentions = [
        MentionCandidate(
            mention_text="Alex Kim",
            mention_type="person",
            chunk_index=0,
            confidence=PASSING_CONFIDENCE,
        ),
        MentionCandidate(
            mention_text="GTM",
            mention_type="object",
            chunk_index=0,
            confidence=PASSING_CONFIDENCE,
        ),
    ]

    build_resolved_entities(resolution_map, mentions, "org:test-org")

    assert len(resolution_map.resolved_entities) == 1
    assert resolution_map.resolved_entities[0].canonical_name == "Alex Kim"
    assert "GTM" not in resolution_map.entries or resolution_map.entries["GTM"].entity_id is None


def test_resolution_map_to_identity_result_sets_resolved_id_for_new() -> None:
    resolution_map = ResolutionMap(
        doc_id="doc:test",
        org_id="org:test-org",
        entries={
            "Alex Kim": ResolutionMapEntry(
                mention_text="Alex Kim",
                entity_id="entity:person:org-test-org:alex-kim",
                resolution_method="new",
                confidence=PASSING_CONFIDENCE,
            ),
        },
        resolved_entities=[],
    )
    resolution_map.resolved_entities.append(
        ResolvedEntity(
            entity_id="entity:person:org-test-org:alex-kim",
            canonical_name="Alex Kim",
            entity_type="person",
            is_new=True,
            confidence=PASSING_CONFIDENCE,
        )
    )

    identity = _resolution_map_to_identity_result(resolution_map)
    assert len(identity.identities) == 1
    assert identity.identities[0].resolved_id == "entity:person:org-test-org:alex-kim"
    assert identity.identities[0].is_new is True
    assert identity.identities[0].confidence == PASSING_CONFIDENCE


def test_format_resolved_entities_shows_new_entities_with_ids() -> None:
    identity = IdentityResolutionResult(
        identities=[
            ResolvedIdentity(
                mention="Alex Kim",
                entity_type="person",
                resolved_id="entity:person:org-dev:alex-kim",
                is_new=True,
                proposed_canonical_name="Alex Kim",
                confidence=PASSING_CONFIDENCE,
            ),
            ResolvedIdentity(
                mention="GTM",
                entity_type="object",
                resolved_id="entity:object:org-dev:gtm",
                is_new=True,
                proposed_canonical_name="GTM",
                confidence=PASSING_CONFIDENCE,
            ),
        ],
    )

    text = _format_resolved_entities(identity)
    assert "Alex Kim → entity:person:org-dev:alex-kim" in text
    assert "GTM" not in text


def test_format_for_prompt_filters_and_dedupes() -> None:
    resolution_map = ResolutionMap(
        doc_id="doc:test",
        org_id="org:test",
        entries={
            "Alex Kim": ResolutionMapEntry(
                mention_text="Alex Kim",
                entity_id="entity:person:org-dev:alex-kim",
                resolution_method="new",
                confidence=PASSING_CONFIDENCE,
            ),
            "Alex Kim's": ResolutionMapEntry(
                mention_text="Alex Kim's",
                entity_id="entity:person:org-dev:alex-kim",
                resolution_method="new",
                confidence=BELOW_THRESHOLD_CONFIDENCE,
            ),
            "GTM": ResolutionMapEntry(
                mention_text="GTM",
                entity_id="entity:object:org-dev:gtm",
                resolution_method="new",
                confidence=BELOW_THRESHOLD_CONFIDENCE,
            ),
        },
        resolved_entities=[],
    )

    prompt = resolution_map.format_for_prompt(min_confidence=MIN_MENTION_CONFIDENCE, dedupe=True)
    assert "Alex Kim → entity:person:org-dev:alex-kim" in prompt
    assert "Alex Kim's" not in prompt
    assert "GTM" not in prompt
    assert prompt.count("→") == 1


def test_llm_entity_cap_constant() -> None:
    assert MAX_LLM_ENTITIES_PER_CHUNK == 20


def test_prefetch_to_context_result_maps_nodes_and_topics() -> None:
    prefetched = {
        "subgraph": {
            "nodes": [
                {
                    "id": "decision:use-postgres",
                    "_type": "Decision",
                    "label": "Use Postgres",
                    "description": "Store sessions in Postgres",
                    "score": 0.95,  # should be filtered from key_fields
                },
            ],
            "edges": [],
        },
        "search_results": {
            "merged": [
                {
                    "id": "decision:use-postgres",
                    "_type": "Decision",
                    "label": "Use Postgres",
                },
                {
                    "id": "constraint:no-vendor-lock-in",
                    "_type": "Constraint",
                    "label": "No vendor lock-in",
                },
            ],
        },
        "traversal": {
            "results": [
                {
                    "node_id": "entity:person:org-dev:alex",
                    "nodes": [
                        {
                            "id": "insight:latency-spike",
                            "_type": "Insight",
                            "name": "Latency spike root cause",
                        },
                    ],
                    "edges": [],
                },
            ],
        },
    }
    scoring = ScoringResult(
        signal_score=0.8,
        signal_types=["decision"],
        reasoning="Contains architecture decisions.",
        suggested_topics=["database architecture"],
    )

    result = _prefetch_to_context_result(prefetched, scoring)

    assert len(result.discovered_nodes) == 3
    assert {n.node_id for n in result.discovered_nodes} == {
        "decision:use-postgres",
        "constraint:no-vendor-lock-in",
        "insight:latency-spike",
    }
    # score field must not leak into key_fields
    decision_node = next(n for n in result.discovered_nodes if n.node_id == "decision:use-postgres")
    assert "score" not in decision_node.key_fields

    assert len(result.topics) == 1
    assert result.topics[0].topic_label == "database architecture"
    assert "Prefetch-only context" in result.context_summary
    assert "Phase 2 LLM skipped" in result.extraction_guidance


def test_prefetch_to_context_result_topic_linkage_via_traversal_edges() -> None:
    """Topic node found in prefetch; traversal edges link it to decisions/constraints."""
    prefetched = {
        "subgraph": {
            "nodes": [
                {"id": "topic:database", "_type": "Topic", "label": "database architecture"},
                {"id": "decision:use-postgres", "_type": "Decision", "label": "Use Postgres"},
                {"id": "constraint:no-vendor-lock-in", "_type": "Constraint", "label": "No vendor lock-in"},
            ],
            "edges": [],
        },
        "search_results": {"merged": []},
        "traversal": {
            "results": [
                {
                    "node_id": "topic:database",
                    "nodes": [],
                    "edges": [
                        {"from": "topic:database", "to": "decision:use-postgres"},
                        {"from": "topic:database", "to": "constraint:no-vendor-lock-in"},
                    ],
                },
            ],
        },
    }
    scoring = ScoringResult(
        signal_score=0.8,
        signal_types=["decision"],
        reasoning="Database decisions.",
        suggested_topics=["database architecture"],
    )

    result = _prefetch_to_context_result(prefetched, scoring)

    assert len(result.topics) == 1
    topic = result.topics[0]
    assert topic.topic_id == "topic:database"
    assert "decision:use-postgres" in topic.related_decisions
    assert "constraint:no-vendor-lock-in" in topic.related_constraints


def test_prefetch_to_context_result_empty_prefetch() -> None:
    scoring = ScoringResult(
        signal_score=0.5,
        signal_types=["signal"],
        reasoning="Low signal document.",
        suggested_topics=[],
    )

    result = _prefetch_to_context_result(None, scoring)

    assert result.discovered_nodes == []
    assert result.topics == []
    assert result.context_summary == "No prefetch data available."
