"""Unit tests for merging per-chunk extraction results."""

from __future__ import annotations

from processor.extraction.merge import merge_chunk_extractions
from processor.resolution.models import ChunkExtractionResult
from processor.schemas.extraction import EntityNode, ExtractionResult


def test_merge_preserves_entity_node_without_llm_edges() -> None:
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:org-nodeops:naman-k",
                label="Naman K",
                entity_type="person",
                canonical_name="Naman K",
                aliases='["Naman"]',
            ),
        ],
        edges=[],
    )

    merged = merge_chunk_extractions([
        ChunkExtractionResult(
            chunk_index=0,
            extraction=extraction,
            chunk_summary="Naman K appears in the document.",
        )
    ])

    assert [node.id for node in merged.nodes] == ["entity:person:org-nodeops:naman-k"]
    assert merged.edges == []
