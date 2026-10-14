"""
processor/config.py — Processing pipeline configuration.

All settings are model-agnostic. Switch providers by changing env vars only:

    # OpenAI (local default)
    PROCESSOR_SCORING_MODEL=openai:gpt-4o-mini
    PROCESSOR_ENTITY_MODEL=openai:gpt-4o-mini
    PROCESSOR_EXTRACTION_MODEL=openai:gpt-4o-mini
    PROCESSOR_LLM_TEMPERATURE=0.1
    OPENAI_API_KEY=sk-...

    # Claude
    PROCESSOR_SCORING_MODEL=anthropic:claude-3-5-haiku-20241022
    PROCESSOR_EXTRACTION_MODEL=anthropic:claude-sonnet-4-20250514
    ANTHROPIC_API_KEY=sk-ant-...

    # Google
    PROCESSOR_SCORING_MODEL=google-gla:gemini-1.5-flash
    PROCESSOR_EXTRACTION_MODEL=google-gla:gemini-1.5-pro
    GEMINI_API_KEY=...

Neo4j credentials are read directly by base/neo4j_client.py (not pydantic-settings):
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=<password>
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessorConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROCESSOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model configuration — provider:model-name format ──────────────────────
    # PydanticAI resolves the provider prefix automatically.
    scoring_model: str = "openai:gpt-4o-mini"
    entity_model: str = "openai:gpt-4o-mini"
    extraction_model: str = "openai:gpt-4o-mini"
    # Empty inherits extraction_model. Synthesis still requires the provider API key.
    repository_profile_model: str | None = None
    # Sampling temperature for all processor LLM calls (0 = deterministic, 1 = max variance).
    llm_temperature: float = 0.1

    # ── Scoring threshold ─────────────────────────────────────────────────────
    # Documents scoring below this value are marked 'skipped'.
    signal_score_threshold: float = 0.5

    # ── Polling ───────────────────────────────────────────────────────────────
    poll_interval_seconds: float = 15.0
    # Max docs to claim per scoring cycle (small model, can be generous)
    scoring_batch_size: int = 50
    # Max docs to claim per extraction cycle (larger model with tool use)
    extraction_batch_size: int = 10

    # ── Provider API keys ─────────────────────────────────────────────────────
    # Only the key for the active provider is required.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    # ── Embedding provider ────────────────────────────────────────────────────
    # "openai"    — use OpenAI text-embedding-3-large/small (default, no local models)
    # "inference" — use the inference-service (BGE-large / MiniLM); requires
    #               PROCESSOR_INFERENCE_URL to point at a running inference-service.
    embedding_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices("PROCESSOR_EMBEDDING_PROVIDER", "embedding_provider"),
    )

    # ── Inference service ─────────────────────────────────────────────────────
    # HTTP endpoint for the inference-service (MiniLM, BGE, GLiNER).
    # Only consulted when embedding_provider="inference".
    inference_url: str = Field(
        default="http://localhost:9111",
        validation_alias=AliasChoices("PROCESSOR_INFERENCE_URL", "inference_url"),
    )

    # ── Eidetix graph query API ───────────────────────────────────────────────
    # REST endpoint of the running webserver (used for graph queries —
    # search, traverse, context).  Writes go directly to Neo4j.
    eidetix_url: str = "http://localhost:8080"
    eidetix_api_key: str = ""  # Bearer token (API_KEY) — required in production
    # Portal gateway — used to read per-org Workspace LLM keys during extraction.
    portal_url: str = Field(
        default="",
        validation_alias=AliasChoices("PROCESSOR_PORTAL_URL", "portal_url"),
    )
    portal_token: str = Field(
        default="",
        validation_alias=AliasChoices("PROCESSOR_PORTAL_TOKEN", "portal_token"),
    )

    # ── Org identity ─────────────────────────────────────────────────────────
    # The org this processor instance belongs to.  Sent as X-Eidetix-Org on
    # all requests to the shared webserver so the webserver routes to the
    # correct env's Postgres + Neo4j.
    # Format: "org:acme-corp" or plain "acme-corp" (normalised server-side).
    # Falls back to ORGANISATION_ID for single-org backward compatibility.
    org_id: str = Field(
        default="",
        validation_alias=AliasChoices("PROCESSOR_ORG_ID", "ORGANISATION_ID", "org_id"),
    )

    # ── S3 (read only — writes handled by connectors) ─────────────────────────
    aws_s3_bucket: str = ""
    aws_region: str = "us-east-1"

    # ── Chunking params ─────────────────────────────────────────────────────────
    # "legacy"             — RecursiveCharacterTextSplitter (original behaviour)
    # "structural_entity"  — heading/paragraph structural split + spaCy sentence-
    #                        boundary sub-split for oversized sections (new default)
    chunking_strategy: str = "structural_entity"

    chunk_size: int = 5000
    chunk_overlap: int = 500
    chunk_min_size: int = 2000  # sections smaller than this are merged with neighbours
    chunk_summary_max_tokens: int = Field(
        default=1000,
        description="Max tokens for cumulative chunk summary."
    )

    # ── Model parallelism cap ───────────────────────────────────────────────────
    # Max concurrent model/CPU ops (GLiNER, embeddings). Prevents OOM.
    max_model_parallelism: int = 5

    # ── NER confidence thresholds ─────────────────────────────────────────────────
    # Default confidence assigned to spaCy NER outputs (spaCy doesn't emit scores).
    ner_spacy_confidence: float = 0.85
    # Minimum confidence to include NER output in downstream processing.
    ner_min_confidence: float = 0.89

    # ── Canonical resolution ────────────────────────────────────────────────────
    # Max concurrent Phase 3 LLM calls in canonical per-chunk extraction.
    extraction_chunk_concurrency: int = 4

    # ── Phase 2 context gathering ───────────────────────────────────────────────
    # When True, skip the Phase 2 LLM agent and build ContextGatheringResult
    # directly from the prefetch (gather_context_batch) response.
    skip_context_llm: bool = True

    # ── Trace linker ────────────────────────────────────────────────────────────
    trace_linker_model: str = "openai:gpt-4o-mini"
    trace_linker_max_iterations: int = 3


@lru_cache(maxsize=1)
def get_config() -> ProcessorConfig:
    return ProcessorConfig()


config = get_config()
