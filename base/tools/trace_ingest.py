from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Any

from base.db.access import invalidate_scope_cache, trace_personal_scope
from base.db.client import db_pool
from base.tools.repository_refresh import normalize_repository_name


async def _ensure_trace_scope_membership(*, org_id: str, employee_id: str) -> None:
    """Upsert per-user trace scope registry + membership rows."""
    scope_id = trace_personal_scope(employee_id)
    pool = db_pool.get()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO access_scopes (id, org_id, source_provider, source_ref, source_label)
                VALUES ($1, $2, 'trace', $3, $4)
                ON CONFLICT (id) DO UPDATE SET
                    source_label = EXCLUDED.source_label
                """,
                scope_id,
                org_id,
                employee_id,
                f"Private trace scope for {employee_id}",
            )
            await conn.execute(
                """
                INSERT INTO channel_members (channel_id, employee_id, external_user_id, source_provider, crawled_at)
                VALUES ($1, $2, $3, 'trace', now())
                ON CONFLICT (channel_id, external_user_id) DO UPDATE SET
                    employee_id = EXCLUDED.employee_id,
                    source_provider = EXCLUDED.source_provider,
                    crawled_at = now()
                """,
                scope_id,
                employee_id,
                employee_id,
            )
    invalidate_scope_cache(employee_id)


def normalize_trace_entries(
    traces: list[dict[str, Any]],
    *,
    project: str,
    org_id: str,
    author_name: str,
    author_employee_id: str,
) -> list[dict[str, Any]]:
    """Backfill shared trace fields before writing to storage."""
    normalized: list[dict[str, Any]] = []
    for trace in traces:
        t = dict(trace)
        if not t.get("project"):
            t["project"] = project
        t.setdefault("org_id", org_id)
        if t.get("repo"):
            t["repo"], _ = normalize_repository_name(str(t["repo"]))

        raw_mentions = t.get("mentions") or []
        typed_mentions: list[dict[str, str]] = []
        for mention in raw_mentions:
            if not isinstance(mention, dict):
                continue
            mention_text = mention.get("mention_text")
            mention_type = mention.get("mention_type")
            if mention_text and mention_type:
                typed_mentions.append(
                    {"mention_text": str(mention_text), "mention_type": str(mention_type)}
                )

        if author_name and not any(
            mention.get("mention_text") == author_name
            and mention.get("mention_type") == "person"
            for mention in typed_mentions
        ):
            typed_mentions.append({"mention_text": author_name, "mention_type": "person"})
        t["mentions"] = typed_mentions

        keywords = t.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        if author_name and author_name not in keywords:
            keywords = list(keywords) + [author_name]
        t["keywords"] = keywords

        if author_name:
            t["_author_name"] = author_name
        if author_employee_id:
            t["_author_employee_id"] = author_employee_id

        normalized.append(t)
    return normalized


async def store_traces_s3(
    *,
    project: str,
    org_id: str,
    traces: list[dict[str, Any]],
    employee_id: str,
) -> dict[str, int]:
    """Store trace payloads in S3 for downstream processing."""
    bucket_set = os.environ.get("AWS_S3_BUCKET") or os.environ.get("PROCESSOR_AWS_S3_BUCKET")
    if not bucket_set:
        raise RuntimeError("S3 not configured — cannot store traces")

    from base.tools.s3 import ensure_bucket_async, write_json_async

    await ensure_bucket_async()
    await _ensure_trace_scope_membership(org_id=org_id, employee_id=employee_id)
    date_str = date.today().isoformat()
    stored, failed = 0, 0

    for trace in traces:
        try:
            trace_id = trace.get("trace_id") or str(uuid.uuid4())
            trace.setdefault("org_id", org_id)
            s3_key = f"traces/{org_id}/{project}/users/{employee_id}/raw/{date_str}/{trace_id}.json"
            await write_json_async(s3_key, trace)
            stored += 1
        except Exception:
            failed += 1

    return {"stored": stored, "failed": failed}
