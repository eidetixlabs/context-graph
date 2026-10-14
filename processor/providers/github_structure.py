"""
processor/providers/github_structure.py — GitHub repository structure processor.
"""

from __future__ import annotations

import json

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker


def _structure_payload_to_text(doc: DocumentRecord, content: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return content
    if not isinstance(payload, dict):
        return content

    repository = str(payload.get("repository") or "").strip()
    ref = str(payload.get("ref") or "").strip()
    total_files = payload.get("total_files")
    top_directories = payload.get("top_directories")
    top_extensions = payload.get("top_extensions")
    key_files = payload.get("key_files")

    title = doc.title or f"{repository} structure map".strip()
    lines = [f"# {title}"]
    if repository:
        lines.append(f"Repository: {repository}")
    if ref:
        lines.append(f"Ref: {ref}")
    if total_files is not None:
        lines.append(f"Total files: {total_files}")

    if isinstance(top_directories, list) and top_directories:
        lines.append("")
        lines.append("## Top Directories")
        for row in top_directories:
            if not isinstance(row, dict):
                continue
            directory = str(row.get("directory") or "").strip()
            count = row.get("file_count")
            if directory:
                lines.append(f"- {directory}: {count}")

    if isinstance(top_extensions, list) and top_extensions:
        lines.append("")
        lines.append("## Top Extensions")
        for row in top_extensions:
            if not isinstance(row, dict):
                continue
            extension = str(row.get("extension") or "").strip()
            count = row.get("file_count")
            if extension:
                lines.append(f"- {extension}: {count}")

    if isinstance(key_files, list) and key_files:
        lines.append("")
        lines.append("## Key Files")
        for file_path in key_files:
            value = str(file_path).strip()
            if value:
                lines.append(f"- {value}")

    return "\n".join(lines).strip() or content


class GithubStructureProcessorWorker(BaseProcessorWorker):
    PROVIDER = "github-structure"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        return _structure_payload_to_text(doc, content)

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        return _structure_payload_to_text(doc, content)

