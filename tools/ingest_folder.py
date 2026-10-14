"""Index a local folder of markdown/text files into Context Graph.

Usage:
    python -m tools.ingest_folder ./examples
    python -m tools.ingest_folder ./docs --url http://localhost:8080 --api-key "$API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]+")


def source_ref_for_path(relative: str) -> str:
    cleaned = _UNSAFE.sub("-", relative.replace("\\", "/"))
    cleaned = cleaned.strip("-._:") or "document"
    if not re.match(r"^[A-Za-z0-9]", cleaned):
        cleaned = f"d{cleaned}"
    return cleaned[:200]


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files.append(path)
    return files


def ingest_file(
    *,
    url: str,
    api_key: str,
    path: Path,
    root: Path,
    doc_type: str,
) -> dict:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"{path} is empty")
    relative = path.relative_to(root).as_posix()
    payload = {
        "title": path.stem.replace("-", " ").replace("_", " "),
        "content": content,
        "doc_type": doc_type,
        "source_ref": source_ref_for_path(relative),
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/ingest/documents",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Eidetix-Org": os.environ.get("ORGANISATION_ID", "org:dev"),
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index a local folder into Context Graph")
    parser.add_argument("folder", type=Path, help="Directory of .md / .txt files")
    parser.add_argument("--url", default=os.environ.get("CONTEXT_GRAPH_URL", "http://localhost:8080"))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CONTEXT_GRAPH_API_KEY") or os.environ.get("API_KEY", ""),
    )
    parser.add_argument("--doc-type", default="context")
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1
    if not args.api_key:
        print("Set --api-key or API_KEY / CONTEXT_GRAPH_API_KEY", file=sys.stderr)
        return 1

    files = iter_text_files(folder)
    if not files:
        print(f"No .md/.txt files under {folder}", file=sys.stderr)
        return 1

    ok = 0
    failed = 0
    for path in files:
        try:
            result = ingest_file(
                url=args.url,
                api_key=args.api_key,
                path=path,
                root=folder,
                doc_type=args.doc_type,
            )
            status = result.get("status") or result.get("processing_state") or "accepted"
            print(f"{path.relative_to(folder)} -> {result.get('id', '?')} ({status})")
            ok += 1
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"{path.relative_to(folder)} FAILED HTTP {exc.code}: {detail}", file=sys.stderr)
            failed += 1
        except Exception as exc:
            print(f"{path.relative_to(folder)} FAILED: {exc}", file=sys.stderr)
            failed += 1

    print(f"Ingested {ok} file(s), {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
