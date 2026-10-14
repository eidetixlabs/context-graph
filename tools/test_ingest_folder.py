from __future__ import annotations

from tools.ingest_folder import iter_text_files, source_ref_for_path


def test_source_ref_sanitizes_relative_paths() -> None:
    assert source_ref_for_path("architecture/decisions.md") == "architecture-decisions.md"
    assert source_ref_for_path("../unsafe") == "unsafe"
    assert source_ref_for_path("notes with spaces.txt") == "notes-with-spaces.txt"


def test_iter_text_files_skips_binaries(tmp_path) -> None:
    (tmp_path / "keep.md").write_text("hello")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "also.txt").write_text("world")
    names = {p.relative_to(tmp_path).as_posix() for p in iter_text_files(tmp_path)}
    assert names == {"keep.md", "nested/also.txt"}
