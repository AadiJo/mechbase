import json
from pathlib import Path

import pytest

from app.rag.ingest import completed_sources
from app.rag.sources import iter_pdfs, parse_source


def test_parse_team_year_filename(tmp_path: Path) -> None:
    path = tmp_path / "254-2025.pdf"
    path.write_bytes(b"binder-version-one")
    source = parse_source(path)
    assert source.team == "254"
    assert source.year == 2025
    assert source.source_id == "254-2025"
    assert source.source_version_id.startswith("254-2025@")


def test_parse_multi_part_filename(tmp_path: Path) -> None:
    path = tmp_path / "4607-2-2024.pdf"
    path.write_bytes(b"binder")
    source = parse_source(path)
    assert source.team == "4607"
    assert source.year == 2024


def test_source_version_changes_with_pdf_content(tmp_path: Path) -> None:
    path = tmp_path / "254-2025.pdf"
    path.write_bytes(b"first")
    first = parse_source(path)
    path.write_bytes(b"second")
    second = parse_source(path)

    assert first.source_id == second.source_id
    assert first.source_version != second.source_version
    assert first.source_version_id != second.source_version_id


def test_iter_pdfs_loads_original_source_urls(tmp_path: Path) -> None:
    pdf = tmp_path / "254-2025.pdf"
    pdf.touch()
    (tmp_path / "sources.json").write_text(
        json.dumps({pdf.name: "https://example.com/254-2025.pdf"}),
        encoding="utf-8",
    )

    sources = iter_pdfs(tmp_path)

    assert sources[0].source_url == "https://example.com/254-2025.pdf"


def test_completed_sources_tracks_latest_content_version(tmp_path: Path) -> None:
    manifest = tmp_path / "ingestion-manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"source": "254-2025.pdf", "source_version": "old"}),
                json.dumps({"source": "254-2025.pdf", "source_version": "new"}),
            ]
        ),
        encoding="utf-8",
    )

    assert completed_sources(manifest) == {"254-2025.pdf": "new"}


def test_iter_pdfs_rejects_non_http_source_urls(tmp_path: Path) -> None:
    pdf = tmp_path / "254-2025.pdf"
    pdf.touch()
    (tmp_path / "sources.json").write_text(
        json.dumps({pdf.name: "javascript:alert(1)"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        iter_pdfs(tmp_path)
