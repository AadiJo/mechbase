import json
from pathlib import Path

from app.rag.sources import iter_pdfs, parse_source


def test_parse_team_year_filename() -> None:
    source = parse_source(Path("data/254-2025.pdf"))
    assert source.team == "254"
    assert source.year == 2025
    assert source.source_id == "254-2025"


def test_parse_multi_part_filename() -> None:
    source = parse_source(Path("data/4607-2-2024.pdf"))
    assert source.team == "4607"
    assert source.year == 2024


def test_iter_pdfs_loads_original_source_urls(tmp_path: Path) -> None:
    pdf = tmp_path / "254-2025.pdf"
    pdf.touch()
    (tmp_path / "sources.json").write_text(
        json.dumps({pdf.name: "https://example.com/254-2025.pdf"}),
        encoding="utf-8",
    )

    sources = iter_pdfs(tmp_path)

    assert sources[0].source_url == "https://example.com/254-2025.pdf"
