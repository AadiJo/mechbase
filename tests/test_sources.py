from pathlib import Path

from app.rag.sources import parse_source


def test_parse_team_year_filename() -> None:
    source = parse_source(Path("data/254-2025.pdf"))
    assert source.team == "254"
    assert source.year == 2025
    assert source.source_id == "254-2025"


def test_parse_multi_part_filename() -> None:
    source = parse_source(Path("data/4607-2-2024.pdf"))
    assert source.team == "4607"
    assert source.year == 2024
