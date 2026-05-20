import re
from pathlib import Path

from app.rag.models import SourceDoc


def parse_source(path: Path) -> SourceDoc:
    stem = path.stem
    year_match = re.search(r"(20\d{2})", stem)
    team_match = re.match(r"(\d+)", stem)
    return SourceDoc(
        path=path,
        team=team_match.group(1) if team_match else None,
        year=int(year_match.group(1)) if year_match else None,
        source_id=stem,
    )


def iter_pdfs(data_dir: Path) -> list[SourceDoc]:
    return [parse_source(path) for path in sorted(data_dir.glob("*.pdf"))]
