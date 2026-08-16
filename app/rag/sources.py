import json
import re
from pathlib import Path

from app.rag.models import SourceDoc


def parse_source(path: Path, source_url: str | None = None) -> SourceDoc:
    stem = path.stem
    year_match = re.search(r"(20\d{2})", stem)
    team_match = re.match(r"(\d+)", stem)
    return SourceDoc(
        path=path,
        team=team_match.group(1) if team_match else None,
        year=int(year_match.group(1)) if year_match else None,
        source_id=stem,
        source_url=source_url,
    )


def iter_pdfs(data_dir: Path) -> list[SourceDoc]:
    source_urls = _load_source_urls(data_dir / "sources.json")
    return [
        parse_source(path, source_url=source_urls.get(path.name))
        for path in sorted(data_dir.glob("*.pdf"))
    ]


def _load_source_urls(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(filename, str) and isinstance(url, str) for filename, url in raw.items()
    ):
        raise ValueError("sources.json must map PDF filenames to original source URLs.")
    return raw
