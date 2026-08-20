from hashlib import sha256
from pathlib import Path


def source_artifact_root(artifact_dir: Path, source_id: str) -> Path:
    """Return an exact, collision-resistant directory for one logical source."""
    source_key = sha256(source_id.encode()).hexdigest()
    return artifact_dir / "sources" / source_key


def generation_namespace(source_version: str, ingestion_id: str | None) -> str:
    if not ingestion_id:
        return source_version
    return f"{source_version}~{ingestion_id}"
