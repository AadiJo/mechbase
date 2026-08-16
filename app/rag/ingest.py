import argparse
import fcntl
import json
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from itertools import islice
from pathlib import Path
from uuid import uuid4

from app.rag.artifacts import generation_namespace, source_artifact_root
from app.rag.config import Settings, get_settings
from app.rag.models import RagDocument, SourceDoc
from app.rag.pdf import extract_documents
from app.rag.sources import iter_pdfs
from app.rag.store import RagStore
from app.rag.voyage_client import VoyageEmbedder

EXTRACTION_SCHEMA_VERSION = "2"
ARTIFACT_COMPLETE_FILE = ".complete.json"
STAGING_NAMESPACE_PATTERN = re.compile(r"~[0-9a-f]{32}$")


def batched(items, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def multimodal_vectors(batch, embedder: VoyageEmbedder, settings):
    vectors: list[list[float] | None] = [None] * len(batch)
    indexed_docs = [(idx, doc) for idx, doc in enumerate(batch) if doc.modality != "text"]
    for mm_batch in batched(indexed_docs, settings.multimodal_batch_size):
        print(f"  embedding multimodal sub-batch ({len(mm_batch)} image/page objects)", flush=True)
        embeddings = embedder.embed_multimodal(
            [doc.text or doc.source_pdf for _, doc in mm_batch],
            [doc.artifact_path for _, doc in mm_batch],
            "document",
        )
        for (idx, _), embedding in zip(mm_batch, embeddings, strict=True):
            vectors[idx] = embedding
    zero = [0.0] * settings.embedding_dim
    return [vector if vector is not None else zero for vector in vectors]


def _completed_source_records(
    manifest_path: Path,
) -> dict[str, tuple[str | None, str | None]]:
    if not manifest_path.exists():
        return {}
    done: dict[str, tuple[str | None, str | None]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            done[record["source"]] = (
                record.get("ingestion_fingerprint"),
                record.get("ingestion_id"),
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def completed_sources(manifest_path: Path) -> dict[str, str | None]:
    return {
        source: fingerprint
        for source, (fingerprint, _ingestion_id) in _completed_source_records(manifest_path).items()
    }


def ingestion_fingerprint(source: SourceDoc, settings: Settings) -> str:
    parameters = {
        "schema": EXTRACTION_SCHEMA_VERSION,
        "source_version": source.source_version,
        "source_url": source.source_url,
        "ocr_min_chars_per_page": settings.ocr_min_chars_per_page,
        "render_dpi": settings.render_dpi,
        "chunk_target_chars": settings.chunk_target_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "text_model": settings.text_model,
        "multimodal_model": settings.multimodal_model,
        "embedding_dim": settings.embedding_dim,
        "max_embed_image_side": settings.max_embed_image_side,
    }
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:16]


def artifact_fingerprint(source: SourceDoc, settings: Settings) -> str:
    parameters = {
        "schema": EXTRACTION_SCHEMA_VERSION,
        "source_version": source.source_version,
        "render_dpi": settings.render_dpi,
    }
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:16]


def publish_artifact_generation(
    artifact_dir: Path,
    source_id: str,
    staging_namespace: str,
    published_namespace: str,
) -> str:
    artifact_root = artifact_dir.resolve()
    root_path = source_artifact_root(artifact_root, source_id)
    if root_path.is_symlink():
        raise RuntimeError(f"Artifact root for {source_id} cannot be a symlink.")
    root = root_path.resolve()
    try:
        root.relative_to(artifact_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Artifact root for {source_id} escapes the artifact directory."
        ) from exc
    staging_path = root / staging_namespace
    published_path = root / published_namespace
    if staging_path.is_symlink() or published_path.is_symlink():
        raise RuntimeError("Artifact generations cannot be symlinks.")
    staging = staging_path.resolve()
    published = published_path.resolve()
    if staging.parent != root or published.parent != root:
        raise RuntimeError("Artifact namespace escapes its source directory.")
    if not staging.is_dir():
        raise RuntimeError(f"Artifact staging generation {staging_namespace} is missing.")
    expected_inventory = _write_artifact_completion_marker(staging)
    _fsync_tree(staging)
    if not published.exists():
        staging.replace(published)
        for directory in [root, root.parent, artifact_root]:
            _fsync_directory(directory)
        return published_namespace
    if published.is_dir() and _artifact_tree_matches(published, expected_inventory):
        shutil.rmtree(staging)
        _fsync_directory(root)
        return published_namespace

    inventory_digest = sha256(
        json.dumps(expected_inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    candidate_namespace = f"{published_namespace}~repair-{inventory_digest}"
    candidate_index = 0
    while True:
        candidate_path = root / candidate_namespace
        if candidate_path.is_symlink():
            raise RuntimeError("Published artifact generation cannot be a symlink.")
        if not candidate_path.exists():
            published = candidate_path
            break
        if candidate_path.is_dir() and _artifact_tree_matches(candidate_path, expected_inventory):
            shutil.rmtree(staging)
            _fsync_directory(root)
            return candidate_namespace
        candidate_index += 1
        candidate_namespace = f"{published_namespace}~repair-{inventory_digest}-{candidate_index}"

    staging.replace(published)
    for directory in [root, root.parent, artifact_root]:
        _fsync_directory(directory)
    return candidate_namespace


def _artifact_inventory(root: Path) -> list[dict[str, int | str]]:
    marker = root / ARTIFACT_COMPLETE_FILE
    inventory = []
    for path in sorted(root.rglob("*")):
        if path == marker:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Artifact tree cannot contain symlinks: {path}")
        if path.is_file():
            digest = sha256()
            with path.open("rb") as artifact:
                while chunk := artifact.read(1024 * 1024):
                    digest.update(chunk)
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    return inventory


def _write_artifact_completion_marker(root: Path) -> dict[str, list[dict[str, int | str]]]:
    inventory = {"files": _artifact_inventory(root)}
    marker = root / ARTIFACT_COMPLETE_FILE
    marker.write_text(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return inventory


def _artifact_tree_matches(
    root: Path,
    expected_inventory: dict[str, list[dict[str, int | str]]],
) -> bool:
    marker = root / ARTIFACT_COMPLETE_FILE
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        actual = {"files": _artifact_inventory(root)}
        return recorded == expected_inventory == actual
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError):
        return False


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Artifact tree cannot contain symlinks: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as artifact:
                os.fsync(artifact.fileno())
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def with_published_artifacts(
    document: RagDocument,
    staging_root: Path,
    published_root: Path,
) -> RagDocument:
    def published_path(path: str) -> str:
        try:
            relative = Path(path).relative_to(staging_root)
        except ValueError as exc:
            raise RuntimeError(f"Artifact path {path} is outside its staging generation.") from exc
        return str(published_root / relative)

    return document.model_copy(
        update={
            "artifact_path": (
                published_path(document.artifact_path) if document.artifact_path else None
            ),
            "linked_artifacts": [published_path(path) for path in document.linked_artifacts],
        }
    )


def remove_artifact_generation(artifact_dir: Path, source_id: str, namespace: str) -> None:
    artifact_root = artifact_dir.resolve()
    root_path = source_artifact_root(artifact_root, source_id)
    if root_path.is_symlink():
        return
    root = root_path.resolve()
    try:
        root.relative_to(artifact_root)
    except ValueError:
        return
    target_path = root / namespace
    if target_path.is_symlink():
        return
    target = target_path.resolve()
    if target.parent != root or not target.is_dir():
        return
    shutil.rmtree(target)


def remove_abandoned_artifact_staging(artifact_dir: Path) -> None:
    artifact_root = artifact_dir.resolve()
    sources_root = artifact_root / "sources"
    if sources_root.is_symlink() or not sources_root.is_dir():
        return
    for source_root in sources_root.iterdir():
        if source_root.is_symlink() or not source_root.is_dir():
            continue
        removed = False
        for candidate in source_root.iterdir():
            if (
                candidate.is_symlink()
                or not candidate.is_dir()
                or not STAGING_NAMESPACE_PATTERN.search(candidate.name)
            ):
                continue
            shutil.rmtree(candidate)
            removed = True
        if removed:
            _fsync_directory(source_root)


@contextmanager
def ingestion_lock(artifact_dir: Path) -> Iterator[None]:
    lock_path = artifact_dir / "ingestion.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another ingestion process is already running.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def ingest_sources(
    sources: list[SourceDoc],
    *,
    batch_size: int,
    force: bool,
    settings: Settings,
    store: RagStore,
    embedder: VoyageEmbedder,
) -> None:
    manifest_path = settings.artifact_dir / "ingestion-manifest.jsonl"
    completed = {} if force else _completed_source_records(manifest_path)
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for source in sources:
            fingerprint = ingestion_fingerprint(source, settings)
            store.initialize_active_generation(source.source_id)
            completed_record = completed.get(source.path.name)
            if completed_record and completed_record[0] == fingerprint:
                recorded_ingestion_id = completed_record[1]
                active_ingestion_id = store.active_generations().get(source.source_id)
                if recorded_ingestion_id is None or recorded_ingestion_id == active_ingestion_id:
                    print(f"Skipping {source.path.name}; already in manifest.", flush=True)
                    continue
            print(f"Ingesting {source.path.name}...", flush=True)
            ingestion_id = uuid4().hex
            staging_namespace = generation_namespace(source.source_version, ingestion_id)
            published_namespace = generation_namespace(
                source.source_version,
                artifact_fingerprint(source, settings),
            )
            source_root = source_artifact_root(settings.artifact_dir, source.source_id)
            staging_root = source_root / staging_namespace
            committed = False
            try:
                docs = extract_documents(
                    source,
                    settings,
                    ingestion_id=ingestion_id,
                    artifact_namespace=staging_namespace,
                )
                if not docs:
                    raise RuntimeError(
                        f"Refusing to replace {source.path.name}: extraction produced no documents."
                    )
                ingested_at = datetime.now(UTC).isoformat()
                docs = [doc.model_copy(update={"ingested_at": ingested_at}) for doc in docs]
                print(
                    f"Extracted {len(docs)} retrieval objects from {source.path.name}.",
                    flush=True,
                )
                artifact_paths = {
                    path
                    for document in docs
                    for path in [document.artifact_path, *document.linked_artifacts]
                    if path
                }
                missing_artifacts = [path for path in artifact_paths if not Path(path).is_file()]
                if missing_artifacts:
                    raise RuntimeError(
                        f"Artifact extraction left {len(missing_artifacts)} referenced files missing."
                    )
                actual_published_namespace = published_namespace
                if artifact_paths:
                    actual_published_namespace = publish_artifact_generation(
                        settings.artifact_dir,
                        source.source_id,
                        staging_namespace,
                        published_namespace,
                    )
                published_root = source_root / actual_published_namespace
                published_docs = [
                    with_published_artifacts(document, staging_root, published_root)
                    for document in docs
                ]
                for batch_idx, batch in enumerate(batched(published_docs, batch_size), start=1):
                    print(f"  embedding text batch {batch_idx} ({len(batch)} objects)", flush=True)
                    text_vectors = embedder.embed_texts(
                        [doc.text or doc.source_pdf for doc in batch], "document"
                    )
                    image_vectors = multimodal_vectors(batch, embedder, settings)
                    store.upsert(batch, text_vectors, image_vectors)
                    print(f"  upserted batch {batch_idx}", flush=True)
                store.publish_source_generation(source.source_id, ingestion_id)
                store.set_active_generation(source.source_id, ingestion_id)
                committed = True
            except BaseException:
                if not committed:
                    try:
                        pointer_committed = (
                            store.active_generations().get(source.source_id) == ingestion_id
                        )
                    except BaseException:
                        # A failed directory fsync may still follow a successful pointer rename.
                        # Preserve the generation unless rollback is known to be safe.
                        pointer_committed = True
                    if not pointer_committed:
                        try:
                            store.delete_source_generation(source.source_id, ingestion_id)
                        except BaseException as cleanup_error:
                            print(
                                f"Could not remove failed generation {ingestion_id}: "
                                f"{cleanup_error}",
                                flush=True,
                            )
                        remove_artifact_generation(
                            settings.artifact_dir,
                            source.source_id,
                            staging_namespace,
                        )
                raise

            store.retire_superseded_source_generations(
                source.source_id,
                ingestion_id,
            )
            store.delete_superseded_source_generations(
                source.source_id,
                ingestion_id,
            )
            manifest.write(
                json.dumps(
                    {
                        "source": source.path.name,
                        "source_id": source.source_id,
                        "source_version": source.source_version,
                        "ingestion_id": ingestion_id,
                        "ingestion_fingerprint": fingerprint,
                        "documents": len(docs),
                    }
                )
                + "\n"
            )
            manifest.flush()
            print(f"Indexed {len(docs)} documents from {source.path.name}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FRC binder PDFs into Qdrant.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--force", action="store_true", help="Reingest sources even if manifest says done."
    )
    args = parser.parse_args()

    settings = get_settings()
    data_dir = (
        settings.data_dir if args.data_dir is None else settings.data_dir.__class__(args.data_dir)
    )
    sources = iter_pdfs(data_dir)
    if args.limit:
        sources = list(islice(sources, args.limit))

    store = RagStore(settings)
    store.ensure_collection()
    embedder = VoyageEmbedder(settings)

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    with ingestion_lock(settings.artifact_dir):
        remove_abandoned_artifact_staging(settings.artifact_dir)
        ingest_sources(
            sources,
            batch_size=args.batch_size,
            force=args.force,
            settings=settings,
            store=store,
            embedder=embedder,
        )


if __name__ == "__main__":
    main()
