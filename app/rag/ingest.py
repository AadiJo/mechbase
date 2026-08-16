import argparse
import json
import shutil
from datetime import UTC, datetime
from hashlib import sha256
from itertools import islice
from pathlib import Path
from uuid import uuid4

from app.rag.artifacts import generation_namespace, source_artifact_root
from app.rag.config import Settings, get_settings
from app.rag.models import SourceDoc
from app.rag.pdf import extract_documents
from app.rag.sources import iter_pdfs
from app.rag.store import RagStore
from app.rag.voyage_client import VoyageEmbedder

EXTRACTION_SCHEMA_VERSION = "2"


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


def completed_sources(manifest_path: Path) -> dict[str, str | None]:
    if not manifest_path.exists():
        return {}
    done: dict[str, str | None] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            done[record["source"]] = record.get("ingestion_fingerprint")
        except (json.JSONDecodeError, KeyError):
            continue
    return done


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


def remove_superseded_artifacts(
    artifact_dir: Path,
    source_id: str,
    current_namespace: str,
) -> None:
    artifact_root = artifact_dir.resolve()
    root_path = source_artifact_root(artifact_root, source_id)
    if root_path.is_symlink():
        return
    root = root_path.resolve()
    try:
        root.relative_to(artifact_root)
    except ValueError:
        return
    if not root.exists():
        return
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir() or candidate.name == current_namespace:
            continue
        shutil.rmtree(candidate)


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
    manifest_path = settings.artifact_dir / "ingestion-manifest.jsonl"
    completed = {} if args.force else completed_sources(manifest_path)
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for source in sources:
            fingerprint = ingestion_fingerprint(source, settings)
            if completed.get(source.path.name) == fingerprint:
                print(f"Skipping {source.path.name}; already in manifest.", flush=True)
                continue
            print(f"Ingesting {source.path.name}...", flush=True)
            ingestion_id = uuid4().hex
            artifact_namespace = generation_namespace(source.source_version, ingestion_id)
            try:
                docs = extract_documents(source, settings, ingestion_id=ingestion_id)
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
                for batch_idx, batch in enumerate(batched(docs, args.batch_size), start=1):
                    print(f"  embedding text batch {batch_idx} ({len(batch)} objects)", flush=True)
                    text_vectors = embedder.embed_texts(
                        [doc.text or doc.source_pdf for doc in batch], "document"
                    )
                    image_vectors = multimodal_vectors(batch, embedder, settings)
                    store.upsert(batch, text_vectors, image_vectors)
                    print(f"  upserted batch {batch_idx}", flush=True)
                store.publish_source_generation(source.source_id, ingestion_id)
            except BaseException:
                try:
                    store.delete_source_generation(source.source_id, ingestion_id)
                except BaseException as cleanup_error:
                    print(
                        f"Could not remove failed generation {ingestion_id}: {cleanup_error}",
                        flush=True,
                    )
                remove_artifact_generation(
                    settings.artifact_dir,
                    source.source_id,
                    artifact_namespace,
                )
                raise
            store.mark_corpus_revision(ingestion_id)
            store.delete_superseded_source_generations(
                source.source_id,
                ingestion_id,
            )
            remove_superseded_artifacts(
                settings.artifact_dir,
                source.source_id,
                artifact_namespace,
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


if __name__ == "__main__":
    main()
