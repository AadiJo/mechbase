import argparse
import json
from itertools import islice
from pathlib import Path

from app.rag.config import get_settings
from app.rag.pdf import extract_documents
from app.rag.sources import iter_pdfs
from app.rag.store import RagStore
from app.rag.voyage_client import VoyageEmbedder


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


def completed_sources(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    done = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["source"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FRC binder PDFs into Qdrant.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="Reingest sources even if manifest says done.")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = settings.data_dir if args.data_dir is None else settings.data_dir.__class__(args.data_dir)
    sources = iter_pdfs(data_dir)
    if args.limit:
        sources = list(islice(sources, args.limit))

    store = RagStore(settings)
    store.ensure_collection()
    embedder = VoyageEmbedder(settings)

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.artifact_dir / "ingestion-manifest.jsonl"
    completed = set() if args.force else completed_sources(manifest_path)
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for source in sources:
            if source.path.name in completed:
                print(f"Skipping {source.path.name}; already in manifest.", flush=True)
                continue
            print(f"Ingesting {source.path.name}...", flush=True)
            docs = extract_documents(source, settings)
            print(f"Extracted {len(docs)} retrieval objects from {source.path.name}.", flush=True)
            for batch_idx, batch in enumerate(batched(docs, args.batch_size), start=1):
                print(f"  embedding text batch {batch_idx} ({len(batch)} objects)", flush=True)
                text_vectors = embedder.embed_texts([doc.text or doc.source_pdf for doc in batch], "document")
                image_vectors = multimodal_vectors(batch, embedder, settings)
                store.upsert(batch, text_vectors, image_vectors)
                print(f"  upserted batch {batch_idx}", flush=True)
            manifest.write(json.dumps({"source": source.path.name, "documents": len(docs)}) + "\n")
            manifest.flush()
            print(f"Indexed {len(docs)} documents from {source.path.name}.", flush=True)


if __name__ == "__main__":
    main()
