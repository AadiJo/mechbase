# FRC Mechanism RAG

Retrieval-only multimodal RAG API for FRC technical binders. It indexes PDF text, OCR text,
rendered pages, extracted images, and image-linked context into Qdrant using Voyage embeddings.

## Stack

- Python/FastAPI API
- PyMuPDF + Tesseract OCR ingestion
- Voyage `voyage-4` text embeddings
- Voyage `voyage-multimodal-3.5` page/image embeddings
- Qdrant vector database
- Docker Compose deployment
- Static image/page artifact serving at `/images/...` (legacy `/artifacts/...` also works)

## Quick Start

```bash
cp .env.example .env
# set VOYAGE_API_KEY in .env; ingestion/search intentionally stop without it
docker compose up -d qdrant
docker compose run --rm ingest python -m app.rag.ingest --data-dir /app/data --limit 3
docker compose up api
```

Optionally add `data/sources.json` as a filename-to-URL map so ingestion can preserve original
HTTP or HTTPS source URLs.

Search:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Authorization: Bearer <mechbase-api-key>' \
  -H 'Content-Type: application/json' \
  -d '{"query":"multi ball shooter","top_k":10}'
```

All retrieval endpoints except `/health` require a Mechbase API key with the
`search:read` permission. Set `CONVEX_HTTP_URL` and `CONVEX_RECORDING_SECRET`
so the service can validate keys and record usage through Convex.

Search results include `artifact_url`, `linked_artifact_urls`, `page_context_url`, and
`page_text_url`. Use them when the agent needs to display images or fetch more page context.

## Hosted MCP

The API also exposes a stateless Streamable HTTP MCP server at `/mcp`. It is additive: the
frontend and all existing REST endpoints continue to use Mechbase API keys without changes.
The MCP endpoint accepts Clerk OAuth access tokens only.

The server provides six read-only tools:

- `search(...)` searches mechanisms with optional team, year, source, mechanism, sort, and
  result-count controls. Results include stable source IDs, evidence classification, calibrated
  score bands, coverage, and an abstention reason when no page is relevant enough.
- `inspect_candidates(ids)` returns labeled MCP image content and extracted page text so the
  model can check visual relevance before anything is displayed.
- `fetch(id)` returns full page text, source metadata, and absolute citation URLs.
- `find_similar(id, top_k)` finds related mechanism pages.
- `render_search_results(ids)` displays only model-selected pages in an inline image rail on MCP
  hosts that support MCP Apps. Other clients still receive its structured result.
- `list_sources(...)` lists indexed technical binders with exact metadata filters, source-name
  search, stable logical IDs plus content versions, provenance when known, and complete text and
  image coverage counts.

Every non-empty mechanism search uses `search` -> `inspect_candidates` ->
`render_search_results`, even when the user does not explicitly ask to inspect or display images.
Inspection images are model-only inputs. The render tool is the only path for displaying images,
so irrelevant RAG results do not appear just because they ranked highly.

Configure the resource server with:

```dotenv
MCP_PUBLIC_BASE_URL=https://api.example.com
MCP_OAUTH_SCOPES=openid
MCP_ALLOWED_ORIGINS=https://chatgpt.com,https://claude.ai
CLERK_OAUTH_ISSUER_URL=https://clerk.example.com
CLERK_SECRET_KEY=sk_live_...
SEARCH_MIN_SCORE=0.35
```

In the same Clerk instance, enable Dynamic client registration and set its default scope to
`openid`. Some MCP clients, including ChatGPT and Claude, omit scopes when they register. The
protected-resource discovery document is served at
`/.well-known/oauth-protected-resource/mcp`.

Inspect a running endpoint with the official MCP Inspector:

```bash
npx @modelcontextprotocol/inspector
```

Select Streamable HTTP, enter the `/mcp` URL, and complete the Clerk login flow. Verify
initialization, tool schemas, authorization failures, representative calls, and invalid inputs.

In ChatGPT developer mode, refresh the plugin connection after changing tool names,
descriptions, schemas, annotations, authentication, or UI resources. Then start a new chat.
Backend-only changes that preserve the advertised MCP metadata do not require a refresh.

Fetch a page image through the API/VPS:

```bash
curl -I http://localhost:8000/images/694-2020/page-019/page.png
```

Fetch all known context for a page:

```bash
curl http://localhost:8000/pages/694-2020.pdf/19
```

Fetch only text for a page:

```bash
curl http://localhost:8000/pages/694-2020.pdf/19/text
```

Run a tiny local smoke test without Qdrant/Voyage:

```bash
python -m venv .venv
. .venv/bin/activate
python -m ensurepip --upgrade --default-pip
pip install -e '.[dev]'
pytest -q -s
```

If you want to initialize Qdrant before ingestion:

```bash
curl -X POST http://localhost:8000/collections/init
```

Run the starter retrieval eval:

```bash
docker compose run --rm ingest python -m app.rag.eval \
  --eval-file /app/evals/frc_mechanism_eval.json --top-k 5
```

The eval includes positive recall cases and unrelated-query abstention cases. Tune
`SEARCH_MIN_SCORE` against the indexed corpus; the floor applies to the raw vector score before
the lexical ranking bonus.
