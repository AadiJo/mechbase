# FRC Mechanism RAG API

Base URL for local development: `http://localhost:8000`

In production, replace the host with the VPS domain. URLs returned by the API are relative paths; clients should resolve them against the base URL.

Except for `/health`, endpoints require a Mechbase API key:

```http
Authorization: Bearer <mechbase-api-key>
```

Keys must include `search:read` by default. Set `REQUIRED_API_KEY_PERMISSION` to
change the required permission. Set `CONVEX_HTTP_URL` and
`CONVEX_RECORDING_SECRET` to record authenticated usage into Convex.

## Health

```http
GET /health
```

Returns service status and the active Qdrant collection.

## Search Mechanisms

```http
POST /search
Authorization: Bearer <mechbase-api-key>
Content-Type: application/json
```

Broad semantic retrieval over text chunks, rendered page images, and extracted mechanism images.

Request:

```json
{
  "query": "multi ball shooter",
  "top_k": 10,
  "team": "254",
  "year": 2024,
  "source": "254-2024.pdf",
  "modality": "text",
  "debug": false
}
```

Optional filters: `team`, `year`, `source`, `modality`. `modality` is one of `text`, `page_image`, `extracted_image`.

Each result includes:

- `text`: evidence snippet
- `source_pdf`, `team`, `year`, `page`
- `artifact_url`: direct URL if this result is an image/page artifact
- `linked_artifact_urls`: related page/mechanism image URLs
- `page_context_url`: fetch full page context
- `page_text_url`: fetch page text only

## List Sources

```http
GET /sources?team=254&year=2024&source=254-2024.pdf
```

Catalogs indexed PDFs. Filters are optional.

Response shape:

```json
{
  "sources": [
    {
      "source_pdf": "254-2024.pdf",
      "team": "254",
      "year": 2024,
      "pages": [1, 2, 3],
      "page_count": 22,
      "text_count": 18,
      "page_image_count": 22,
      "extracted_image_count": 18,
      "sample_image_urls": ["/images/254-2024/page-014/page.png"]
    }
  ]
}
```

## Get Source Summary

```http
GET /sources/{source_pdf}
```

Returns one source summary.

Example:

```http
GET /sources/254-2024.pdf
```

## Search Sources

```http
POST /sources/search
Content-Type: application/json
```

Source/page-level discovery. Use this when the agent wants to narrow by team/year/source/topic before fetching exact context.

Request:

```json
{
  "query": "254 drum shooter",
  "top_k": 5,
  "team": "254",
  "year": 2024,
  "source": "254-2024.pdf"
}
```

If `query` is omitted, this behaves like filtered source listing and returns candidate source/page entries.

Response:

```json
{
  "query": "254 drum shooter",
  "matches": [
    {
      "source_pdf": "254-2024.pdf",
      "team": "254",
      "year": 2024,
      "page": 14,
      "score": 0.703,
      "best_snippets": ["The Shooter ... uses feed rollers and quad-flywheels..."],
      "image_urls": ["/images/254-2024/page-014/page.png"],
      "page_context_url": "/pages/254-2024.pdf/14",
      "page_text_url": "/pages/254-2024.pdf/14/text"
    }
  ]
}
```

## Page Context

```http
GET /pages/{source_pdf}/{page}
```

Returns all known context for one PDF page: combined text, text chunks, rendered page URL, extracted image URLs, and retrieval object IDs.

Example:

```http
GET /pages/694-2020.pdf/19
```

## Page Text Only

```http
GET /pages/{source_pdf}/{page}/text
```

Returns only the combined text for one indexed page.

## Similar Pages

```http
GET /similar?result_id=254-2024_14_text_0&top_k=5
GET /similar?source_pdf=254-2024.pdf&page=14&top_k=5
```

Returns pages similar to a known result or page. Use this after finding one strong mechanism to branch to related mechanisms.

## Image Context

```http
GET /image-context?result_id=254-2024_14_image_0
GET /image-context?image_url=/images/254-2024/page-014/image-000.jpx
```

Returns surrounding page text and sibling image URLs for one image/result.

## Serve Images

```http
GET /images/{source_id}/page-{page}/page.png
GET /images/{source_id}/page-{page}/image-{index}.{ext}
```

Serves rendered pages and extracted images through the API/VPS. Legacy `/artifacts/...` URLs are also mounted, but new clients should use `/images/...`.

## Collection Init

```http
POST /collections/init
```

Creates the Qdrant collection if it does not exist. Usually only needed during setup.

## Recommended Agent Flow

1. Use `POST /search` for broad mechanism discovery.
2. Use `POST /sources/search` when narrowing by team/year/source/topic.
3. Use `GET /pages/{source_pdf}/{page}` for full evidence before detailed answers.
4. Use `/similar` to branch from one strong page to comparable mechanisms.
5. Use `/image-context` when the user or agent is focused on one image.
6. Show image URLs from `artifact_url`, `linked_artifact_urls`, `image_urls`, or `page_image_url`.
7. Cite `source_pdf`, `team`, `year`, and `page`.
