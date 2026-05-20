# Prompt For Adding The FRC Mechanism RAG As Agent Tools

Use this prompt with an implementation agent in the project where you want to add the RAG.

```text
You are adding FRC mechanism retrieval tools to an existing agent.

The RAG service is retrieval-only. It searches FRC technical binders and returns text evidence, source metadata, image URLs, and page-context URLs. The goal is high recall: the agent should be able to discover broadly, narrow by source/team/year, fetch exact page context, inspect images, and find similar mechanisms.

Base URL:
- Development: http://localhost:8000
- Production: use RAG_BASE_URL or the configured VPS URL.

Add these tools:

1. search_frc_mechanisms
Purpose: broad semantic mechanism discovery.
Endpoint: POST /search
Input:
- query: string, required
- top_k: integer, optional, default 10
- team: string, optional
- year: integer, optional
- source: string, optional PDF filename
- modality: string, optional: text, page_image, extracted_image
Output: ranked results with id, score, source_pdf, team, year, page, modality, text, artifact_url, linked_artifact_urls, page_context_url, page_text_url.

2. search_frc_sources
Purpose: narrow by team/year/source/topic before opening exact pages.
Endpoint: POST /sources/search
Input:
- query: string, optional
- top_k: integer, optional, default 10
- team: string, optional
- year: integer, optional
- source: string, optional PDF filename
Output: grouped source/page matches with snippets, image_urls, page_context_url, page_text_url.

3. list_frc_sources
Purpose: catalog indexed PDFs and resolve ambiguity like “254 shooter” across years.
Endpoint: GET /sources
Input:
- team: string, optional
- year: integer, optional
- source: string, optional
Output: source summaries with page counts and sample images.

4. get_frc_source_summary
Purpose: orient inside one binder.
Endpoint: GET /sources/{source_pdf}
Input:
- source_pdf: string
Output: source summary.

5. get_frc_page_context
Purpose: exact drill-down for one page.
Endpoint: GET /pages/{source_pdf}/{page}
Input:
- source_pdf: string
- page: integer
Output: full page text, text_chunks, page_image_url, image_urls, result_ids.

6. get_frc_page_text
Purpose: text-only exact drill-down.
Endpoint: GET /pages/{source_pdf}/{page}/text
Input:
- source_pdf: string
- page: integer
Output: page text.

7. find_similar_frc_pages
Purpose: branch from one good result/page to comparable mechanisms.
Endpoint: GET /similar
Input:
- result_id: string, optional
- source_pdf: string, optional
- page: integer, optional
- top_k: integer, optional, default 10
Rules: provide either result_id or source_pdf + page.
Output: seed info and ranked similar pages.

8. get_frc_image_context
Purpose: explain or inspect one retrieved image.
Endpoint: GET /image-context
Input:
- result_id: string, optional
- image_url: string, optional
Rules: provide either result_id or image_url.
Output: surrounding page text, page_image_url, sibling image_urls, source metadata.

Implementation rules:
- Resolve relative URLs against RAG_BASE_URL before exposing them to the user or another model.
- Use timeouts. On network/API failure, return a clear tool error that retrieval is unavailable.
- Deduplicate page-context fetches by (source_pdf, page).
- Preserve source_pdf, team, year, and page in tool outputs.

Recommended reasoning flow:
- For broad questions like “What teams have the best ball shooters?”, call search_frc_mechanisms with top_k 10-20 and maybe source search if results are scattered.
- For follow-up questions in another chat like “What’s special about 254’s drum shooter?”, start with search_frc_sources(query="254 drum shooter", team="254"). If the year is unknown, do not guess; compare returned years/pages.
- Once a likely page appears, call get_frc_page_context before giving detailed claims.
- If one page is strong, call find_similar_frc_pages to find related mechanisms across teams/years.
- If the user asks about an image or a result has a useful image_url, call get_frc_image_context.
- If a query misses, retry with synonyms before saying the corpus lacks it.

Synonym hints:
- ball shooter: power cell shooter, cargo shooter, note shooter, flywheel shooter, backspin shooter, hooded shooter
- drum shooter: flywheel, roller, wheel, hood, launcher
- intake: collector, floor pickup, roller intake, feeder intake
- indexer: serializer, conveyor, hopper, magazine
- climber: hang, cage, chain, winch, trap
- end effector: manipulator, grabber, wrist, scorer

Answering rules:
- Cite team, year, source_pdf, and page.
- Include image links when helpful.
- Group similar designs together rather than dumping raw results.
- State uncertainty if results are ambiguous across years.
```
