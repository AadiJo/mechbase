SELECTED_RESULTS_WIDGET_URI = "ui://mechbase/selected-results.html"

SELECTED_RESULTS_WIDGET_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: dark;
      --background: #000;
      --surface: #0b0b0b;
      --line: #2a2a2a;
      --muted: #a1a1aa;
      --text: #fff;
      --page: #e8e8e8;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--background);
      color: var(--text);
      font: 14px/1.4 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .component {
      min-width: 0;
      border-block: 1px solid var(--line);
      background: var(--background);
    }

    .component-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      padding: 12px 2px;
    }

    .component-head strong {
      font-size: 14px;
      letter-spacing: -0.01em;
    }

    .component-head span {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .rail {
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(238px, 32%);
      gap: 10px;
      overflow-x: auto;
      padding: 0 0 12px;
      scroll-snap-type: x mandatory;
      scrollbar-color: #444 transparent;
      scrollbar-width: thin;
    }

    .result {
      min-width: 0;
      padding: 0;
      overflow: hidden;
      scroll-snap-align: start;
      border: 1px solid var(--line);
      border-radius: 0;
      background: var(--surface);
      color: inherit;
      cursor: pointer;
      text-align: left;
    }

    .result:focus-visible {
      outline: 2px solid #fff;
      outline-offset: 2px;
    }

    .page-image {
      display: block;
      width: 100%;
      height: 280px;
      object-fit: contain;
      background: var(--page);
    }

    .result-copy {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
      padding: 10px 11px 11px;
      border-top: 1px solid var(--line);
    }

    .result-copy strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }

    .result-copy small {
      display: block;
      margin-top: 3px;
      overflow: hidden;
      color: var(--muted);
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .open-icon {
      color: var(--muted);
      font-size: 16px;
      line-height: 1;
    }

    .empty {
      padding: 14px 2px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }

    @media (max-width: 680px) {
      .rail { grid-auto-columns: minmax(238px, 78%); }
      .page-image { height: 250px; }
    }
  </style>
</head>
<body>
  <section class="component" aria-labelledby="component-title">
    <header class="component-head">
      <strong id="component-title">Selected mechanism images</strong>
      <span id="result-count"></span>
    </header>
    <div class="rail" id="results" aria-label="Selected FRC binder pages and figures"></div>
    <div class="empty" id="empty" hidden>No displayable images were selected.</div>
  </section>

  <script>
    const resultsElement = document.getElementById("results");
    const countElement = document.getElementById("result-count");
    const emptyElement = document.getElementById("empty");

    function metadata(item) {
      const parts = [];
      if (item.team) parts.push(`Team ${item.team}`);
      if (item.year) parts.push(String(item.year));
      if (item.asset_kind === "figure") {
        parts.push(`figure ${item.asset_id}`);
      } else {
        parts.push(`page ${item.page}`);
      }
      return parts.join(" · ");
    }

    function openResult(item) {
      if (window.openai?.openExternal) {
        window.openai.openExternal({ href: item.url, redirectUrl: false });
        return;
      }
      window.open(item.url, "_blank", "noopener,noreferrer");
    }

    function resultElement(item, index) {
      const button = document.createElement("button");
      button.className = "result";
      button.type = "button";
      button.addEventListener("click", () => openResult(item));

      const image = document.createElement("img");
      image.className = "page-image";
      image.src = item.image_url;
      image.alt = item.asset_kind === "figure"
        ? `${item.title} extracted figure`
        : `${item.title} technical binder page`;
      image.loading = index === 0 ? "eager" : "lazy";

      const copy = document.createElement("span");
      copy.className = "result-copy";

      const labels = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = item.title;
      const detail = document.createElement("small");
      detail.textContent = metadata(item);
      labels.append(title, detail);

      const openIcon = document.createElement("span");
      openIcon.className = "open-icon";
      openIcon.setAttribute("aria-hidden", "true");
      openIcon.textContent = "↗";

      copy.append(labels, openIcon);
      button.append(image, copy);
      return button;
    }

    function render(output) {
      const results = Array.isArray(output?.results) ? output.results : [];
      resultsElement.replaceChildren(...results.map(resultElement));
      const pageCount = results.filter((item) => item.asset_kind !== "figure").length;
      const figureCount = results.length - pageCount;
      const counts = [];
      if (pageCount) counts.push(`${pageCount} ${pageCount === 1 ? "page" : "pages"}`);
      if (figureCount) counts.push(`${figureCount} ${figureCount === 1 ? "figure" : "figures"}`);
      countElement.textContent = counts.join(" · ");
      emptyElement.hidden = results.length > 0;
    }

    if (window.openai?.toolOutput) render(window.openai.toolOutput);

    window.addEventListener("openai:set_globals", (event) => {
      const globals = event.detail?.globals ?? event.detail ?? {};
      if (Object.hasOwn(globals, "toolOutput")) render(globals.toolOutput);
    });

    window.addEventListener("message", (event) => {
      if (event.source !== window.parent) return;
      const message = event.data;
      if (message?.method === "ui/notifications/tool-result") {
        render(message.params?.structuredContent);
      }
    }, { passive: true });
  </script>
</body>
</html>
""".strip()
