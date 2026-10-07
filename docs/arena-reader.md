# Arena reader

`/arena/feed` reads the saved links in `content/are.na.md` and the public saved-link index for [Aaron's Curius account](https://curius.app/aaron-pham). Quartz emits the page shell and `static/arena-feed.json`; the existing Cloudflare Worker merges Curius and owns authentication, read marks, notes, and article capture.

## Curius catalogue

The Worker imports the complete `/api/users/3584/searchLinks` index, independently of the 30-link pages used by the Curius profile. It stores the validated index in the existing private R2 bucket and checks for updates when the feed is requested, at most once every 15 minutes. Reading, notes and resource requests reuse that snapshot. A failed refresh retains the previous index and waits a minute before retrying; an initial failure returns an explicit error so an incomplete catalogue cannot silently replace the queue.

Arena and Curius links share the existing normalized-URL article identity. Tracking parameters (including `curius` and `utm_*`) are removed before duplicate matching. Content parameters, URL fragments, trailing slashes and HTTP/HTTPS distinctions are preserved. A duplicate occupies one queue entry, keeping Arena's authored title, tags and saved notes alongside every Curius link ID. Curius-only entries are labeled `curius`, which also works as a queue search term. The Curius index supplies link metadata; its full-text snippets, highlights and comments are not imported as reader notes.

Curius `toRead: true` joins the Later queue. A false or unset value does not mark anything read. Existing read marks, saved copies and reader notes keep their article IDs, including when a Curius-only link is subsequently saved in Arena. Reader notes for Curius-only links have a source URL and a null Arena occurrence, so exports never invent a Markdown backfill target. The reader does not write to Curius.

## Reading and notes

The default queue shuffles unread `later: true` links first, then the remaining unread links. The shuffle seed, skipped links, selected article, and reading position stay on the current device. Only an explicit read action changes completion in D1. Saving a note or opening a source leaves completion unchanged.

The feed excludes video entries, the `video` channel, YouTube and Vimeo links, and direct video files. Watch URLs with a YouTube video ID are also excluded when their saved hostname has a typo. Filtering happens before the API returns the queue, so counts, search, and shuffle share the same entries. The full catalogue, saved snapshots, notes, and read marks remain available.

Text notes open in a modal drawer on phones and a side panel on larger screens. Drafts are written to IndexedDB before server acknowledgement. Saved notes and read marks belong to the authenticated account. Revision checks prevent a stale device from overwriting a newer edit. Conflicts keep the local draft available for reconciliation.

Selecting text in the reader can create a note with a quotation, surrounding text, and the saved article version. HTML articles use one extracted reading view. PDFs use the existing PDF viewer. Unsupported, inaccessible, or incomplete sources retain an Open original link.

Wikipedia article links use the garden's existing summary popovers on hover and keyboard focus. The reader derives preview metadata from the sanitized destination URL, including mobile Wikipedia links. Escape dismisses the preview; clicking still opens the original link.

### Keyboard shortcuts

The reading view exposes these shortcuts on its action buttons:

| Key       | Action                                                  |
| --------- | ------------------------------------------------------- |
| `q`       | Toggle the queue                                        |
| `Shift+N` | Toggle notes, or start a note from the selected passage |
| `r`       | Mark an unread link read and advance                    |
| `Shift+O` | Open the original link in a new tab                     |
| `n`       | Go to the next link without changing its read status    |

Shortcuts pause while typing in search or notes, using a menu, or holding Control, Command, or Alt. Holding a shortcut key does not repeat its action. Read and next wait for an in-progress read save to finish. The notes inbox keeps its own controls.

## Access and development

Arena's build-time parsing, page emission, feed and search manifests, and component resources are skipped in ordinary watch and serve builds, using the same `watch && !force` condition as the LLM corpus. Use `pnpm swarm --force` to enable Arena for local development. Production builds include Arena. Skipped incremental updates preserve existing Arena output.

The reader uses the existing GitHub comments login. The OAuth callback issues a separate signed, HttpOnly owner session with an immutable `github:<numeric id>` subject. `ARENA_OWNER_LOGIN` defaults to `aarnphm`; `SESSION_SECRET` signs both the session and resource capabilities. The flashcards API resolves its login from the same owner session, so only the owner's reviews are scheduled. Keep the secret in Cloudflare Secrets or local environment files.

For local development, add this setting to the ignored `.env.local`:

```dotenv
ARENA_READER_DEV=true
```

This setting only permits requests whose URL hostname is `localhost`, `127.0.0.1`, or `::1`. It creates a separate `dev:aarnphm` identity. Use the Worker port, normally `http://localhost:8080/arena/feed`; the Quartz-only server does not serve the reader API. An already running Wrangler process must load the local environment setting before it takes effect.

The required bindings are:

| Binding                     | Responsibility                                            |
| --------------------------- | --------------------------------------------------------- |
| `ASSETS`                    | Feed shell and fixed catalogue asset                      |
| `ARENA_READER`              | D1 read marks and text notes                              |
| `ARENA_CONTENT`             | Private R2 article versions, images/PDFs and Curius index |
| `BROWSER`                   | Anonymous Cloudflare browser capture                      |
| `ARENA_RENDER_RATE_LIMITER` | Browser launch limit per authenticated reader             |

The Browser binding uses Cloudflare during local development, so local cache misses use real browser time. Reader D1 and R2 stay local unless their commands explicitly target remote storage. The configured development host is `localhost`, which keeps the loopback identity and origin checks consistent with the local request. See [Cloudflare's Browser Run development reference](https://developers.cloudflare.com/browser-run/reference/wrangler/).

Apply only the reader migration during reader setup:

```sh
pnpm exec wrangler d1 migrations apply ARENA_READER --local
pnpm exec wrangler d1 migrations apply ARENA_READER --remote
```

The remote command changes the configured reader database. The normal deployment migration scripts include it. Provisioning storage and applying a migration do not deploy the Worker or publish a new Quartz build.

### Wrangler setup

`wrangler.toml` records the provisioned database IDs and migration directories for `COMMENTS_ROOM`, `FLASHCARDS`, and `ARENA_READER`. Use the repository's installed Wrangler to inspect them and apply all configured migrations:

```sh
pnpm exec wrangler d1 list
pnpm db:migrate:local
pnpm db:migrate
```

The migration scripts run each database sequentially. Local D1 commands share Wrangler's runtime state and can contend for a SQLite lock when run concurrently. Repeating these scripts applies only outstanding migrations.

Rate limits are declared in `wrangler.toml`:

| Binding                     | Namespace | Limit                              | Key                          |
| --------------------------- | --------- | ---------------------------------- | ---------------------------- |
| `MCP_RATE_LIMITER`          | `1001`    | 120 requests per 60 seconds        | Client IP                    |
| `ARENA_RENDER_RATE_LIMITER` | `1002`    | 10 browser launches per 60 seconds | Authenticated reader subject |

Cloudflare creates these bindings with the Worker version; there is no separate rate-limit namespace creation command. Keep namespace IDs distinct and stable because Workers in the same account that use the same namespace share counters. Limits apply per Cloudflare location and are eventually consistent. They reduce bursts of browser work; a strict global browser spending cap would require separate accounting. Saved article responses do not consume the browser-launch allowance. See [Cloudflare's rate-limiting binding reference](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/).

Validate the bindings without publishing:

```sh
pnpm exec wrangler deploy --dry-run
```

A normal Worker deployment activates the bindings along with its code and configured site assets. Verify the active version after deployment:

```sh
pnpm exec wrangler deployments status --json
pnpm exec wrangler versions view <active-version-id>
```

The active version should list all three D1 bindings and both rate limiters. A successful dry-run validates configuration and bundling; it does not activate a new version. See [Cloudflare's versions and deployments reference](https://developers.cloudflare.com/workers/versions-and-deployments/).

## Capture and persistent storage

Opening a link checks R2 first. A usable saved copy is returned without starting Chromium. First HTML visits launch a bounded browser session, then extract and sanitize the resulting document in a separate context. Source requests pass through the Worker's public-address fetch boundary. Publisher cookies, authorization headers, source forms, and publisher scripts are never installed in the reader page.

Every HTML capture and resource request identifies itself as `GardenArenaReader/1.0 (https://aarnphm.xyz/arena)`, including redirects, extractor API requests, and image delivery. This follows [Defuddle's identified fetch setup](https://github.com/kepano/defuddle/blob/main/src/fetch.ts) and [Wikimedia's User-Agent requirement](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy). Wikipedia content uses Defuddle's built-in `#mw-content-text` extractor; it needs no separate proxy or hosted conversion API. Before extraction, image RDFa `resource` attributes are removed so Defuddle does not mistake Wikipedia file-description pages for image sources.

Substack `/p/…` articles use their initial HTML directly in the isolated Defuddle context. Publication subdomains and custom domains identified by Substack's `X-Served-By` response header skip publisher script execution. This avoids requests for analytics and account state that can obscure the supplied article or produce an unrelated incomplete-copy warning. Initial HTTP errors still respect the reader's retry cooldown; Defuddle requires a successful source response.

Defuddle 0.19.3 is the sole HTML extractor, using its full browser bundle for equation conversion. The URL flow follows [Obsidian Web Clipper's reader](https://github.com/obsidianmd/obsidian-clipper/blob/main/src/core/reader-view.ts) and [defuddle.md's converter](https://github.com/kepano/defuddle/blob/main/website/src/convert.ts): assign the resolved source URL to the inert document, run `parseAsync()` with the default article selection and standardization, and display the extracted HTML. Images retain lazy-loading hints; small-image filtering is disabled because source images are not downloaded during capture.

Rendered captures replace MathJax's duplicate visual glyphs with its expanded assistive MathML before the 2 MiB document limit. Executable scripts and styles are discarded after rendering; JSON-LD metadata remains available to Defuddle. Inline SVG diagrams inside article/main content are captured as PNG resources before sanitization, with at most 24 figures, 2048 pixels per dimension, and 8 MiB total. These static images preserve the rendered diagram; interactive controls remain available on the original. Captured figures are persisted before publishing the snapshot and use the existing private image delivery path. Raw SVG markup remains forbidden in reader content.

The parsing context blocks direct network requests. Defuddle's async extractors use a separate Worker bridge that permits public GET requests through the existing address and redirect checks, with at most eight requests and an eight-second extraction deadline inside the capture's existing request and byte budgets. Empty or timed-out async extraction falls back to synchronous Defuddle on a fresh document clone. Publisher scripts cannot access this bridge. The reader runs the installed package locally; it does not call defuddle.md's hosted API.

DOMPurify sanitizes the extracted article after metadata, math, and site-specific extraction. Code language, MathML with LaTeX sources, and callout structure survive; retained classes are scoped to the reader. Local section links and footnotes retain their targets, and images use validated resource delivery. New snapshots store only the cleaned `readerHtml`; older snapshots with a `documentHtml` field remain readable through their extracted content. A raw page body is never substituted for a failed extraction.

Snapshots are immutable. A conditional R2 state update claims one article render at a time and publishes the completed snapshot after storage succeeds. Expired claims can be recovered. Failed captures enter a retry cooldown. Explicit refresh creates a new version; a complete previous copy remains the default when a refresh only produces partial content.

New captures record the Defuddle extraction profile. Previously saved copies keep their storage keys and remain readable, including article versions referenced by notes. Opening a saved article does not recapture it after an extractor update. Use Refresh from source to create a version with the current extractor.

Images and PDF bytes are stored when requested. The saved snapshot lists the permitted resources; each delivery URL carries a short-lived signature scoped to its article, snapshot, and resource. These signatures grant access only to the saved resource, without granting access to read marks or notes.

arXiv abstract, HTML, and PDF links open the paper in the existing PDF viewer. Explicit paper versions are preserved. The original saved URL and article ID remain unchanged, so read marks and notes stay attached. A cached abstract switches to a PDF snapshot on the next open; its old snapshot remains available to quoted notes. This route needs no browser capture, and the PDF bytes use the same on-demand R2 resource storage.

X and Twitter post URLs use Defuddle's async extractor in the isolated parsing context. It receives the original URL and a blank document, skips the source timeline, and retrieves the post or full linked X article through its FxTwitter API path, with its own oEmbed text fallback. The reader displays the sanitized HTML once, without mounting Twitter widgets. Images use the same validated resource delivery as other articles.

GitHub file URLs (`blob` and `raw`), `raw.githubusercontent.com`, legacy `raw.github.com`, and named raw Gist files use a source-code snapshot. The Worker fetches public raw bytes through the same address, redirect, timeout, and 2 MiB limits, rejects binary/non-UTF-8 responses, and preserves the source text. No publisher browser session or article extraction is needed. The reader renders a focusable `pre` with Garden's code font and syntax-color variables; supported filename extensions receive Highlight.js tokens. Unknown languages and files above 200,000 characters remain complete plain text. Existing GitHub article snapshots switch to source capture when next opened, retaining their old versions for notes. Repository roots continue through normal article extraction.

These copies record the `twitter-defuddle-0.19.3-purify-1` profile and are saved in R2. A previously scraped or embedded post upgrades on its next open while retaining its old snapshot and notes. A failed extraction keeps the prior copy and enters the normal retry cooldown. Profiles and direct X article URLs continue through the HTML reader. The ordinary Arena item viewer retains its existing Twitter embeds.

Successful copies have no automatic age expiry. D1 holds no render catalogue, browsing history, or scroll log. Reader responses use `private, no-store`; R2 provides persistence without a public CDN cache. All article versions are retained, including versions referenced by notes. Storage cleanup can be added with reference checks later.

## API

All endpoints require a verified reader session, except a resource URL with a valid scoped signature. State mutations also require an exact same-origin `Origin` and an `X-Arena-Subject` header matching the authenticated reader.

| Method      | Path                                                                  | Result                                              |
| ----------- | --------------------------------------------------------------------- | --------------------------------------------------- |
| GET         | `/api/arena/feed?seed=…`                                              | Versioned, ordered catalogue and read marks         |
| POST        | `/api/arena/articles/:id/render`                                      | Saved copy or on-demand render; accepts `refresh`   |
| GET         | `/api/arena/articles/:id/render-status`                               | Pending, ready, or unavailable capture state        |
| GET         | `/api/arena/articles/:id/snapshots/:snapshotId`                       | A specific saved version with renewed resource URLs |
| GET, HEAD   | `/api/arena/articles/:id/snapshots/:snapshotId/resources/:resourceId` | Validated image or PDF bytes; supports ranges       |
| PUT         | `/api/arena/articles/:id/read`                                        | Reversible read mark with expected revision         |
| GET         | `/api/arena/articles/:id/notes`                                       | Notes and deletion tombstones for an article        |
| GET         | `/api/arena/notes?view=inbox`                                         | All account notes and deletion tombstones           |
| GET         | `/api/arena/notes?view=ready`                                         | Export bundle of ready notes                        |
| PUT, DELETE | `/api/arena/notes/:id`                                                | Revision-checked note update or tombstone           |
| POST        | `/api/arena/notes/:id/export-receipt`                                 | Acknowledge a verified backfill revision            |

The Worker chooses source URLs from the merged Arena and Curius catalogue. Clients supply stable article IDs, never arbitrary capture URLs. Note input preserves Markdown whitespace, validates occurrence/snapshot ownership, and has bounded lengths. Database and catalogue failures return an error rather than an empty successful state.

## Backfill contract

The Notes inbox can mark notes Ready and download their JSON export. The bundle contains each note's stable UUID, revision, source URL, occurrence, quotation, snapshot ID, and SHA-256 body hash. Downloading an export does not mark it backfilled.

A separate future workflow should locate and verify the current Markdown block by URL and occurrence, write the selected text, verify that write, then submit an export receipt with the exported revision. A receipt should identify its body hash and Markdown target. Acknowledging an older revision never clears a newer ready edit. The reader does not modify `content/are.na.md`.

## Verification

Focused tests cover the real Arena parser and emitted output, stable URL identities and queue order, WebCrypto sessions and resource signatures, actual local D1 revision races, R2 claims and resource ranges, browser DOM extraction, and local draft reconciliation. Browser verification should exercise both the phone drawer and desktop panel through the running Worker, including a cache hit, a saved note after reload, and an unavailable source.
