# Arena reader

`/arena/feed` reads the saved links in `content/are.na.md`. Quartz emits the page shell and `static/arena-feed.json`; the existing Cloudflare Worker owns authentication, read marks, notes, and article capture.

## Reading and notes

The default queue shuffles unread `later: true` links first, then the remaining unread links. The shuffle seed, skipped links, selected article, and reading position stay on the current device. Only an explicit read action changes completion in D1. Saving a note or opening a source leaves completion unchanged.

Text notes open in a modal drawer on phones and a side panel on larger screens. Drafts are written to IndexedDB before server acknowledgement. Saved notes and read marks belong to the authenticated account. Revision checks prevent a stale device from overwriting a newer edit. Conflicts keep the local draft available for reconciliation.

Selecting text in Reader view can create a note with a quotation, surrounding text, and the saved article version. Full-document view runs in an isolated iframe. PDFs use the existing PDF viewer, and supported videos use their provider's embed. Unsupported, inaccessible, or incomplete sources retain an Open original link.

## Access and development

The reader uses the existing GitHub comments login. The OAuth callback issues a separate signed, HttpOnly owner session with an immutable `github:<numeric id>` subject. `ARENA_OWNER_LOGIN` defaults to `aarnphm`; `SESSION_SECRET` signs both the session and resource capabilities. Keep the secret in Cloudflare Secrets or local environment files.

For local development, add this setting to the ignored `.env.local`:

```dotenv
ARENA_READER_DEV=true
```

This setting only permits requests whose URL hostname is `localhost`, `127.0.0.1`, or `::1`. It creates a separate `dev:aarnphm` identity. Use the Worker port, normally `http://localhost:8080/arena/feed`; the Quartz-only server does not serve the reader API. An already running Wrangler process must load the local environment setting before it takes effect.

The required bindings are:

| Binding                     | Responsibility                                      |
| --------------------------- | --------------------------------------------------- |
| `ASSETS`                    | Feed shell and fixed catalogue asset                |
| `ARENA_READER`              | D1 read marks and text notes                        |
| `ARENA_CONTENT`             | Private R2 article versions and visited images/PDFs |
| `BROWSER`                   | Anonymous Cloudflare browser capture                |
| `ARENA_RENDER_RATE_LIMITER` | Browser launch limit per authenticated reader       |

The Browser binding uses Cloudflare during local development, so local cache misses use real browser time. Reader D1 and R2 stay local unless their commands explicitly target remote storage. The configured development host is `localhost`, which keeps the loopback identity and origin checks consistent with the local request. See [Cloudflare's Browser Run development reference](https://developers.cloudflare.com/browser-run/reference/wrangler/).

Apply only the reader migration during reader setup:

```sh
pnpm exec wrangler d1 migrations apply ARENA_READER --local
pnpm exec wrangler d1 migrations apply ARENA_READER --remote
```

The remote command changes the configured reader database. The normal deployment migration scripts include it. Provisioning storage and applying a migration do not deploy the Worker or publish a new Quartz build.

## Capture and persistent storage

Opening a link checks R2 first. A usable saved copy is returned without starting Chromium. First HTML visits launch a bounded browser session, then extract and sanitize the resulting document in a separate context. Source requests pass through the Worker's public-address fetch boundary. Publisher cookies, authorization headers, source forms, and publisher scripts are never installed in the reader page.

Snapshots are immutable. A conditional R2 state update claims one article render at a time and publishes the completed snapshot after storage succeeds. Expired claims can be recovered. Failed captures enter a retry cooldown. Explicit refresh creates a new version; a complete previous copy remains the default when a refresh only produces partial content.

Images and PDF bytes are stored when requested. The saved snapshot lists the permitted resources; each delivery URL carries a short-lived signature scoped to its article, snapshot, and resource. These signatures permit images inside the opaque full-document iframe without granting access to read marks or notes.

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

The Worker chooses source URLs from the generated catalogue. Clients supply stable article IDs, never arbitrary capture URLs. Note input preserves Markdown whitespace, validates occurrence/snapshot ownership, and has bounded lengths. Database and catalogue failures return an error rather than an empty successful state.

## Backfill contract

The Notes inbox can mark notes Ready and download their JSON export. The bundle contains each note's stable UUID, revision, source URL, occurrence, quotation, snapshot ID, and SHA-256 body hash. Downloading an export does not mark it backfilled.

A separate future workflow should locate and verify the current Markdown block by URL and occurrence, write the selected text, verify that write, then submit an export receipt with the exported revision. A receipt should identify its body hash and Markdown target. Acknowledging an older revision never clears a newer ready edit. The reader does not modify `content/are.na.md`.

## Verification

Focused tests cover the real Arena parser and emitted output, stable URL identities and queue order, WebCrypto sessions and resource signatures, actual local D1 revision races, R2 claims and resource ranges, browser DOM extraction, and local draft reconciliation. Browser verification should exercise both the phone drawer and desktop panel through the running Worker, including a cache hit, a saved note after reload, and an unavailable source.
