# Embedding progress harness

A local static HTML/JavaScript workspace for the running PDF embedding queue. A loopback server pushes read-only snapshots over WebSocket. The observer reads the runner status, verified page cache, publication manifests and batch logs. It does not start, stop or modify the embedding job.

## Run

```sh
pnpm harness:embeddings
# http://127.0.0.1:8091

# Optional port and sampling interval, in seconds
pnpm harness:embeddings --port 8092 --interval 5

# One observation, without the server
.venv/bin/python harness/embeddings/snapshot.py --once

# End-to-end replay against isolated PDFs and cache records
pnpm exec tsx --test harness/embeddings/server.e2e.test.ts
```

The page is static HTML, CSS and browser JavaScript, with no frontend build step. The local server is required for filesystem observations and WebSocket updates. It uses the existing `ws`/`tsx` dependencies and the repository Python environment. `index.html` can be edited directly; refresh the page for HTML, CSS or client changes. Restart this harness for server or Python changes. Ctrl+C closes the harness and its observer, leaving the PDF runner and model servers intact.

The detached instance started for this task records its PID and output under `quartz/.quartz-cache/embedding-harness/`. Its browser reconnects automatically after a server restart and preserves the current search, filter and expanded paper rows. No login service or scheduled monitor is installed.

## What the counts mean

- **OCR cached:** valid pages for the current source bytes and extraction recipe, including pages still inside an unfinished batch. Duplicate PDF files with identical bytes can share cached pages.
- **Published:** pages present in the generated PDF manifest with a matching source hash. A page can produce several passage vectors, or no text vectors.
- **Local site output:** matching pages in `public/embeddings/pdf/manifest.json`. This measures the local watcher's output, not a deployed site.
- **Remaining papers:** documents whose full requested page range has not yet been published. The table separately shows each paper's remaining OCR pages.
- **Remaining OCR estimate:** distinct uncached source/page pairs multiplied by the mean elapsed time of up to 50 newly extracted pages from the current runner process and extraction recipe. A page enters the sample only when its cache record verifies and has no migration origin. Repeated log entries count once. At least five samples are required; earlier runs and migrated pages do not provide new timing evidence. Duplicate PDFs share OCR work, while the coverage totals retain each document reference. The display includes the sample size and number of sampled PDFs. Embedding, publication, pauses and future throughput changes are excluded, so this estimates remaining OCR processing time.
- **Cache updated:** the file's last cache-write time, which can represent migration as well as inference.

The current page is the first requested page of the active batch that has no verified cache record. It can briefly represent the next page while the extractor prepares it. The inventory refreshes every 60 seconds; status, cache and publication observations refresh every three seconds by default. Socket freshness and the runner heartbeat are reported separately. Missing data appears as `–`.

`inputs.py` owns public source identities and cache validation, reusing the PDF builder and `PageCache`. `logs.py` reduces native batch logs. `snapshot.py` assembles observations. `server.ts` owns only the loopback HTTP/WebSocket transport and its child observer. The frontend preserves document row elements so live updates retain expanded rows and keyboard focus. All displayed source text uses text nodes.

The observer reads the runner's process start time once to bound the timing logs.
If process identity is unavailable, it withholds the estimate. The runner and
inference servers continue unchanged when the harness restarts.

Evidence and replay results: [embedding harness verification](../../docs/verification/embedding-harness/README.md).

Remaining-time regression replay:

```sh
pnpm exec tsx --test harness/embeddings/timing.e2e.test.ts
```

This exercises the observer process with actual PDFs, checksummed cache files and
batch logs. It covers old-run and old-recipe exclusion, migrated pages, repeated
timing records, the five-page minimum, slow pages, duplicate PDFs, further page
completion and zero remaining OCR work. See [the estimate correction](../../docs/verification/embedding-harness-eta/README.md).

## Acceptance

Acceptance cases, recorded before implementation:

- Count eligible documents/pages from the same public inventory as the PDF builder. Count current-recipe OCR pages only after cache validation; distinguish those pages from generated vectors and pages in the local site output.
- Show the active document/page, batch budget, complete and remaining documents/pages, recent publications, and an explicitly labelled OCR-only estimate from measured page timings.
- Push updated counts over an existing WebSocket connection after cache/publication changes. Show disconnected, stale-runner, stopped, complete and error states without discarding the last known counts.
- Keep the document list searchable and filterable while updates arrive. Preserve focus and expanded rows. Support both themes, keyboard operation, reduced motion and 320px widths.
- Bind only to loopback, serve an explicit static-file allowlist, reject writes and foreign WebSocket origins, and shut down the owned observer with the harness server. The active PDF runner and watcher remain untouched.
- Exercise the real HTTP/WebSocket server against isolated PDFs, native page-cache files, manifests and logs before verifying the live queue in a browser. Save commands, snapshots and screenshots under `docs/verification/embedding-harness/artifacts/`.
