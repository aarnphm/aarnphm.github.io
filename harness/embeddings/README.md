# Embedding progress harness

A local static HTML/JavaScript workspace for the running PDF embedding queue. A loopback server pushes read-only snapshots over WebSocket. The observer reads the runner status, verified page cache, publication manifests and batch logs. The queue controls enqueue new or changed public PDFs and wake one serialized dispatcher. The observer itself does not modify jobs.

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

The current work lists pages admitted to the rolling extraction pipeline. The batch grid follows each physical page, including completions that arrive out of order. Completed counts come from verified cache records, so repeated log lines cannot increase progress. The displayed batch size uses its actual page plan. Older logs without admission events retain the first-pending-page fallback. The inventory refreshes every 60 seconds; status, cache and publication observations refresh every three seconds by default. Socket freshness and the runner heartbeat are reported separately. Missing data appears as `–`.

Owned model health requires the recorded process and matching HTTP identity. Reused servers are checked by HTTP identity and stay running after client work. The OCR readout shows active and queued requests, completion failures, prefix caching and continuous batching. Its token rate uses differences in the native completed-token counter over up to 30 seconds, with at least 10 seconds of observations. Tokens from a long request enter this counter when that request completes; this is a completed-output rate rather than instantaneous decode throughput.

`inputs.py` owns public source identities and cache validation, reusing the PDF builder and `PageCache`. `logs.py` reduces native batch logs. `models.py` reads model health and throughput. `snapshot.py` assembles observations. `server.ts` owns the loopback transport and child observer; `queue-api.ts` validates same-origin queue requests and invokes the queue client. The frontend preserves document row elements so live updates retain expanded rows and keyboard focus. All displayed source text uses text nodes.

The observer reads the runner's process start time once to bound the timing logs.
If process identity is unavailable, it withholds the estimate. The runner and
inference servers continue unchanged when the harness restarts.

Evidence and replay results: [embedding harness verification](../../docs/verification/embedding-harness/README.md).

Remaining-time regression replay:

```sh
pnpm exec tsx --test harness/embeddings/timing.e2e.test.ts
pnpm exec tsx --test harness/embeddings/streaming.e2e.test.ts
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
- Bind only to loopback, serve an explicit static-file allowlist, reject unrelated writes and foreign origins, and shut down the owned observer with the harness server. The active PDF runner and watcher remain untouched.
- Exercise the real HTTP/WebSocket server against isolated PDFs, native page-cache files, manifests and logs before verifying the live queue in a browser. Save commands, snapshots and screenshots under `docs/verification/embedding-harness/artifacts/`.

## Add PDFs to the queue

After a new public paper appears in the watcher’s search inventory, use **queue missing PDFs** or expand its row and choose **queue PDF**. Failed jobs expose **retry PDF**. Equivalent commands:

```sh
pnpm pdf:queue
pnpm pdf:queue --slug thoughts/papers/example.pdf
```

The client checks source hashes and full physical-page coverage. Complete papers are skipped; repeated requests share one pending job. Requests persist in `quartz/.quartz-cache/pdf-search/queue.json`. One detached dispatcher holds `.queue-worker.lock`, waits for the existing runner lock, and drains later arrivals. Its PID and log are `queue-worker.pid` and `queue-worker.log` beside the queue. Failed jobs retain their error and require an explicit retry.

The runner reuses compatible EmbeddingGemma 2 servers on 8000 or 8003 and compatible OCR on 8001. It verifies pinned model identity, starts a server only when needed on a free port, and stops only processes it owns. Cached OCR pages bypass OCR startup. A fully published selection loads no model. `VLLM_URL` can also direct `pnpm embed:notes` to an existing compatible server. Base and image builders retain their own encoder contracts.

Queue replay and server reuse evidence: [PDF queue verification](../../docs/verification/pdf-queue/README.md).
