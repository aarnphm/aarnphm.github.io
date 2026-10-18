# Notebook runtime trust boundary

JavaScript notebook cells execute with the site's origin authority after a
reader runs a cell. Published source comes from repository notebooks. Saved
edits come from the reader's local browser storage or editor. Mounting a
notebook and opening a URL with a `code` query parameter do not execute source.

The runtime uses a dedicated Web Worker to keep execution off the page's main
thread and to support interruption. The worker has no DOM or `localStorage`
interface. It retains same-origin capabilities, including IndexedDB and native
XHR with ambient cookies. Treat executed JavaScript as trusted code with those
capabilities.

The notebook `fetch` helper routes sibling assets through the page's asset
resolver. This helper supports notebook asset loading. It does not restrict all
network APIs available to evaluated JavaScript.

Worker messages check their source marker and runtime ID. The notebook client
sanitizes HTML outputs with DOMPurify before insertion. These controls protect
message routing and output rendering; executed code retains its worker-origin
authority.

Do not feed remote untrusted source into this runtime. A feature that runs such
source requires a separate execution origin with an explicit network and
storage policy. Changing `eval` to another JavaScript execution primitive would
retain the same origin capabilities.

## Verification

Run the existing development watcher and confirm its `build:ready` event. The
local Worker normally serves the resulting assets on port 8080.

```sh
node quartz/scripts/verify-javascript-runtime.mjs
```

`GARDEN_TEST_ORIGIN` overrides the local site address. `ARENA_TEST_CHROME`
overrides the Chromium executable. The harness uses an isolated browser
profile, the watcher-served notebook client and worker, and a local fixture
server with a test-only cookie. It records asset URLs and SHA-256 digests,
browser output, a screenshot, and the result in
`docs/verification/code-scanning-remediation/artifacts/runtime/`.

The fixture verifies explicit execution, published and local source selection,
output sanitization, sibling asset routing, worker message rejection,
interruption, restart, and the same-origin capabilities described above. Its
cookie and endpoint contain fixture data only. No provider tokens or production
accounts are involved.
