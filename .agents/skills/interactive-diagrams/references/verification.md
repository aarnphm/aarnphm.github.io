# Figure verification

Inspect the existing `quartz/scripts/dev.ts` process and its current output. The source watcher rebuilds TSX and inline TypeScript as well as styles; check the current watcher implementation if behavior changes. Do not restart a user-owned process or launch a full build just to verify a figure.

After a source edit, wait for its matching new `build:ready`, confirm the HTTP endpoint is available, and inspect the rendered page. Temporary HTTP failure during a rebuild is not proof of a broken edit. If no matching build arrives, inspect the watcher logs and report that blocker.

Check the figure at 320px and desktop width, in the light and dark themes, at 200% and 400% zoom, and with its actual controls. At zoom, look for lines that end short of their node, labels that collide or clip, and colours that disagree with the caption or legend. For interactive changes, exercise keyboard operation and SPA navigation cleanup. Restore any test-only control state after inspection.

A focused Sass compile can establish that a stylesheet compiles. It does not prove rendered geometry or interaction. Browser DOM injection may help prototype a layout; report it as prototype evidence and verify the actual source-rendered component separately. Do not infer KaTeX output from CSS alone.
