# obsidian-bookcase

Read [plugin workflow](../../../../docs/agent-obsidian-plugins.md) when changing this local plugin.

- Use `npm run build` for its TypeScript check and installed esbuild bundle.
- Preserve user-owned watchers, note contents, and plugin IDs.
- Register listeners, observers and timers for unload cleanup. Verify scrolling, the lift, the page panel, theme changes and repeated reloads in Obsidian.
- Import the shared `quartz/util/base/bookcase/geometry.ts` and `styles.ts` only through `src/scene-model.ts`; never copy or fork them.
- Keep scroll frames cheap: windowing changes only a shelf's children, a finished cover patches only its books, and markup, margin shelves and the next lift are prepared in idle time. Check with `node harness/run.mjs` (SPEC 13).
