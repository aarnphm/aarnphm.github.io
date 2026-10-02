# obsidian-marked

Read [plugin workflow](../../../../docs/agent-obsidian-plugins.md) when changing this local plugin.

- Use `npm run build` for its TypeScript check and installed esbuild bundle.
- Preserve user-owned watchers, note contents, and plugin IDs.
- Register rendering resources for unload cleanup. Verify Reading view, Live Preview, settings changes, and repeated reloads in Obsidian.
- Reuse Garden's established `rough-notation` dependency and native Obsidian Markdown/MathJax rendering.
