# obsidian-leap

Read [plugin workflow](../../../../docs/agent-obsidian-plugins.md) before changing this local plugin.

- Use `npm run build` for the TypeScript check and installed esbuild bundle.
- Reuse Obsidian's native CodeMirror and Vim adapter, plus Garden's installed Markdown parser. Preserve note contents and other plugins' mappings.
- Register session, overlay, editor, and key-handler cleanup. Verify Source, Live Preview, Reading, native Vim visual/operator/count behavior, cancellation, and repeated reloads.
- Match the configured Neovim mappings in `~/.config/nvim/after/plugin/motion.lua`; describe parser differences explicitly.
