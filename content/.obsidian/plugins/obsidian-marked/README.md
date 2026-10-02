# Garden Marked

Render Garden's `::text::` syntax with Rough Notation in Obsidian Reading view and Live Preview. Bare markers use `h2`, matching Quartz. The initial annotation is a box.

```markdown
This is ::marked text::.
This is ::marked text{h3}:: with a different palette color.
::**Formatting**, [[thoughts/Sets|links]], `code`, and $x^2${h4}::
```

Open **Settings → Garden Marked** to choose the default annotation: box, underline, circle, highlight, strike through, crossed off, or brackets. Each level from `h1` to `h7` can override the type and color. Empty colors follow the garden's theme palette (`--rose`, `--love`, `--lime`, `--gold`, `--pine`, `--foam`, `--iris`). Custom colors accept CSS color values, including theme variables. Changes apply immediately to open notes and are saved locally.

Markers can wrap visually across lines. Keep each marker on one Markdown source line. Markdown formatting, links, inline code, and native Obsidian MathJax rendering inside markers are preserved. Code examples, frontmatter, comments, math expressions containing delimiter examples, and escaped markers stay literal. An unfinished marker stays editable.

In Live Preview, place the cursor inside a marker to edit its Markdown source. Source mode always preserves the original syntax. Annotation SVGs ignore pointer input so links remain usable. Rendering resources are removed when the note or plugin unloads.

The plugin bundles Garden's existing `rough-notation` library. Settings apply locally in Obsidian; Quartz continues to use its own rendering configuration.

## Development

```sh
npm ci
npm run build
```

The scoped build checks TypeScript and emits the installed `main.js`. Generated bundles, dependencies, and `data.json` settings are ignored. `npm run dev` starts an esbuild watcher, so inspect running processes before starting one.
