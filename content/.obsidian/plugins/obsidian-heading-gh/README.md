# Obsidian headings

Use **Jump to heading (gh)** to navigate the active Markdown note in Reading view, Live Preview, or Source mode. The picker recognizes ATX and Setext headings, excludes frontmatter and fenced code, and indents each row by its heading level. This vault binds the command to Ctrl+B. In Reading view, focus the note and press `g` then `h` within one second. In the editor, the existing Vimrc `gh` mapping invokes the same command.

Heading labels use Obsidian’s Markdown renderer, including native MathJax math, emphasis, code, and wikilink aliases:

```markdown
## **Math** $\frac{x^2}{\sqrt{y}} = \alpha$ and [[thoughts/Sets|sets]]
```

Rendered links supply their label text; clicking a row jumps to that heading. In Reading view the command scrolls the rendered note; in the editor it moves the cursor to the heading.

Use ↑/↓, j/k, or Ctrl+n/Ctrl+p to move. Enter jumps, and Esc closes the picker. Initial letters select headings; when several headings share an initial, the picker shows a second set of letter hints. The default list starts directly below the title, with no help strip or reserved hint column. Keyboard focus follows the selected row and uses the same background fill. The list keeps a single outer border and dividers between rows.

Styles inherit the active Obsidian theme. Closing the picker unloads its Markdown rendering component. Disabling the plugin closes any open picker.

Run `npm run build` from this directory to typecheck and rebuild the installed bundle. Reload the plugin or use the existing Hot Reload plugin. Preserve the plugin ID `obsidian-headings` and command ID `heading-gh-navigator`; the local directory is named `obsidian-heading-gh`. Generated bundles remain ignored.
