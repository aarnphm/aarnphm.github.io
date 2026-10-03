# Obsidian sidenotes

Render Garden sidenotes in Reading view and Live Preview. Source mode keeps the original Markdown and adds a margin preview wherever the pane has room, without inserting a rendered label or foldout into the source text. In Live Preview, moving the cursor into a sidenote exposes its source for editing.

```markdown
A sentence with {{sidenotes[label]: Supporting **Markdown**, [[thoughts/Sets|links]], and $x \in A$.}} more text.

{{sidenotes: An automatically numbered note.}}

{{sidenotes<inline: true>[details]: An inline foldout.}}

{{sidenotes<left: true, right: false, internal: [[thoughts/Sets]]>[[[thoughts/Sets|sets]]]: Supporting text.}}
```

Click the label, or focus it and press Enter or Space, to open or close the note. Escape closes an open note and returns focus to its label. Hovering the label or focusing it with the keyboard changes the linked note's border to the theme accent. When the note pane has space beside the text, sidenotes open in the margins and align with their labels. Each pane balances notes between left and right by choosing the side with the lower occupied height. Notes on the same side stack without overlapping. At narrow widths, Reading view and Live Preview use inline foldouts; Source mode keeps the Markdown without a margin preview. `left` and `right` default to `true`; set one to `false` to restrict a note to the other margin. `inline` or `dropdown` always uses an inline foldout outside Source mode. Markdown examples inside code remain literal.

The command **Insert sidenote template** preserves selected text and selects the label placeholder when no text is selected.

Styles inherit Obsidian's current theme variables. The vault's enabled `plugin-ui` snippet applies the shared Arena styling to plugin dialogs, settings, menus, and notices. Each authored plugin also ships its own component styles.

Run `npm run build` from this directory to typecheck and rebuild the installed `main.js`. Reload the plugin, or let the existing Hot Reload plugin handle the bundle change. Keep the manifest ID `obsidian-sidenotes` stable. Generated bundles remain ignored.
