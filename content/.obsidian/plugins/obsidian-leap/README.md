# Garden Leap

Local Obsidian navigation using the Leap mappings configured in
`~/.config/nvim/after/plugin/motion.lua`. Build with `npm run build`, then enable
**Garden Leap** in Community plugins.

| Keys            | Action                                                              |
| --------------- | ------------------------------------------------------------------- |
| `f` + character | Jump forward to the nearest visible matching character.             |
| `F` + character | Jump backward to the nearest visible matching character.            |
| `t` + character | Jump forward, stopping immediately before the match.                |
| `T` + character | Jump backward, stopping immediately after the match.                |
| `ga`            | Select the smallest syntax node at the cursor.                      |
| `gA`            | Select the smallest syntax ancestor spanning multiple source lines. |

Character searches use one character, are case-sensitive, and span the visible
editor viewport across lines. Folded or replaced source text is excluded. Spaces,
tabs, and line endings match each other; opening brackets, closing brackets, and
quotation marks each form an equivalence group. The first and last characters of
matching runs remain targets.

After the first jump, `f` or `t` advances through the original search direction;
`F` or `T` returns toward the origin. Space or Enter advances and Backspace goes back.
Displayed labels select a remaining target. Labels retain their original target
indices. The alphabet covers one group of targets; traversal reaches later
targets. Backward traversal from the first target wraps to the last. Escape clears
the temporary markers while retaining the jump. Unrelated keys end the session and continue to Vim.
Enter immediately after `f/F/t/T` repeats the previous search character.

Source and Live Preview support Vim Normal, Visual, and operator motions, such as
`2fx`, `vfx`, `dfx`, and `dtx`. Counts and character operators commit immediately.
The plugin preserves native undo and repeat handling.

During `ga`, `a` or Enter expands the selection and `A` or Backspace shrinks it.
Moving backward from the smallest ancestor wraps to the root. `gA` uses Enter and
Backspace. A pending syntax operator, such as `dga`, waits for a displayed label
or Enter to commit the chosen ancestor. `a` also accepts the first ancestor in
`dga`. Counts do not change syntax ancestry.

## Syntax parsers

The editor uses the installed Lezer Markdown parser with GFM support for links,
emphasis, paragraphs, lists, blockquotes, and fenced code. Meaningful native
embedded-language nodes are retained when they fit inside the Markdown node.
This supplies syntax-aware selection without a second Tree-sitter runtime.
Node boundaries depend on these parsers and can differ from Neovim's Tree-sitter
grammars.

Parsing is cached against the immutable editor document and resumes under a
cooperative time budget. Unfinished semantic trees are withheld; an unusually
large document may require another invocation. A single very large Markdown
block can exceed the budget before the parser yields.

## Reading view and commands

Reading view supports `f/F/t/T` and `ga/gA` through rendered text and DOM ranges.
It excludes controls, sidenote bodies, hidden text, and other panes. The rendered
caret supplies the origin, or the visible pane supplies a start when no caret is
present. Reading syntax selection follows rendered semantic elements; `gA`
chooses block elements rather than source line ranges. Vim operators and counts
belong to the source editor.

The command palette also exposes four character searches, **Select syntax node**,
and **Select syntax node by lines**. Commands work with Vim disabled.
Reading shortcuts leave `gh` available to the headings plugin.

## Settings and lifecycle

Settings control editor Vim mappings, Reading shortcuts, and target labels.
Colors use the vault's theme tokens and shared Garden plugin styling. Navigation,
outside edits or selections, scrolling, resizing, and unloading clear transient
markers. Unloading removes only mappings owned by this plugin, retaining mappings
installed later by another plugin.

The source package contains the implementation. Generated `main.js`, local
`data.json`, and `node_modules` follow the existing plugin ignore policy.
