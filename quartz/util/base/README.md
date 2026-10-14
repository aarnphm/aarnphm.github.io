# bases compiler + runtime (quartz implementation)

status: active
last updated: 2026-10-14

this directory contains the obsidian bases compiler, interpreter, and runtime helpers used by quartz to render `.base` files. it is designed to match obsidian bases syntax and semantics with deterministic evaluation and consistent diagnostics.

You can test it out with any of the base file in my vault here:

```bash
pnpm exec tsx quartz/scripts/inspect-base.ts content/antilibrary.base > /tmp/antilibrary-ast-ir.json

jq '.expressions[] | {context, kind, source, ast}' /tmp/antilibrary-ast-ir.json
jq '.expressions[] | {context, kind, ir}' /tmp/antilibrary-ast-ir.json
```

## scope

- parse base expressions (filters, formulas, summaries, property expressions)
- compile expressions to bytecode ir
- interpret bytecode with a deterministic stack vm
- resolve file, note, formula, and property values
- render views (table, list, cards/gallery, kanban, board, calendar, map)
- surface parse and runtime diagnostics in base output

## architecture (pipeline)

1. parse `.base` yaml (plugin: `quartz/plugins/transformers/bases.ts`)
2. parse expressions into ast (`compiler/parser.ts`)
3. compile ast to ir (`compiler/ir.ts`)
4. evaluate ir per row with caches (`compiler/interpreter.ts`)
5. render views and diagnostics (`render.ts`)

## modules

- `compiler/lexer.ts`: tokenizer with span tracking and regex support
- `compiler/parser.ts`: pratt parser for expression grammar and error recovery
- `compiler/ir.ts`: bytecode instruction set + compiler
- `compiler/interpreter.ts`: stack vm, value model, coercions, methods, functions
- `compiler/diagnostics.ts`: diagnostics types and helpers
- `compiler/schema.ts`: summary config schema and builtins
- `compiler/properties.ts`: property expression builder for columns and config keys
- `render.ts`: view rendering and diagnostics output
- `query.ts`: summaries and view summary helpers
- `types.ts`: base config types and yaml parsing helpers

## value model (runtime)

runtime values are tagged unions with explicit kinds:

- null, boolean, number, string
- date, duration
- list, object
- file, link
- regex, html, icon, image

coercions are permissive to match obsidian behavior. comparisons prefer type-aware equality (links resolve to files when possible, dates compare by time, etc), with fallbacks when resolution fails.

## expression features (spec parity)

- operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `&&`, `||`, `!`, `+`, `-`, `*`, `/`, `%`
- member and index access
- function calls and method calls
- list literals and regex literals
- `this` binding with embed-aware scoping
- list helpers (`filter`, `map`, `reduce`) using implicit locals `value`, `index`, `acc`
- summary context helpers: `values` (column values) and `rows` (row files)

## diagnostics

- parser diagnostics are collected with spans at compile time
- runtime diagnostics are collected during evaluation and deduped per context
- base views render diagnostics above the view output

## this scoping

- main base file: `this` resolves to the base file
- embedded base: `this` resolves to the embedding file
- row evaluation: `file` resolves to the row file

## performance decisions

- bytecode ir keeps evaluation linear and stable
- per-build backlink index avoids n^2 scans
- property cache memoizes property expressions per file
- formula cache memoizes formula evaluation per file

## view rendering

- table, list, cards/gallery, kanban, board, calendar, map
- map rendering expects coordinates `[lat, lon]` and map config fields
- view filters combine with base filters via logical and

### Kanban

Native Obsidian 1.14 [Kanban views](https://obsidian.md/help/bases/views/kanban) use `type: kanban` and require `groupBy`. Quartz renders the same notes as columns within a focusable scroll region. Card titles and images link to the note. The published view supports browsing and search; property edits and note creation remain in Obsidian.

```yaml
views:
  - type: kanban
    name: by status
    groupBy:
      property: note.status
      direction: ASC
    groupOrder:
      - to watch
      - finished
      - null
    order:
      - note.title
      - note.year
    sort:
      - property: note.title
        direction: ASC
    columnWidth: 280
    image: note.posters
    imageFit: cover
    imageAspectRatio: 1.7777777778
```

- `groupOrder` defines group order and visibility for grouped tables, lists, galleries, and Kanban views. Only listed values appear, including explicitly listed empty columns. `null` includes notes with missing or empty values. `[]` hides all groups. Omit the key for automatic group order. This follows the [Bases syntax](https://obsidian.md/help/bases/syntax).
- Visibility applies before sorting and the view limit, so hidden groups do not consume the limit. Counts include the visible notes; empty columns contribute zero.
- List-valued grouping properties place a note in each distinct member's group. Empty lists join the missing-value group. A note is visible when any membership matches `groupOrder`; overall counts and limits count unique notes, while group counts include their own members.
- Obsidian 1.14 can display a list-valued group as one combined label, as observed with film co-production lists. The Cinematheque `analysis` view uses the first studio entry as its main production company. Its saved `primary_studio: "list(note.studio)[0]"` formula handles lists and single studio names, and `groupBy.property: formula.primary_studio` places each film in one column. Studio order in the note defines this convention; the full credit list stays in `note.studio`. The other views retain their own grouping rules.
- The first property in `order` becomes the Kanban card title. Without `order`, the title uses `file.name`. Remaining properties retain their display names, formulas, links, and value rendering.
- `columnWidth` is a positive width in pixels (default `280`). Phone columns fit within `85vw`. `imageFit` accepts `cover` or `contain`; native `imageAspectRatio` is height divided by width. Values above `1` produce tall covers, `1` produces squares, and values below `1` produce wide covers. For example, `1.7777777778` gives a cover about 496 px high at 279 px wide. Quartz converts this to the inverse CSS aspect ratio. Without an image ratio, Kanban uses the existing `16 / 9` landscape default. Images accept a URL, a local path, a wikilink, or the first image in a list.
- Galleries also accept native `imageFit` and `imageAspectRatio`. Legacy `cardAspect` and `groupAspects` retain their width divided by height semantics. Native `imageAspectRatio` takes precedence over `cardAspect`. Existing `cardSize` and `groupSizes` continue to work.
- Search retains columns, updates their visible counts, and displays an empty message when a column has no matching cards. Note links retain keyboard navigation and visible focus.
- Full-page Kanban fills the remaining viewport beneath the toolbar. Embedded boards keep a bounded height. Sticky headings cover the scrollport's top edge; the scroll region has no outer outline.
- Columns meet at single dividers, and film rows use the Bases table's surfaces, separators, and padding. Studio headings use the table's compact monospace typography and remain static during hover, focus, and vertical scrolling.
- Column headings and card titles use one line with an ellipsis and a full-title tooltip. Overflowing card titles reveal their end on hover or keyboard focus, then restore the ellipsis when the interaction ends. Reduced motion keeps text static. The title script scopes listeners, resize observation, and animation cancellation to the navigation root.

Verification commands and repeatable artifacts: [Bases Kanban checks](../../../docs/verification/bases-kanban/README.md).
