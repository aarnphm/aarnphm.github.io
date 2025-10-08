# micromark-extension-wikilink

micromark syntax extension for Obsidian wikilinks with proper tokenization.

## overview

this package provides a micromark extension that tokenizes Obsidian-flavored wikilinks during markdown parsing, creating first-class AST nodes instead of relying on regex-based text replacement.

**future phases** (see `/Users/aarnphm/workspace/garden/docs/wikilink-parser-improvement.md`):

- p1: advanced Obsidian features

## features

### syntax support

- basic links: `[[target]]`
- aliases: `[[target|display text]]`
- anchors: `[[target#heading]]`
- block references: `[[target#^block-id]]`
- embeds: `![[file]]`
- combined: `[[target#anchor|alias]]`
- escaping: `[[file\|name]]`, `[[file\#hash]]`
- complex paths: `[[path/to/file]]`, `[[file (with) parens]]`
- multiple anchors: `[[file#h1#h2#h3]]` (Obsidian subheading style)

### edge cases handled

- empty components: `[[]]`, `[[#anchor]]`, `[[|alias]]`
- whitespace: `[[ spaced ]]` → target is `spaced`
- special characters in paths
- context awareness (doesn't parse inside code blocks/inline code)

## architecture

### tokenizer state machine

```
start → (embed?) → [ → [ → target → (anchor?) → (alias?) → ] → ] → ok
                              ↓          ↓          ↓
                           consume    consume    consume
                              ↓          ↓          ↓
                           chunks     chunks     chunks
```

**states**:

1. `start`: detect `!` embed marker or `[` opening
2. `openFirst`: consume first `[`
3. `openSecond`: consume second `[`
4. `targetStart`: begin target consumption
5. `targetInside`: consume target chars, handle escaping, detect delimiters
6. `anchorMarker`: consume `#`
7. `anchorStart`: begin anchor consumption
8. `anchorInside`: consume anchor chars (allows multiple `#`)
9. `aliasMarker`: consume `|`
10. `aliasStart`: begin alias consumption
11. `aliasInside`: consume alias chars (allows any content)
12. `closeFirst`: consume first `]`
13. `closeSecond`: consume second `]`, finalize

**escaping**: backslash `\` escapes the next character in target and anchor sections. the backslash is consumed during tokenization (not included in output).

### token types

defined in `types.ts`:

- `wikilink` - container
- `wikilinkEmbedMarker` - `!` prefix
- `wikilinkOpenMarker` - `[[`
- `wikilinkTarget` - file path
- `wikilinkAnchorMarker` - `#` delimiter
- `wikilinkAnchor` - heading/block text
- `wikilinkAliasMarker` - `|` delimiter
- `wikilinkAlias` - display text
- `wikilinkCloseMarker` - `]]`

chunk tokens (`wikilinkTargetChunk`, etc.) are used internally by micromark for buffering.

### mdast node structure

```typescript
interface WikilinkNode {
  type: "wikilink"
  value: string // original text: "[[target#anchor|alias]]"
  data: {
    wikilink: {
      raw: string // same as value
      target: string // "target"
      anchor?: string // "#anchor"
      alias?: string // "alias"
      embed: boolean // false
    }
  }
  position?: Position
}
```

the `WikilinkNode` is created during markdown parsing and contains structured metadata that can be consumed by downstream transformers.

## usage

### standalone

```typescript
import { fromMarkdown } from "mdast-util-from-markdown"
import { wikilink, wikilinkFromMarkdown } from "./micromark-extension-wikilink"

const tree = fromMarkdown("[[page#section|alias]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})
```

### with unified/remark

```typescript
import { unified } from "unified"
import remarkParse from "remark-parse"
import { remarkWikilink } from "./micromark-extension-wikilink"

const processor = unified().use(remarkParse).use(remarkWikilink)
```

### options

the `remarkWikilink` plugin and `wikilinkFromMarkdown` extension accept an options object:

```typescript
interface RemarkWikilinkOptions {
  obsidian?: boolean // enable Obsidian-style anchor handling (default: false)
}
```

#### obsidian mode

when `obsidian: true`, anchor handling matches Obsidian's behavior:

**nested heading resolution**: anchors with multiple `#` segments use only the last segment:

```typescript
// obsidian: true
"[[NVIDIA#cuda]]" → anchor: "#cuda"
"[[file#Parent#Child#Grandchild]]" → anchor: "#grandchild"

// obsidian: false (default)
"[[NVIDIA#cuda]]" → anchor: "#cuda"
"[[file#Parent#Child#Grandchild]]" → anchor: "#Parent#Child#Grandchild"
```

**anchor slugification**: anchors are normalized using github-slugger for consistency with heading IDs:

```typescript
// obsidian: true
"[[file#Section Title]]" → anchor: "#section-title"
"[[file#architectural skeleton of $ mu$]]" → anchor: "#architectural-skeleton-of--mu"

// obsidian: false (default)
"[[file#Section Title]]" → anchor: "#Section Title"
```

**usage**:

```typescript
import { unified } from "unified"
import remarkParse from "remark-parse"
import { remarkWikilink } from "./micromark-extension-wikilink"

// enable Obsidian mode
const processor = unified().use(remarkParse).use(remarkWikilink, { obsidian: true })

// or with fromMarkdown
import { fromMarkdown } from "mdast-util-from-markdown"
import { wikilink, wikilinkFromMarkdown } from "./micromark-extension-wikilink"

const tree = fromMarkdown("[[file#Parent#Child]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
// wikilink.data.wikilink.anchor === "#child"
```

## testing

comprehensive test suite in `index.test.ts` covers:

- basic links (simple, alias, anchor, block ref)
- embeds (prefix, with anchor, with alias)
- edge cases (empty components, multiple anchors)
- escaping (pipes, hashes, brackets)
- complex paths (slashes, spaces, special chars)
- multiple wikilinks in text
- context awareness (code blocks, inline code)
- malformed input handling
- position tracking

run tests:

```bash
pnpm exec tsx --test quartz/util/micromark-extension-wikilink/index.test.ts
```

## design decisions

### why consume backslashes during tokenization?

standard markdown behavior: escape characters are processed during parsing, not preserved in the AST. this matches how other micromark extensions work (e.g., `\*` becomes `*`).

consumers get clean data: `[[file\|name]]` → `target: "file|name"`, not `"file\\|name"`.

### why undefined for empty components?

semantic clarity: `[[target|]]` means "no alias provided", not "alias is empty string". undefined makes this explicit and prevents confusion with intentional empty strings.

consistency: matches how optional fields work throughout the mdast ecosystem.

### why allow multiple `#` in anchors?

Obsidian supports nested headings via `[[file#h1#h2]]` syntax. the tokenizer treats everything after `#` as anchor text, allowing this pattern naturally.

### why separate WikilinkNode from Link?

preserves metadata: wikilink-specific data (embed flag, original syntax) is maintained for downstream processing.

enables transformation: OFM transformer can convert WikilinkNode → Link/Image/Html based on context.

backward compatibility: existing code using regex-based parsing can migrate gradually.

## performance characteristics

- single-pass tokenization (no backtracking)
- O(n) complexity where n = input length
- minimal allocations (chunk buffering reuses strings)
- context-aware (skips code blocks automatically via micromark)

## files

- `index.ts` - public API exports
- `syntax.ts` - micromark tokenizer state machine
- `fromMarkdown.ts` - mdast node creation from tokens
- `types.ts` - TypeScript type definitions and augmentations
- `index.test.ts` - comprehensive test suite
- `README.md` - this file

## compatibility

- micromark: ^4.0.0
- mdast-util-from-markdown: ^2.0.0
- TypeScript: strict mode
- Node.js: 22+ (ESM)

## references

- [micromark documentation](https://github.com/micromark/micromark)
- [mdast specification](https://github.com/syntax-tree/mdast)
- [Obsidian wikilinks](https://help.obsidian.md/Linking+notes+and+files/Internal+links)
- [wikilink parser improvement plan](/Users/aarnphm/workspace/garden/docs/wikilink-parser-improvement.md)

## future work

see improvement plan for:

- integration with OFM transformer (phase 2)
- downstream plugin updates (phase 3)
- advanced Obsidian features: folder notes, title-based linking, transclusion ranges (phase 4)
- performance optimization and caching (phase 5)
