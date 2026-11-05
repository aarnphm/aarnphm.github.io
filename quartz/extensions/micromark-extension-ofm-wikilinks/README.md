# OFM wikilinks

This extension parses Garden wiki references into lossless Markdown nodes. The [authoring guide](authoring.md) covers links, embeds, captions, image CSS, PDF settings, transclusions and table escaping. The [compatibility profile](../../compatibility/README.md) records the supported Obsidian behavior and Garden extensions.

## Parsing and source identity

`wikilink()` supplies the micromark tokenizer. `wikilinkFromMarkdown()` compiles its tokens. `remarkWikilink` registers both with a Unified processor.

```typescript
import { fromMarkdown } from 'mdast-util-from-markdown'
import { wikilink, wikilinkFromMarkdown } from './index'

const tree = fromMarkdown('[[page#Section|**label**]]', {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})
```

A `wikilink` node holds:

- `value`: the authored reference, including delimiters.
- `data.reference`: immutable `ReferenceSyntax`, with the raw target, each fragment, settings, label, embed flag and UTF-16 source spans.
- `data.referenceFields`: compiler-decoded target, fragments, settings and label. Markdown character references and escapes are decoded here. An explicit empty label remains `''`.

Parsing performs no catalog lookup and emits no HTML annotations. Code and escaped literals keep their Markdown meaning. Generated fragments are marked `source: 'generated'`; they cannot claim local fragment offsets as positions in the containing document.

For literal properties outside Markdown, use `parseLiteralReference` or `extractLiteralReferences` from `quartz/util/references/literal.ts`. Those helpers share the grammar and keep their own escape-decoding policy. Editor callers use `parseReference` to distinguish complete, incomplete and invalid input.

## Resolution and rendering

Garden indexes document headings, blocks and Base views before resolving references. `resolveDocumentReferences` records found, missing, ambiguous, external and unsupported outcomes. A found record carries source identity, a typed selector, interpreted settings and its published route. Resolution leaves authored syntax unchanged.

`referenceHastHandler` from `quartz/util/references/render.ts` converts a node to a link, media element, Canvas placeholder or typed transclusion placeholder. Garden installs it in `mdastToHastRoot`. Standalone consumers install it explicitly:

```typescript
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'
import { remarkWikilink } from './index'
import { referenceHastHandler } from '../../util/references/render'

const processor = unified()
  .use(remarkParse)
  .use(remarkWikilink)
  .use(remarkRehype, { handlers: { wikilink: referenceHastHandler } })
  .use(rehypeStringify)
```

Standalone rendering projects a destination without claiming that a file exists. Catalog resolution supplies duplicate-heading identity, implicit labels and selected Base views in the full pipeline. Garden's HTML transformers add captions, media controls, URL decoration and browser resources after conversion.

The HTML pipeline normalizes raw HTML before semantic transformers run. A private document identity reconnects normalized elements to typed records. The pipeline removes that transport attribute before publication and retains the typed record for cache restoration and transclusion copies. Serialized semantic-looking HTML attributes cannot authorize a transclusion.

## Edits and serialization

`wikilinkToMarkdown()` writes the captured reference bytes. Use `editWikilink(node, patch)` for deliberate changes; it validates the patch and returns a replacement node or diagnostics. Patches contain raw component text, with reserved delimiters escaped.

```typescript
const result = editWikilink(node, {
  target: 'new-page',
  fragments: ['Parent', 'Child'],
  label: '**new label**',
})
if (result.kind === 'complete') {
  // Replace the authored node with result.node.
}
```

Routing, heading selection, label projection and provider interpretation must leave serialization unchanged. The syntax owner also exposes `editReference` for callers that work directly with `ReferenceSyntax`.

## Ownership and verification

| Owner                                                             | Responsibility                                                              |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `syntax.ts`, `fromMarkdown.ts`, `toMarkdown.ts`, `edit.ts`        | Markdown tokenization, compilation, source serialization and explicit edits |
| `quartz/util/references/model.ts`, `scan.ts`, `syntax.ts`         | Shared grammar, raw fields, spans and incomplete editor input               |
| `request.ts`, `literal.ts`                                        | Host decoding, typed requests and literal-property helpers                  |
| `catalog.ts`, `document.ts`, `target.ts`, `route.ts`              | Source identity, indexed selectors, settings and publication routes         |
| `markdown.ts`, `render.ts`, `render-media.ts`, `hast.ts`          | Document resolution, shared HAST conversion and transient binding           |
| `quartz/util/transclude-props.ts`, `quartz/components/renderPage` | Typed selection, occurrence scoping, cycle limits and expansion             |

The [adapter-removal record](../../compatibility/adapter-removal.md) lists the phase gates and replay evidence. Run affected parser suites and real pipeline fixtures when changing this boundary. Browser behavior requires the matching fixture watcher's `build:ready`, HTTP availability and a fresh served-browser inspection.
