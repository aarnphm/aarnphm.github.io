# ofm-wikilink

[micromark](https://github.com/micromark/micromark) extensions to support Obsidian-flavored wikilinks with proper tokenization.

## what is this?

this package contains extensions to add support for Obsidian-style wikilink syntax to [`micromark`](https://github.com/micromark/micromark) through first-class AST nodes instead of regex-based text replacement.

the package provides a complete tokenizer state machine that handles wikilinks during markdown parsing, creating structured metadata for downstream processing. it supports the full range of Obsidian wikilink features including embeds, anchors, aliases, and complex path structures.

as there is no formal specification, this extension follows the behavior described in [Obsidian's help documentation](https://help.obsidian.md/Linking+notes+and+files/Internal+links) while handling edge cases and whitespace according to markdown conventions.

## when to use this

this project is useful if you want to support Obsidian-flavored wikilinks in your markdown pipeline.

you can use these extensions when working with [`micromark`](https://github.com/micromark/micromark) or [`unified`](https://github.com/unifiedjs/unified)/[`remark`](https://github.com/remarkjs/remark). the package provides both low-level micromark extensions and high-level remark plugins.

## install

this package is ESM only. in Node.js (version 22+), install with `pnpm`:

```sh
pnpm install ofm-wikilink
```

## use

### standalone with micromark

```typescript
import { fromMarkdown } from "mdast-util-from-markdown"
import { wikilink, wikilinkFromMarkdown } from "ofm-wikilink"

const tree = fromMarkdown("[[page#section|alias]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()], // obsidian: true by default
})
```

yields:

```javascript
{
  type: 'wikilink',
  value: '[[page#section|alias]]',
  data: {
    wikilink: {
      raw: '[[page#section|alias]]',
      target: 'page',
      anchor: '#section',
      alias: 'alias',
      embed: false
    }
  }
}
```

### with unified/remark

```typescript
import { unified } from "unified"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import rehypeStringify from "rehype-stringify"
import { remarkWikilink } from "ofm-wikilink"

const processor = unified()
  .use(remarkParse)
  .use(remarkWikilink) // obsidian: true by default
  .use(remarkRehype)
  .use(rehypeStringify)

const html = await processor.process("[[page|link]] and ![[image.png|caption]]")
```

yields:

```html
<a href="page">link</a> and
<figure>
  <img src="image.png" />
  <figcaption>caption</figcaption>
</figure>
```

## api

this package exports the identifiers `wikilink`, `wikilinkFromMarkdown`, and `remarkWikilink`.
there is no default export.

### `wikilink()`

creates an extension for `micromark` to enable wikilink syntax.

###### parameters

- `none`

###### returns

extension for `micromark` that can be passed in `extensions`, to enable wikilink syntax parsing ([`Extension`](https://github.com/micromark/micromark#syntaxextension)).

### `wikilinkFromMarkdown(options?)`

creates an extension for `micromark` to support wikilinks when building mdast.

###### parameters

- `options` (`RemarkWikilinkOptions`, optional) - configuration object

###### returns

extension for `micromark` that can be passed in `mdastExtensions`, to support wikilinks when constructing mdast ([`FromMarkdownExtension`](https://github.com/syntax-tree/mdast-util-from-markdown#extension)).

### `remarkWikilink(options?)`

unified/remark plugin to support wikilinks.

###### parameters

- `options` (`RemarkWikilinkOptions`, optional) - configuration object

###### returns

unified plugin that can be passed to `.use()`.

### `RemarkWikilinkOptions`

configuration options for wikilink processing:

```typescript
interface RemarkWikilinkOptions {
  /**
   * enable Obsidian-style nested anchor handling and automatic hast conversion.
   * default: true
   *
   * when true: uses internal slugification and annotates nodes for automatic HTML conversion
   * when false: returns raw wikilink nodes without hName annotations
   */
  obsidian?: boolean

  /**
   * file extensions to strip before slugifying.
   * default: ['.md', '.base']
   * only applies when obsidian: true
   */
  stripExtensions?: string[]
}
```

#### obsidian mode

when `obsidian: true` (default), two key behaviors are enabled:

**1. automatic hast conversion** - nodes are annotated with `hName`, `hProperties`, and `hChildren` for automatic HTML conversion:

```typescript
// obsidian: true (default)
unified().use(remarkWikilink).use(remarkRehype)
// [[Page]] → <a href="page">page</a>

// obsidian: false - no annotations, raw wikilink nodes only
unified().use(remarkWikilink, { obsidian: false }).use(remarkRehype)
// [[Page]] → raw wikilink node (user must handle conversion)
```

**2. obsidian-style anchor handling:**

**nested heading resolution** - anchors with multiple `#` segments use only the last segment:

```typescript
// obsidian: true (default)
"[[NVIDIA#cuda]]" → anchor: "#cuda"
"[[file#Parent#Child#Grandchild]]" → anchor: "#grandchild"

// obsidian: false
"[[NVIDIA#cuda]]" → anchor: "#cuda"
"[[file#Parent#Child#Grandchild]]" → anchor: "#Parent#Child#Grandchild"
```

**anchor slugification** - anchors are normalized using github-slugger:

```typescript
// obsidian: true (default)
"[[file#Section Title]]" → anchor: "#section-title"
"[[file#architectural skeleton of $ mu$]]" → anchor: "#architectural-skeleton-of-mu"

// obsidian: false
"[[file#Section Title]]" → anchor: "#Section Title"
```

#### strip extensions

strips file extensions before slugifying paths (only applies when obsidian mode is enabled):

```typescript
const processor = unified()
  .use(remarkParse)
  .use(remarkWikilink, {
    stripExtensions: [".md", ".mdx", ".base"], // default: ['.md', '.base']
  })

// input: [[notes.md]]
// output: <a href="notes">notes</a>

// input: [[data.base]]
// output: <a href="data">data</a>

// input: [[notes.mdx]]
// output: <a href="notes">notes</a> (with custom stripExtensions)
```

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
- whitespace normalization: `[[ spaced ]]` → target is `spaced`
- special characters in paths
- context awareness (doesn't parse inside code blocks/inline code)

## html

when using the unified pipeline with `remarkRehype`, wikilinks are automatically converted to appropriate HTML elements based on their type:

### regular links

wikilinks are expressed as standard anchor tags:

```typescript
// input: [[target]]
// output: <a href="target">target</a>

// input: [[target|alias]]
// output: <a href="target">alias</a>

// input: [[target#heading]]
// output: <a href="target#heading">target</a>
```

### image embeds

image embeds with dimensions or captions become figures:

```typescript
// input: ![[photo.jpg|400x300]]
// output: <img src="photo.jpg" width="400" height="300">

// input: ![[photo.jpg|A beautiful sunset]]
// output: <figure><img src="photo.jpg"><figcaption>A beautiful sunset</figcaption></figure>

// input: ![[photo.jpg|A beautiful sunset|400x300]]
// output: <figure><img src="photo.jpg" width="400" height="300"><figcaption>A beautiful sunset</figcaption></figure>
```

### media embeds

video and audio files are converted to appropriate HTML5 elements:

```typescript
// input: ![[demo.mp4]]
// output: <video src="demo.mp4" controls loop></video>

// input: ![[song.mp3]]
// output: <audio src="song.mp3" controls></audio>
```

### pdf embeds

pdf files are embedded as iframes:

```typescript
// input: ![[paper.pdf]]
// output: <iframe src="paper.pdf" class="pdf"></iframe>
```

### block transcludes

embedded sections and blocks become blockquotes with metadata:

```typescript
// input: ![[notes#summary]]
// output: <blockquote class="transclude" data-url="notes" data-block="#summary">
//           <a href="notes#summary" class="transclude-inner">Transclude of notes #summary</a>
//         </blockquote>
```

the conversion to HTML happens automatically through mdast annotations (`data.hName`, `data.hProperties`, `data.hChildren`) that `mdast-util-to-hast` uses during the remark-to-rehype transformation.

## css

wikilinks rendered as links can be styled using standard selectors:

```css
a {
  color: #ff00ff;
}
```

figure elements with captions can be styled:

```css
figure {
  margin: 1em 0;
}

figcaption {
  font-style: italic;
  text-align: center;
}
```

transcluded blocks have dedicated classes:

```css
blockquote.transclude {
  border-left: 3px solid #ccc;
  padding-left: 1em;
}

a.transclude-inner {
  text-decoration: none;
}
```

## syntax

wikilinks form with the following structure:

```bnf
<wikilink> ::= <embed>? "[[" <target>? <anchor>? <alias>? "]]"

<embed>    ::= "!"
<target>   ::= <content>
<anchor>   ::= "#" <content>
<alias>    ::= "|" <content>

<content>  ::= (<char> | <escaped>)+
<escaped>  ::= "\" <char>
<char>     ::= any character except "]", "|", "#" (unless escaped)
```

the tokenizer implements a state machine that processes wikilinks in a single pass:

**states**:

1. `start` - detect `!` embed marker or `[` opening
2. `openFirst` - consume first `[`
3. `openSecond` - consume second `[`
4. `targetStart` - begin target consumption
5. `targetInside` - consume target chars, handle escaping, detect delimiters
6. `anchorMarker` - consume `#`
7. `anchorStart` - begin anchor consumption
8. `anchorInside` - consume anchor chars (allows multiple `#`)
9. `aliasMarker` - consume `|`
10. `aliasStart` - begin alias consumption
11. `aliasInside` - consume alias chars (allows any content)
12. `closeFirst` - consume first `]`
13. `closeSecond` - consume second `]`, finalize

**escaping**: backslash `\` escapes the next character in target and anchor sections. the backslash is consumed during tokenization (not included in output).

**whitespace**: prefix and suffix whitespace in all components is stripped during parsing: `[[ target | alias ]]` becomes `target` and `alias`.

## types

the package defines custom mdast node types:

### `Wikilink`

```typescript
interface Wikilink extends Node {
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
    // hast conversion annotations (set when obsidian mode is enabled)
    hName?: string // target HTML element: 'a', 'img', 'video', 'audio', 'iframe', 'blockquote'
    hProperties?: Record<string, any> // HTML attributes
    hChildren?: (HastElement | HastText)[] // nested hast nodes
  }
}
```

the `Wikilink` node extends the standard mdast `Node` interface and includes structured metadata. when processed through the unified pipeline, the `hName`, `hProperties`, and `hChildren` fields enable automatic conversion to HTML via `mdast-util-to-hast`.

## architecture

### tokenizer state machine

the tokenizer processes wikilinks through a finite state machine:

```
start → (embed?) → [ → [ → target → (anchor?) → (alias?) → ] → ] → ok
                              ↓          ↓          ↓
                           consume    consume    consume
                              ↓          ↓          ↓
                           chunks     chunks     chunks
```

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

### performance characteristics

- single-pass tokenization (no backtracking)
- O(n) complexity where n = input length
- minimal allocations (chunk buffering reuses strings)
- context-aware (skips code blocks automatically via micromark)

## examples

### basic usage

```typescript
import { fromMarkdown } from "mdast-util-from-markdown"
import { wikilink, wikilinkFromMarkdown } from "ofm-wikilink"

// simple link
const tree1 = fromMarkdown("[[page]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})
// → wikilink.data.wikilink = { target: 'page', embed: false }

// with alias
const tree2 = fromMarkdown("[[page|display text]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})
// → wikilink.data.wikilink = { target: 'page', alias: 'display text', embed: false }

// with anchor
const tree3 = fromMarkdown("[[page#section]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})
// → wikilink.data.wikilink = { target: 'page', anchor: '#section', embed: false }

// embed with all features
const tree4 = fromMarkdown("![[file#heading|caption]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})
// → wikilink.data.wikilink = { target: 'file', anchor: '#heading', alias: 'caption', embed: true }
```

### with unified pipeline

```typescript
import { unified } from "unified"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import rehypeStringify from "rehype-stringify"
import { remarkWikilink } from "ofm-wikilink"

// default usage with obsidian mode (automatic)
const processor = unified()
  .use(remarkParse)
  .use(remarkWikilink) // obsidian: true by default
  .use(remarkRehype)
  .use(rehypeStringify)

const html = await processor.process("[[Page1|link]] and ![[image.png|caption|100x200]]")
// → <a href="page1">link</a> and <figure><img src="image.png" width="100" height="200"><figcaption>caption</figcaption></figure>

// with custom strip extensions
const customProcessor = unified()
  .use(remarkParse)
  .use(remarkWikilink, {
    stripExtensions: [".md", ".mdx"],
  })
  .use(remarkRehype)
  .use(rehypeStringify)
```

### raw mode (obsidian: false)

if you explicitly set `obsidian: false`, nodes won't be annotated with `hName`/`hProperties`/`hChildren`:

```typescript
const tree = fromMarkdown("[[target]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: false })],
})

// tree contains:
// {
//   type: 'wikilink',
//   value: '[[target]]',
//   data: {
//     wikilink: { raw: '[[target]]', target: 'target', embed: false }
//     // no hName/hProperties/hChildren - user must handle hast conversion
//   }
// }
```

this is useful if you want to handle wikilink conversion yourself in a custom rehype plugin.

## compatibility

this package was tested to work with:

- micromark version 4+
- mdast-util-from-markdown version 2+
- unified version 11+
- Node.js version 22+
- TypeScript strict mode

## security

this package is safe. it performs no file system operations and does not execute arbitrary code. wikilink targets are parsed as strings and passed through to the mdast tree without modification (beyond whitespace normalization and escape sequence processing).

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
pnpm exec tsx --test quartz/extensions/micromark-extension-ofm-wikilinks/index.test.ts
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

### why separate Wikilink from Link?

preserves metadata: wikilink-specific data (embed flag, original syntax) is maintained for downstream processing.

enables transformation: transformers can convert `Wikilink` → `Link`/`Image`/`Html` based on context.

backward compatibility: existing code using regex-based parsing can migrate gradually.

## related

- [micromark](https://github.com/micromark/micromark) - markdown parser
- [mdast](https://github.com/syntax-tree/mdast) - markdown abstract syntax tree
- [unified](https://github.com/unifiedjs/unified) - interface for parsing and transforming content
- [remark](https://github.com/remarkjs/remark) - markdown processor
- [Obsidian internal links](https://help.obsidian.md/Linking+notes+and+files/Internal+links) - official documentation

## files

- `index.ts` - public API exports
- `syntax.ts` - micromark tokenizer state machine
- `fromMarkdown.ts` - mdast node creation from tokens
- `toMarkdown.ts` - mdast to markdown serialization
- `types.ts` - TypeScript type definitions and augmentations
- `index.test.ts` - comprehensive test suite
- `README.md` - this file

## future work

see `/Users/aarnphm/workspace/garden/docs/wikilink-parser-improvement.md` for planned improvements:

- integration with OFM transformer (phase 2)
- downstream plugin updates (phase 3)
- advanced Obsidian features: folder notes, title-based linking, transclusion ranges (phase 4)
- performance optimization and caching (phase 5)

## contribute

contributions welcome. open issues for bugs or feature requests. pull requests should include tests and maintain existing code style.

## license

MIT
