# `micromark-extension-ofm-sidenotes`

A micromark extension to support Obsidian-flavored sidenotes with rich syntax.

## Features

- **Inline Sidenotes**: `{{sidenotes[label]: content}}`
- **Reference-style Sidenotes**: `{{sidenotes[^label]}}` (reference) and `{{sidenotes[label]}}: content` (definition)
- **Rich Labels**: Support for markdown, wikilinks, and LaTeX within labels (e.g., `{{sidenotes[$E=mc^2$]: ...}}`).
- **Properties**: Metadata support via angle brackets (e.g., `{{sidenotes[label]<dropdown: true>: ...}}`).

## Syntax

### 1. Inline Sidenotes

Standard inline sidenotes with a label and content.

```markdown
Here is a point {{sidenotes[label]: This is the sidenote content.}}
```

### 2. Properties

You can add key-value properties inside angle brackets. This is useful for passing metadata to the renderer (e.g., for forcing inline display or dropdowns).

**Supported formats:**

- `{{sidenotes[label]<key: value>: ...}}`
- `{{sidenotes<key: value>[label]: ...}}`

```markdown
{{sidenotes[Note]<inline: true>: This forces the sidenote to stay inline.}}
```

### 3. Reference-style (Footnote-like)

Separate the marker from the content, similar to Markdown footnotes.

**Reference:**
Use a caret `^` at the start of the label to indicate a reference.

```markdown
This is a statement {{sidenotes[^ref1]}}.
```

**Definition:**
Define the content elsewhere using the block syntax.

```markdown
{{sidenotes[ref1]}}:
This is the multiline content of the sidenote.
It supports block elements like lists:

- Item 1
- Item 2
```

### 4. Rich Content in Labels

Labels can contain nested markdown, including wikilinks and math.

**Wikilinks:**

```markdown
{{sidenotes[[[WikiLink]]]: Content about the link.}}
```

**LaTeX (Math):**

```markdown
{{sidenotes[$x^2$]: Content about the equation.}}
```

**Long Labels:**
Labels can contain spaces and punctuation.

```markdown
{{sidenotes[A very long, descriptive label.]: Content.}}
```

## AST (MDAST)

This extension introduces three new node types to the mdast tree:

1.  **`sidenote`**: Inline sidenote.
    - `data.sidenoteParsed.label`: The raw label string.
    - `data.sidenoteParsed.labelNodes`: Parsed phrasing content of the label.
    - `data.sidenoteParsed.content`: The raw content string.
    - `children`: Parsed phrasing content of the body.
    - `data.sidenoteParsed.properties`: Dictionary of properties.

2.  **`sidenoteReference`**: Reference marker.
    - `label`: The label string (caret stripped).
    - `labelNodes`: Parsed phrasing content of the label.

3.  **`sidenoteDefinition`**: Block definition.
    - `label`: The label string.
    - `labelNodes`: Parsed phrasing content of the label.
    - `children`: Block content of the definition body.

## Usage

```typescript
import { micromark } from "micromark"
import { sidenote, sidenoteDefinition } from "micromark-extension-ofm-sidenotes"

const output = micromark(input, {
  extensions: [sidenote(), sidenoteDefinition()],
  // ... htmlExtensions for rendering
})
```

For recursive parsing (e.g., wikilinks inside labels), you must pass the relevant extensions to `fromMarkdown`:

```typescript
import { fromMarkdown } from "mdast-util-from-markdown"
import { sidenoteFromMarkdown } from "micromark-extension-ofm-sidenotes"
import { wikilink, wikilinkFromMarkdown } from "micromark-extension-ofm-wikilinks"

const tree = fromMarkdown(input, {
  extensions: [sidenote(), wikilink()],
  mdastExtensions: [
    sidenoteFromMarkdown({
      micromarkExtensions: [wikilink()],
      mdastExtensions: [wikilinkFromMarkdown()],
    }),
  ],
})
```
