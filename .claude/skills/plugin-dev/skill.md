# Plugin Development

Develop and test Quartz plugins following the repository's architecture and constraints.

## When to Use

- User wants to create a new transformer, emitter, or filter plugin
- User needs help debugging or modifying existing plugins
- User wants to understand the plugin architecture
- User needs to validate plugin implementation against best practices

## Instructions

### Understanding Plugin Types

**Transformers** (`quartz/plugins/transformers/`)
- Transform markdown/HTML content (mdast/hast trees)
- Can provide `textTransform`, `markdownPlugins`, `htmlPlugins`
- **NEVER use `fs` operations** - only transform AST trees
- Examples: citations, syntax highlighting, sidenotes

**Filters** (`quartz/plugins/filters/`)
- Decide whether content should be published
- Simple boolean logic based on frontmatter or content
- Examples: draft removal, private content filtering

**Emitters** (`quartz/plugins/emitters/`)
- Generate output files from processed content
- Can perform file I/O operations
- Generate HTML pages, assets, indexes, etc.
- Examples: content pages, 404 pages, asset copying

### Creating a Filter Plugin

Template for `quartz/plugins/filters/<name>.ts`:

```typescript
import { QuartzFilterPlugin } from "../types"

export const FilterName: QuartzFilterPlugin<{}> = () => ({
  name: "FilterName",
  shouldPublish(_ctx, [_tree, vfile]) {
    // Return true to publish, false to skip
    const shouldPublish = /* your logic here */
    return shouldPublish
  },
})
```

Access frontmatter: `vfile.data?.frontmatter?.field`

### Creating a Transformer Plugin

Template for `quartz/plugins/transformers/<name>.ts`:

```typescript
import { QuartzTransformerPlugin } from "../types"

export const TransformerName: QuartzTransformerPlugin = () => {
  return {
    name: "TransformerName",

    // Optional: transform raw text before markdown parsing
    textTransform(_ctx, src) {
      return src
    },

    // Optional: add remark plugins (operate on mdast)
    markdownPlugins(_ctx) {
      return []
    },

    // Optional: add rehype plugins (operate on hast)
    htmlPlugins(_ctx) {
      return []
    },

    // Optional: add external resources (CSS/JS)
    externalResources(_ctx) {
      return {}
    },
  }
}
```

**Important Constraints:**
- Do NOT create nested unist processors in markdownPlugins/htmlPlugins
- Use existing remark/rehype plugins from the ecosystem
- Only transform AST trees, never use `fs` operations
- For file operations, use emitters or `renderPage.tsx`

### Creating an Emitter Plugin

Template for `quartz/plugins/emitters/<name>.tsx`:

```typescript
import { QuartzEmitterPlugin } from "../types"
import { FilePath } from "../../util/path"

export const EmitterName: QuartzEmitterPlugin = () => {
  return {
    name: "EmitterName",

    async emit(ctx, content, resources) {
      const fps: FilePath[] = []

      // Your emission logic here
      // Can use fs operations, write files, etc.

      return fps
    },

    // Optional: handle incremental builds
    async partialEmit(ctx, content, resources, changeEvents) {
      return null // or implement partial rebuild
    },

    // Optional: specify components used
    getQuartzComponents(ctx) {
      return []
    },
  }
}
```

### Working with AST Trees

**mdast (Markdown AST):**
- Used in remark plugins
- Nodes: paragraph, heading, text, link, etc.
- Use `unist-util-visit` to traverse
- Use `mdast-util-*` packages for transformations

**hast (HTML AST):**
- Used in rehype plugins
- Nodes: element, text, etc.
- Use `hast-util-*` packages
- Can use `hastscript` to create elements

Example traversal:
```typescript
import { visit } from "unist-util-visit"

markdownPlugins(_ctx) {
  return [
    () => (tree, file) => {
      visit(tree, "text", (node) => {
        // Transform text nodes
      })
    }
  ]
}
```

### Testing Plugins

1. Make changes to plugin file
2. The dev server (`dev.ts`) watches for changes
3. Test with sample content in `content/`
4. Check build output in browser
5. Run `pnpm check` for validation
6. Run `tsc --noEmit` for type checking

### Adding Plugin to Configuration

Edit `quartz.config.ts` to include your plugin:

```typescript
import { TransformerName } from "./quartz/plugins/transformers/<name>"

// In plugins section:
transformers: [
  // ... existing transformers
  TransformerName(),
],
```

## Examples

### Example 1: Simple Filter - Hide WIP Content

```typescript
// quartz/plugins/filters/wip.ts
import { QuartzFilterPlugin } from "../types"

export const RemoveWIP: QuartzFilterPlugin = () => ({
  name: "RemoveWIP",
  shouldPublish(_ctx, [_tree, vfile]) {
    const isWIP = vfile.data?.frontmatter?.wip === true
    return !isWIP
  },
})
```

### Example 2: Text Transformer - Custom Syntax

```typescript
// quartz/plugins/transformers/customSyntax.ts
import { QuartzTransformerPlugin } from "../types"

export const CustomSyntax: QuartzTransformerPlugin = () => {
  return {
    name: "CustomSyntax",

    textTransform(_ctx, src) {
      // Transform ::keyword:: to <mark>keyword</mark>
      return src.replace(/::(\w+)::/g, '<mark>$1</mark>')
    },
  }
}
```

### Example 3: Remark Plugin - Add Reading Time

```typescript
// quartz/plugins/transformers/readingTime.ts
import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import readingTime from "reading-time"

export const ReadingTime: QuartzTransformerPlugin = () => {
  return {
    name: "ReadingTime",

    markdownPlugins(_ctx) {
      return [
        () => (tree, file) => {
          let text = ""
          visit(tree, "text", (node) => {
            text += node.value
          })

          const stats = readingTime(text)
          file.data.readingTime = stats.text
        }
      ]
    },
  }
}
```

## Common Packages

**AST Traversal:**
- `unist-util-visit` - Visit nodes
- `unist-util-select` - CSS-like selectors
- `unist-util-remove` - Remove nodes

**mdast Utilities:**
- `mdast-util-to-string` - Extract text
- `mdast-util-to-hast` - Convert to HTML AST
- `mdast-util-find-and-replace` - Find/replace

**hast Utilities:**
- `hast-util-to-string` - Extract text
- `hast-util-to-html` - Serialize to HTML
- `hastscript` - Create elements
- `hast-util-find-and-replace` - Find/replace

**Micromark (low-level):**
- `micromark-util-character` - Character utilities
- `micromark-factory-space` - Whitespace handling

## Notes

- All plugins use TypeScript with strict mode
- Follow 2-space indentation and ES modules
- Use PascalCase for plugin names
- Plugins are pure functions that return instances
- The build system uses workerpool for parallel processing
- Never use `fs` in transformers (breaks parallelization)
- Check existing plugins for patterns before creating new ones
- The `BuildCtx` contains configuration and helper methods
