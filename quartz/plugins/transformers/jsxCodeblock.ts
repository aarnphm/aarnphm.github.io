import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import { Code, Root as MdRoot } from "mdast"
import { VFile } from "vfile"
import { unified } from "unified"
import remarkParse from "remark-parse"
import remarkMdx from "remark-mdx"
import remarkRehype from "remark-rehype"
import { Element, Properties, Root as HastRoot } from "hast"
import { htmlToJsx } from "../../util/jsx"
import render from "preact-render-to-string"
import { fromHtml } from "hast-util-from-html"
import { FilePath } from "../../util/path"

type JsxBlock = {
  id: string
  code: string
  imports: string[]
}

const IMPORT_REGEX = /imports\s*=\s*\{([^}]*)\}/i

function parseImports(meta: string | null | undefined): string[] {
  if (!meta) return []
  const match = meta.match(IMPORT_REGEX)
  if (!match) return []
  return match[1]
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
}

type MdxAttribute =
  | {
      type: "mdxJsxAttribute"
      name: string
      value?: string | { type: "mdxJsxAttributeValueExpression"; value: string }
    }
  | {
      type: "mdxJsxExpressionAttribute"
      value: string
    }

function parseAttributeExpression(raw: string): unknown {
  const trimmed = raw.trim()
  if (!trimmed.length) return null
  if (trimmed === "true") return true
  if (trimmed === "false") return false
  if (trimmed === "null") return null
  if (trimmed === "undefined") return undefined
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return Number(trimmed)
  }

  const startsWithQuote = trimmed.startsWith("'") || trimmed.startsWith('"')
  const endsWithQuote = trimmed.endsWith("'") || trimmed.endsWith('"')
  if (startsWithQuote && endsWithQuote) {
    return trimmed.slice(1, -1)
  }

  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    try {
      return JSON.parse(trimmed)
    } catch {
      return trimmed
    }
  }

  try {
    return JSON.parse(trimmed)
  } catch {
    return trimmed
  }
}

function normalizePropValue(value: unknown): Properties[string] {
  if (value === null || value === undefined) return undefined
  if (Array.isArray(value)) {
    return value.map((item) => String(item))
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value
  }
  if (typeof value === "object") {
    return JSON.stringify(value)
  }
  return String(value)
}

function attributesToProps(attributes: MdxAttribute[] = []): Properties {
  const props: Properties = {}
  for (const attr of attributes) {
    if (attr.type === "mdxJsxAttribute") {
      if (!attr.name) continue
      if (attr.value === null || attr.value === undefined) {
        props[attr.name] = true
        continue
      }
      if (typeof attr.value === "string") {
        props[attr.name] = attr.value
        continue
      }
      const parsed = parseAttributeExpression(attr.value.value ?? "")
      props[attr.name] = normalizePropValue(parsed)
    }
  }

  return props
}

const mdxProcessor = unified()
  .use(remarkParse)
  .use(remarkMdx)
  .use(remarkRehype, {
    allowDangerousHtml: true,
    handlers: {
      mdxJsxFlowElement: (state: any, node: any) => ({
        type: "element",
        tagName: node.name,
        properties: attributesToProps(node.attributes as MdxAttribute[]),
        children: state.all(node),
      }),
      mdxJsxTextElement: (state: any, node: any) => ({
        type: "element",
        tagName: node.name,
        properties: attributesToProps(node.attributes as MdxAttribute[]),
        children: state.all(node),
      }),
      mdxFlowExpression: (_state: any, node: any) => ({
        type: "text",
        value: String(node.value ?? ""),
      }),
      mdxTextExpression: (_state: any, node: any) => ({
        type: "text",
        value: String(node.value ?? ""),
      }),
    },
  })

function renderJsxSnippet(snippet: string): HastRoot | null {
  try {
    return mdxProcessor.runSync(mdxProcessor.parse(snippet)) as HastRoot
  } catch {
    return null
  }
}

function getJsxBlockMap(file: VFile): Map<string, JsxBlock> {
  const blocks = (file.data.jsxBlocks as JsxBlock[] | undefined) ?? []
  return new Map(blocks.map((block) => [block.id, block]))
}

function ensureJsxBlocks(file: VFile): JsxBlock[] {
  if (!file.data.jsxBlocks) {
    file.data.jsxBlocks = []
  }
  return file.data.jsxBlocks as JsxBlock[]
}

export const JsxCodeblock: QuartzTransformerPlugin = () => {
  let counter = 0

  return {
    name: "JsxCodeblock",
    markdownPlugins() {
      return [
        () => (tree: MdRoot, file: VFile) => {
          visit(tree, "code", (node: Code) => {
            if (node.lang !== "jsx") return
            const imports = parseImports(node.meta)
            if (imports.length === 0) return

            const id = `jsx-block-${counter++}`

            const blocks = ensureJsxBlocks(file)
            blocks.push({ id, code: node.value, imports })

            node.data ??= {}
            node.data.hProperties ??= {}
            node.data.hProperties["data-jsx-block-id"] = id
          })
        },
      ]
    },
    htmlPlugins() {
      return [
        () => (tree: HastRoot, file: VFile) => {
          const blocksMap = getJsxBlockMap(file)
          if (blocksMap.size === 0) return

          const collectedImports = new Set<string>(
            (file.data.jsxImports as string[] | undefined) ?? [],
          )

          visit(tree, "element", (node: Element, index, parent) => {
            if (!parent || index === undefined) return
            if (node.tagName !== "pre" || !node.children?.length) return

            const codeNode = node.children.find(
              (child): child is Element => child.type === "element" && child.tagName === "code",
            )
            if (!codeNode) return

            const rawId = codeNode.properties?.["data-jsx-block-id"]
            const blockIdValue = Array.isArray(rawId) ? rawId[0] : rawId
            if (typeof blockIdValue !== "string") return

            const block = blocksMap.get(blockIdValue)
            if (!block) return

            const renderedRoot = renderJsxSnippet(block.code)
            if (!renderedRoot) return

            const jsxNode = htmlToJsx(
              (file.data.filePath ?? (file.path as FilePath)) as FilePath,
              renderedRoot,
            )
            if (!jsxNode) return
            const htmlString = render(jsxNode)
            const renderedHast = fromHtml(htmlString, { fragment: true }) as HastRoot

            parent.children.splice(index, 1, ...renderedHast.children)
            block.imports.forEach((name) => collectedImports.add(name))
            blocksMap.delete(blockIdValue)
          })

          if (collectedImports.size > 0) {
            file.data.jsxImports = Array.from(collectedImports)
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    jsxImports?: string[]
    jsxBlocks?: JsxBlock[]
  }
}
