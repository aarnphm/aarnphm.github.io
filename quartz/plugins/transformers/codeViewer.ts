import { QuartzTransformerPlugin } from "../../types/plugin"
import { Code, Root } from "mdast"
import { BuildCtx } from "../../util/ctx"
import path from "path"
import { FilePath } from "../../util/path"
import { readFile } from "fs/promises"
import { visit, SKIP } from "unist-util-visit"
import type { Wikilink } from "../../util/wikilinks"
import type { Parent } from "unist"

type Options = {
  /** File extensions to treat as code files (lowercase, with dot). */
  exts?: string[]
}

const DEFAULT_EXTS = new Set<string>([
  ".py",
  ".rs",
  ".go",
  ".c",
  ".cu",
  ".cc",
  ".cpp",
  ".h",
  ".hpp",
  ".m",
  ".mm",
  ".java",
  ".kt",
  ".swift",
  ".scala",
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".sh",
  ".bash",
  ".zsh",
  ".fish",
  ".sql",
  ".yaml",
  ".yml",
  ".toml",
  ".json",
  ".mdx",
  ".css",
  ".scss",
  ".hs",
  ".rb",
  ".php",
])

function languageFromExt(ext: string): string | undefined {
  const e = ext.replace(/^\./, "").toLowerCase()
  switch (e) {
    case "py":
      return "python"
    case "ts":
    case "tsx":
      return e
    case "js":
    case "jsx":
      return e
    case "rs":
      return "rust"
    case "go":
      return "go"
    case "c":
      return "c"
    case "cc":
    case "cpp":
    case "hpp":
      return "cpp"
    case "h":
      return "c"
    case "m":
    case "mm":
      return "objective-c"
    case "java":
      return "java"
    case "kt":
      return "kotlin"
    case "swift":
      return "swift"
    case "scala":
      return "scala"
    case "sh":
    case "bash":
    case "zsh":
    case "fish":
      return "bash"
    case "sql":
      return "sql"
    case "yaml":
    case "yml":
      return "yaml"
    case "toml":
      return "toml"
    case "json":
      return "json"
    case "mdx":
      return "mdx"
    case "css":
    case "scss":
      return e
    case "hs":
      return "haskell"
    case "rb":
      return "ruby"
    case "php":
      return "php"
    case "cu":
      return "cuda"
    default:
      return e
  }
}

function resolveToRelativePath(
  ctx: BuildCtx,
  target: string,
  currentMdRel: string,
): FilePath | null {
  const relDir = path.posix.dirname(currentMdRel)
  const targetBase = path.posix.basename(target)

  // 1) sibling resolution
  const sibling = path.posix.join(relDir, target)
  if (ctx.allFiles.includes(sibling as FilePath)) {
    return sibling as FilePath
  }

  // 2) explicit path from vault root
  if (target.includes("/") && ctx.allFiles.includes(target as FilePath)) {
    return target as FilePath
  }

  // 3) basename fallback
  const match = ctx.allFiles.find((fp) => path.posix.basename(fp) === targetBase)
  return (match ?? null) as FilePath | null
}

async function readCodeFile(ctx: BuildCtx, resolvedRel: FilePath) {
  const abs = path.posix.join(ctx.argv.directory, resolvedRel)
  const buf = await readFile(abs)
  return buf.toString("utf8")
}

export const CodeViewer: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { exts: Array.from(DEFAULT_EXTS), ...userOpts }
  const exts = new Set(opts.exts!.map((e) => e.toLowerCase()))

  // We implement as a Markdown transformer that preempts OFM on code embeds.
  // This ensures resulting <pre><code> are present for later syntax-highlighting.
  return {
    name: "CodeViewer",
    markdownPlugins(ctx) {
      return [
        () => {
          return async (tree: Root, file) => {
            // Replace embed wikilink nodes with placeholder code blocks so later passes can hydrate
            visit(tree, "wikilink", (node, index, parent) => {
              if (index === undefined || !parent) return
              const wikilinkNode = node as unknown as Wikilink
              const data = wikilinkNode.data?.wikilink
              if (!data?.embed) return

              const fp = (data.target ?? "").trim()
              if (!fp) return

              const ext = path.extname(fp).toLowerCase()
              if (!ext || !exts.has(ext)) return

              const lang = languageFromExt(ext)
              const base = path.posix.basename(fp)
              const codeNode: Code = {
                type: "code",
                lang,
                meta: `title="${base}"`,
                value: "",
              }

              const dataAny = codeNode as unknown as { data?: Record<string, any> }
              dataAny.data = { ...(dataAny.data ?? {}), codeTranscludeTarget: fp }

              const parentNode = parent as Parent
              if (!parentNode.children) return
              parentNode.children.splice(index, 1, codeNode)
              return [SKIP, index]
            })

            // Resolve and populate code content
            const promises: Promise<void>[] = []
            visit(tree, "code", (node: Code) => {
              const dataAny = node as unknown as { data?: Record<string, any> }
              const target = dataAny.data?.codeTranscludeTarget as string | undefined
              if (!target) return
              const currentRel = file.data.relativePath as string
              const resolved = resolveToRelativePath(ctx as BuildCtx, target, currentRel)
              if (!resolved) return
              const titleBase = path.posix.basename(resolved)
              const ext = path.extname(resolved)
              node.lang = languageFromExt(ext)
              node.meta = node.meta
                ? `${node.meta} path="${resolved}"`
                : `title="${titleBase}" path="${resolved}"`

              promises.push(
                readCodeFile(ctx as BuildCtx, resolved).then((content) => {
                  node.value = content
                  const deps: string[] = (file.data.codeDependencies as string[] | undefined) ?? []
                  if (!deps.includes(resolved)) {
                    file.data.codeDependencies = [...deps, resolved]
                  }
                }),
              )
            })
            await Promise.all(promises)
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    codeDependencies: string[]
  }
}
