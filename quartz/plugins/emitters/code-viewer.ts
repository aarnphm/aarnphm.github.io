import { QuartzEmitterPlugin } from "../types"
import { FilePath, FullSlug, joinSegments, slugifyFilePath } from "../../util/path"
import path from "path"
import fs from "fs"
import { glob } from "../../util/glob"
import { TEXT_EXTENSIONS, MIME_MAPPINGS } from "../../util/mime"
import DepGraph from "../../depgraph"
import { GlobalConfiguration } from "../../cfg"
import { write } from "./helpers"
import { unescapeHTML } from "../../util/escape"

function generateCodeViewer(
  cfg: GlobalConfiguration,
  slug: FullSlug,
  code: string,
  ext: string,
): string {
  const language = MIME_MAPPINGS[`${ext}`][1] || "plaintext"
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <meta name="robots" content="noindex, nofollow">
  <title>${unescapeHTML(slug)}</title>
  <style>
    :root {
      color-scheme: light dark;
    }
    body {
      margin: 0;
      padding: 1rem;
      font-family: "${cfg.theme.typography.code}", ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
      line-height: 1.5;
      tab-size: 2;
    }
    pre {
      margin: 0;
      word-wrap: break-word;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
<article class="popover-hint">
<pre language="${language}">
${code}
</pre>
</article>
</body>
</html>`
}

export const CodeViewer: QuartzEmitterPlugin = () => {
  return {
    name: "CodeViewer",
    getQuartzComponents() {
      return []
    },
    async getDependencyGraph(ctx, _content, _resources) {
      const { argv } = ctx
      const graph = new DepGraph<FilePath>()

      // Glob all relevant file types
      const fps = await glob(`**/*.*`, argv.directory, ctx.cfg.configuration.ignorePatterns)

      for (const fp of fps) {
        const ext = path.extname(fp)
        if (!TEXT_EXTENSIONS.has(ext)) continue

        const src = joinSegments(argv.directory, fp) as FilePath

        // Both the original file and its HTML viewer
        const viewerFile = joinSegments(argv.output, `${fp}.viewer.html`) as FilePath
        const originalFile = joinSegments(argv.output, fp) as FilePath

        graph.addEdge(src, viewerFile)
        graph.addEdge(src, originalFile)
      }

      return graph
    },
    async emit(ctx, _content, _resources): Promise<FilePath[]> {
      const { argv } = ctx
      const fps = await glob(`**/*.*`, argv.directory, ctx.cfg.configuration.ignorePatterns)
      const res: FilePath[] = []

      for (const fp of fps) {
        const ext = path.extname(fp)
        if (!TEXT_EXTENSIONS.has(path.extname(fp))) continue

        const src = joinSegments(argv.directory, fp) as FilePath

        // Read the source code
        const code = await fs.promises.readFile(src, "utf-8")
        const slug = slugifyFilePath(fp as FilePath)

        // Create the viewer HTML
        const viewer = await write({
          ctx,
          content: generateCodeViewer(ctx.cfg.configuration, slug, code, ext),
          slug,
          ext: "",
        })

        res.push(viewer)
      }

      return res
    },
  }
}
