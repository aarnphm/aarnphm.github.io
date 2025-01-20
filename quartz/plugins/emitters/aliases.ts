import { FilePath, joinSegments, resolveRelative, simplifySlug } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"
import DepGraph from "../../depgraph"
import { getAliasSlugs } from "../transformers/frontmatter"

export const AliasRedirects: QuartzEmitterPlugin = () => ({
  name: "AliasRedirects",
  getQuartzComponents: () => [],
  async getDependencyGraph(ctx, content, _resources) {
    const graph = new DepGraph<FilePath>()

    const { argv } = ctx
    for (const [_tree, file] of content) {
      for (let slug of getAliasSlugs(file.data.aliases ?? [], argv, file)) {
        graph.addEdge(file.data.filePath!, joinSegments(argv.output, slug + ".html") as FilePath)
      }
    }

    return graph
  },
  async emit(ctx, content, _resources): Promise<FilePath[]> {
    const fps: FilePath[] = []

    for (const [_tree, file] of content) {
      const ogSlug = simplifySlug(file.data.slug!)

      if (file.data.aliases && file.data.aliases?.length > 0) {
        for (const slug of file.data.aliases!) {
          const redirUrl = resolveRelative(slug, file.data.slug!)
          const fp = await write({
            ctx,
            content: `
<!DOCTYPE html>
<html lang="en-us">
<head>
<title>${ogSlug}</title>
<link rel="canonical" href="${redirUrl}">
<meta name="robots" content="noindex">
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=${redirUrl}">
</head>
</html>
`,
            slug,
            ext: ".html",
          })

          fps.push(fp)
        }
      }
    }
    return fps
  },
})
