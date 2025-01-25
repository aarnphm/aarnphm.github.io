import { FilePath, resolveRelative, simplifySlug } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"
import DepGraph from "../../depgraph"

export const AliasRedirects: QuartzEmitterPlugin = () => ({
  name: "AliasRedirects",
  getQuartzComponents: () => [],
  getDependencyGraph: async (_ctx, _content, _resource) => new DepGraph<FilePath>(),
  async emit(ctx, content, _resources): Promise<FilePath[]> {
    const fps: FilePath[] = []

    for (const [_, file] of content) {
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
