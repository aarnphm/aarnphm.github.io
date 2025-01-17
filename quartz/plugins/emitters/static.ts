import { FilePath, QUARTZ, joinSegments } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import fs from "node:fs/promises"
import path from "node:path"
import { glob } from "../../util/glob"
import DepGraph from "../../depgraph"

export const Static: QuartzEmitterPlugin = () => ({
  name: "Static",
  getQuartzComponents() {
    return []
  },
  async getDependencyGraph({ argv, cfg }, _content, _resources) {
    const graph = new DepGraph<FilePath>()

    const staticPath = joinSegments(QUARTZ, "static")
    const fps = await glob("**", staticPath, cfg.configuration.ignorePatterns)
    for (const fp of fps) {
      graph.addEdge(
        joinSegments("static", fp) as FilePath,
        joinSegments(argv.output, "static", fp) as FilePath,
      )
    }

    return graph
  },
  async emit({ argv, cfg }, _content, _resources): Promise<FilePath[]> {
    const staticPath = joinSegments(QUARTZ, "static")
    const outputStaticPath = joinSegments(argv.output, "static")
    const fps = await glob("**", staticPath, cfg.configuration.ignorePatterns)
    await fs.mkdir(outputStaticPath, { recursive: true })

    // Copy files individually if they don't exist
    for (const fp of fps) {
      const srcPath = joinSegments(staticPath, fp)
      const destPath = joinSegments(outputStaticPath, fp)

      try {
        // Check if destination exists
        await fs.access(destPath)
      } catch {
        // File doesn't exist, create directory and copy
        await fs.mkdir(path.dirname(destPath), { recursive: true })
        await fs.cp(srcPath, destPath, { recursive: true, dereference: true })
      }
    }

    return fps.map((fp) => joinSegments(argv.output, "static", fp)) as FilePath[]
  },
})
