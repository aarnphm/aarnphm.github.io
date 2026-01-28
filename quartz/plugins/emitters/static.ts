import fs from "node:fs/promises"
import { dirname } from "path"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { glob } from "../../util/glob"
import { FilePath, QUARTZ, joinSegments } from "../../util/path"

export const Static: QuartzEmitterPlugin = () => ({
  name: "Static",
  async *emit({ argv, cfg }, _content, _resources) {
    const staticPath = joinSegments(QUARTZ, "static")
    const outputStaticPath = joinSegments(argv.output, "static")
    const fps = await glob("**", staticPath, cfg.configuration.ignorePatterns)
    await fs.mkdir(outputStaticPath, { recursive: true })
    for (const fp of fps) {
      const src = joinSegments(staticPath, fp) as FilePath
      const dest = joinSegments(outputStaticPath, fp) as FilePath
      await fs.mkdir(dirname(dest), { recursive: true })
      await fs.copyFile(src, dest)
      yield dest
    }
  },
})
