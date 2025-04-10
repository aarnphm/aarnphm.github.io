import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import path from "path"
import fs from "node:fs/promises"
import { glob } from "../../util/glob"
import { Argv } from "../../util/ctx"
import { QuartzConfig } from "../../cfg"

const filesToCopy = async (argv: Argv, cfg: QuartzConfig) => {
  // glob all non MD files in content folder and copy it over
  const patterns = ["**/*.md", ...cfg.configuration.ignorePatterns]

  // Skip PDFs when running in Cloudflare Pages
  if (process.env.CF_PAGES === "1") {
    patterns.push("**/*.pdf", "**.ddl", "**.mat")
  }

  return await glob("**", argv.directory, patterns)
}

const name = "Assets"
export const Assets: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit({ argv, cfg }, _content, _resources) {
      const assetsPath = argv.output
      const fps = await filesToCopy(argv, cfg)
      for (const fp of fps) {
        const ext = path.extname(fp)
        const src = joinSegments(argv.directory, fp) as FilePath
        const name = (slugifyFilePath(fp as FilePath, true) + ext) as FilePath
        const dest = joinSegments(assetsPath, name) as FilePath

        try {
          // Check if destination exists
          const srcStat = await fs.stat(src)
          let shouldCopy = true

          try {
            const destStat = await fs.stat(dest)
            // Only copy if source is newer than destination
            shouldCopy = srcStat.mtimeMs > destStat.mtimeMs
          } catch {
            // Destination doesn't exist, should copy
            shouldCopy = true
          }

          if (shouldCopy) {
            const dir = path.dirname(dest) as FilePath
            await fs.mkdir(dir, { recursive: true })
            await fs.copyFile(src, dest)
          }

          yield dest
        } catch (err) {
          console.warn(`[emit:${name}] Failed to process asset: ${fp}`, err)
        }
      }
    },
  }
}
