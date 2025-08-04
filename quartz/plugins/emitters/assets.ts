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

const copyFile = async (argv: Argv, fp: FilePath) => {
  const ext = path.extname(fp)
  const src = joinSegments(argv.directory, fp) as FilePath
  const name = (slugifyFilePath(fp, true) + (ext.includes("pdf") ? "" : ext)) as FilePath
  const dest = joinSegments(argv.output, name) as FilePath

  const srcStat = await fs.stat(src)
  let shouldCopy = true

  try {
    const destStat = await fs.stat(dest)
    // Only copy if source is newer thescapeHTML(toHtml(tree as Root, { allowDangerousHtml: true }))an destination
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

  return dest
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: "Assets",
    async *emit({ argv, cfg }) {
      const fps = await filesToCopy(argv, cfg)
      for (const fp of fps) {
        yield copyFile(argv, fp)
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      for (const changeEvent of changeEvents) {
        const ext = path.extname(changeEvent.path)
        if (ext === ".md") continue

        if (changeEvent.type === "add" || changeEvent.type === "change") {
          yield copyFile(ctx.argv, changeEvent.path)
        } else if (changeEvent.type === "delete") {
          const name = slugifyFilePath(changeEvent.path)
          const dest = joinSegments(ctx.argv.output, name) as FilePath
          await fs.unlink(dest)
        }
      }
    },
  }
}
