import fs from "node:fs/promises"
import path from "path"
import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { Argv } from "../../util/ctx"
import { glob } from "../../util/glob"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"

const filesToCopy = async (argv: Argv, cfg: QuartzConfig) => {
  // glob all non MD files in content folder and copy it over
  const patterns = ["**/*.md", "**/*.base", "**/*.ipynb", ...cfg.configuration.ignorePatterns]

  // Skip PDFs when running in Cloudflare Pages
  if (process.env.CF_PAGES === "1" || argv.watch) {
    patterns.push("**/*.pdf", "**.ddl", "**.mat")
  }

  return await glob("**", argv.directory, patterns)
}

const copyFile = async (argv: Argv, fp: FilePath) => {
  const src = joinSegments(argv.directory, fp) as FilePath

  const name = slugifyFilePath(fp)
  const dest = joinSegments(argv.output, name) as FilePath

  // ensure dir exists
  const dir = path.dirname(dest) as FilePath
  await fs.mkdir(dir, { recursive: true })

  await fs.copyFile(src, dest)
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
