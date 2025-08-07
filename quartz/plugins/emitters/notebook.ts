import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import path from "path"
import fs from "node:fs/promises"
import { glob } from "../../util/glob"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { Argv } from "../../util/ctx"
import { spawn } from "child_process"
import { styleText } from "node:util"
import * as process from "process"

const notebookFiles = async (argv: Argv, cfg: QuartzConfig) => {
  return await glob("**/*.ipynb", argv.directory, [...cfg.configuration.ignorePatterns])
}

// Special case for template path resolution
// Use path.resolve for absolute paths instead of process.cwd()
function getTemplatePath(argv: Argv): string {
  return path.resolve(argv.directory, "templates")
}

function runConvertCommand(argv: Argv, nbPath: string, targetSlug: string, outputDir: string) {
  const command = process.env.CF_PAGES === "1" ? "python" : "uvx"
  const nbConvertArgs = [
    "--with",
    "jupyter-contrib-nbextensions",
    "--with",
    "notebook<7",
    "--from",
    "jupyter-core",
    "jupyter",
    "nbconvert",
    `--TemplateExporter.extra_template_basedirs=${getTemplatePath(argv)}`,
    "--to",
    "html",
    "--template=quartz-notebooks",
    nbPath,
    "--log-level",
    "50",
    "--output",
    targetSlug,
    "--output-dir",
    outputDir,
  ]

  // Special case for Cloudflare Pages
  const args =
    process.env.CF_PAGES === "1" ? ["-m", "uv", "tool", "run", ...nbConvertArgs] : nbConvertArgs

  return spawn(command, args, {
    env: { ...process.env }, // Ensure we pass environment variables
  })
}

export const NotebookViewer: QuartzEmitterPlugin = () => {
  return {
    name: "NotebookViewer",
    async *partialEmit() {},
    async *emit({ argv, cfg }) {
      if (argv.watch) return []

      const fps = await notebookFiles(argv, cfg)

      for (const fp of fps) {
        const src = joinSegments(argv.directory, fp) as FilePath
        const outputName = (slugifyFilePath(fp as FilePath, true) + ".html") as FilePath
        const dest = joinSegments(argv.output, outputName) as FilePath
        const dir = path.dirname(dest) as FilePath

        try {
          await fs.mkdir(dir, { recursive: true })

          // Create a simple promise that resolves when the child process exits
          const result = new Promise<FilePath>((resolve, reject) => {
            const proc = runConvertCommand(argv, src, outputName, argv.output)

            proc.on("error", (err) => {
              console.error(`Failed to start subprocess for ${fp}:`, err)
              reject(err)
            })

            proc.on("exit", (code) => {
              if (code === 0) {
                resolve(dest)
              } else {
                reject(new Error(`Process exited with code ${code}`))
              }
            })
          })

          yield result
        } catch (err) {
          console.error(styleText("red", `\n[emit:NotebookViewer] Error processing ${fp}:`), err)
          continue
        }
      }
    },
  }
}
