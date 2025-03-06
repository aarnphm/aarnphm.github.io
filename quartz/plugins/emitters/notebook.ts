import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import DepGraph from "../../depgraph"
import path from "path"
import fs from "node:fs/promises"
import { glob } from "../../util/glob"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { Argv } from "../../util/ctx"
import { spawn } from "child_process"
import { styleText } from "node:util"

const notebookFiles = async (argv: Argv, cfg: QuartzConfig) => {
  return await glob("**/*.ipynb", argv.directory, [...cfg.configuration.ignorePatterns])
}

function runConvertCommand(argv: Argv, nbPath: string, targetSlug: string, outputDir: string) {
  const command = process.env.CF_PAGES === "1" ? "python" : "uvx"
  const nbConvertArgs = [
    "--with",
    "jupyter-contrib-nbextensions",
    "--with",
    "notebook<7",
    "jupyter",
    "nbconvert",
    `--TemplateExporter.extra_template_basedirs=${joinSegments(process.cwd(), argv.directory, "templates")}`,
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
  return spawn(command, args)
}

const name = "NotebookViewer"
export const NotebookViewer: QuartzEmitterPlugin = () => {
  return {
    name,
    skipDuringServe: true,
    async getDependencyGraph() {
      return new DepGraph<FilePath>()
    },
    async emit({ argv, cfg }, _content, _resources): Promise<FilePath[]> {
      const outputDir = argv.output
      const fps = await notebookFiles(argv, cfg)
      const res: FilePath[] = []

      if (fps.length === 0 || process.env.VERCEL_ENV) return res

      let completed = 0
      let errors = 0
      const updateProgress = () => {
        const percent = Math.round((completed / fps.length) * 100)
        process.stdout.write(
          `\r[emit:NotebookViewer] Converting notebooks: ${completed}/${fps.length} (${percent}%)` +
            (errors > 0 ? styleText("yellow", ` (${errors} errors)`) : ""),
        )
      }

      for (const fp of fps) {
        const src = joinSegments(argv.directory, fp) as FilePath
        const outputName = (slugifyFilePath(fp as FilePath, true) + ".html") as FilePath
        const dest = joinSegments(outputDir, outputName) as FilePath
        const dir = path.dirname(dest) as FilePath

        try {
          await fs.mkdir(dir, { recursive: true })
          runConvertCommand(argv, src, outputName, outputDir)
          res.push(dest)
          completed++
          updateProgress()
        } catch (err) {
          console.error(styleText("red", `\n[emit:NotebookViewer] Error processing ${fp}:`), err)
          errors++
          updateProgress()
          continue
        }
      }

      if (argv.verbose) {
        console.log() // New line after progress
        const summaryColor = errors > 0 ? "yellow" : "green"
        console.log(
          styleText(
            summaryColor,
            `[emit:NotebookViewer] Completed conversion of ${completed} notebooks` +
              (errors > 0 ? ` (${errors} errors)` : ""),
          ),
        )
      }

      return res
    },
  }
}
