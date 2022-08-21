import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import path from "path"
import fs from "node:fs/promises"
import { glob } from "../../util/glob"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { Argv } from "../../util/ctx"
import { spawn } from "child_process"
import { availableParallelism } from "node:os"
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
      if (argv.watch && !argv.force) return []

      const fps = await notebookFiles(argv, cfg)

      if (fps.length === 0) {
        return
      }

      const resolveWorkerLimit = () => {
        if (argv.concurrency && argv.concurrency > 0) {
          return argv.concurrency
        }

        try {
          return availableParallelism()
        } catch {
          return 1
        }
      }

      const maxWorkers = Math.max(1, Math.min(resolveWorkerLimit(), fps.length))

      type NotebookTaskResult =
        | { status: "fulfilled"; dest: FilePath; fp: FilePath; id: number }
        | { status: "rejected"; error: unknown; fp: FilePath; id: number }

      let taskId = 0
      const inflight = new Map<number, Promise<NotebookTaskResult>>()

      const convertNotebook = async (fp: FilePath): Promise<FilePath> => {
        const src = joinSegments(argv.directory, fp) as FilePath
        const outputName = (slugifyFilePath(fp as FilePath, true) + ".html") as FilePath
        const dest = joinSegments(argv.output, outputName) as FilePath
        const dir = path.dirname(dest) as FilePath

        await fs.mkdir(dir, { recursive: true })

        await new Promise<void>((resolve, reject) => {
          const proc = runConvertCommand(argv, src, outputName, argv.output)

          proc.on("error", reject)

          proc.on("exit", (code) => {
            if (code === 0) {
              resolve()
            } else {
              reject(new Error(`Process exited with code ${code}`))
            }
          })
        })

        return dest
      }

      const enqueue = (fp: FilePath) => {
        const id = taskId++
        const task = convertNotebook(fp)
          .then((dest) => ({ status: "fulfilled" as const, dest, fp, id }))
          .catch((error) => ({ status: "rejected" as const, error, fp, id }))
        inflight.set(id, task)
      }

      const takeNext = async (): Promise<NotebookTaskResult | null> => {
        if (inflight.size === 0) {
          return null
        }

        const settled = await Promise.race(inflight.values())
        inflight.delete(settled.id)
        return settled
      }

      for (const fp of fps) {
        enqueue(fp)
        if (inflight.size >= maxWorkers) {
          const result = await takeNext()
          if (!result) {
            continue
          }

          if (result.status === "fulfilled") {
            yield result.dest
          } else {
            console.error(
              styleText("red", `\n[emit:NotebookViewer] Error processing ${result.fp}:`),
              result.error,
            )
          }
        }
      }

      while (inflight.size > 0) {
        const result = await takeNext()
        if (!result) {
          continue
        }

        if (result.status === "fulfilled") {
          yield result.dest
        } else {
          console.error(
            styleText("red", `\n[emit:NotebookViewer] Error processing ${result.fp}:`),
            result.error,
          )
        }
      }
    },
  }
}
