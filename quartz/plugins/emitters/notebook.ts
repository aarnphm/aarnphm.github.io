import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import DepGraph from "../../depgraph"
import { glob } from "../../util/glob"
import { FilePath, FullSlug, joinSegments, slugifyFilePath } from "../../util/path"
import { Argv } from "../../util/ctx"
import { execFile } from "child_process"
import { write } from "./helpers"
import chalk from "chalk"
import { promisify } from "util"
import { homedir } from "os"

const notebookFiles = async (argv: Argv, cfg: QuartzConfig) => {
  return await glob("**/*.ipynb", argv.directory, [...cfg.configuration.ignorePatterns])
}

const execFileAsync = promisify(execFile)

async function convertNotebook(nbPath: string) {
  const args = [
    "--with",
    "jupyter-contrib-nbextensions",
    "--with",
    "notebook<7",
    "jupyter",
    "nbconvert",
    "--to",
    "html",
    "--template",
    "lab",
    "--stdout",
    "--log-level",
    "50",
    nbPath,
  ]

  const { stdout } = await execFileAsync(`${homedir()}/.local/bin/uvx`, args, {
    maxBuffer: 1024 * 1024 * 128,
  })
  return stdout
}

const processNotebook = (content: string): string => {
  content = content.replace('<body class="', '<body class="popover-hint ')
  return content
}

async function batchConvert(
  notebooks: { path: string; slug: string }[],
  concurrency: number = 4,
): Promise<{ slug: string; content: string }[]> {
  const results: { slug: string; content: string }[] = []

  // Process notebooks in chunks to control concurrency
  for (let i = 0; i < notebooks.length; i += concurrency) {
    const chunk = notebooks.slice(i, i + concurrency)
    const chunkPromises = chunk.map(async ({ path, slug }) => {
      try {
        const content = await convertNotebook(path)
        return { slug, content: processNotebook(content) }
      } catch (err) {
        console.error(chalk.red(`Error processing ${path}: ${(err as Error).message}`))
        return null
      }
    })

    const chunkResults = await Promise.all(chunkPromises)
    results.push(...chunkResults.filter((r): r is { slug: string; content: string } => r !== null))
  }

  return results
}

export const NotebookViewer: QuartzEmitterPlugin = () => {
  return {
    name: "NotebookViewer",
    getQuartzComponents() {
      return []
    },
    async getDependencyGraph() {
      return new DepGraph<FilePath>()
    },
    async emit(ctx, _content, _resources): Promise<FilePath[]> {
      const { argv, cfg } = ctx
      const fps = await notebookFiles(argv, cfg)
      if (fps.length === 0) return []

      console.log(chalk.blue(`[emit:NotebookViewer] Processing ${fps.length} notebooks...`))

      const notebooks = fps.map((fp) => ({
        path: joinSegments(argv.directory, fp) as string,
        slug: slugifyFilePath(fp as FilePath),
      }))

      const results = await batchConvert(notebooks, ctx.argv.concurrency ?? 4)

      const res = await Promise.all(
        results.map(async ({ slug, content }) => {
          return await write({
            ctx,
            content,
            slug: slug as FullSlug,
            ext: "",
          })
        }),
      )

      console.log(
        chalk.green(`[emit:NotebookViewer] Successfully processed ${res.length} notebooks`),
      )
      return res
    },
  }
}
