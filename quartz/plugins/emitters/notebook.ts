import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import DepGraph from "../../depgraph"
import { glob } from "../../util/glob"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { Argv } from "../../util/ctx"
import { execFile } from "child_process"
import { write } from "./helpers"
import chalk from "chalk"
import { promisify } from "util"

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

  const { stdout } = await execFileAsync("uvx", args, {
    maxBuffer: 1024 * 1024 * 128,
  })
  return stdout
}

const processNotebook = (content: string): string => {
  content = content.replace('<body class="', '<body class="popover-hint ')
  return content
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
      const fpaths: Promise<FilePath>[] = []

      console.log(chalk.blue(`[emit:NotebookViewer] Processing ${fps.length} notebooks...`))

      const notebooks = fps.map((fp) => ({
        path: joinSegments(argv.directory, fp) as string,
        slug: slugifyFilePath(fp as FilePath, true),
      }))

      for (const [_, item] of notebooks.entries()) {
        const { path, slug } = item
        const content = await convertNotebook(path)
        fpaths.push(
          write({
            ctx,
            content: processNotebook(content),
            slug,
            ext: ".html",
          }),
        )
      }
      return await Promise.all(fpaths)
    },
  }
}