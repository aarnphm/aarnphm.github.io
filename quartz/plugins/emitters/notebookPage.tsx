import { QuartzConfig } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import DepGraph from "../../depgraph"
import path from "path"
import fs from "node:fs"
import { glob } from "../../util/glob"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { Argv } from "../../util/ctx"
import { spawn } from "child_process"

const notebookFiles = async (argv: Argv, cfg: QuartzConfig) => {
  return await glob("**/*.ipynb", argv.directory, [...cfg.configuration.ignorePatterns])
}

const runConvertCommand = async (nbPath: string, targetSlug: string, outputDir: string) => {
  const command = "uvx"
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
    nbPath,
    "--log-level",
    "50",
    "--output",
    targetSlug,
    "--output-dir",
    outputDir,
  ]
  return spawn(command, args)
}

export const NotebookPage: QuartzEmitterPlugin = () => {
  return {
    name: "NotebookPage",
    getQuartzComponents() {
      return []
    },
    async getDependencyGraph() {
      return new DepGraph<FilePath>()
    },
    async emit({ argv, cfg }, _content, _resources): Promise<FilePath[]> {
      const outputDir = argv.output
      const fps = await notebookFiles(argv, cfg)
      const res: FilePath[] = []

      for (const fp of fps) {
        const src = joinSegments(argv.directory, fp) as FilePath
        const outputName = (slugifyFilePath(fp as FilePath, true) + ".html") as FilePath

        const dest = joinSegments(outputDir, outputName) as FilePath
        const dir = path.dirname(dest) as FilePath
        await fs.promises.mkdir(dir, { recursive: true }) // ensure dir exists
        await runConvertCommand(src, outputName, outputDir)
        res.push(dest)
      }
      return res
    },
  }
}
