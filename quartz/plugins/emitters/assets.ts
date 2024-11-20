import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import path from "path"
import fs from "fs"
import { glob } from "../../util/glob"
import DepGraph from "../../depgraph"
import { Argv } from "../../util/ctx"
import { QuartzConfig } from "../../cfg"
import chalk from "chalk"

const NAME = "Assets"

const filesToCopy = async (argv: Argv, cfg: QuartzConfig) => {
  // glob all non MD, images files in content folder and copy it over
  return await glob("**", argv.directory, ["**/*.md", ...cfg.configuration.ignorePatterns])
}

async function ensureDirectoryExists(dir: string): Promise<boolean> {
  try {
    await fs.promises.mkdir(dir, { recursive: true })
    return true
  } catch (error) {
    console.error(chalk.red(`[emit:${NAME}] Failed to create directory "${dir}":`), error)
    return false
  }
}

async function checkFilePermissions(filePath: string): Promise<boolean> {
  try {
    // Check if we can access the file
    await fs.promises.access(filePath, fs.constants.R_OK | fs.constants.W_OK)
    return true
  } catch {
    return false
  }
}

interface RetryOptions {
  initialDelay?: number
  maxDelay?: number
  maxRetries?: number
  retryOnPermissionDenied?: boolean
}

const defaultRetryOptions: RetryOptions = {
  initialDelay: 100, // Start with 100ms delay
  maxDelay: 10000, // Don't wait longer than 10 seconds
  maxRetries: 3, // Try 3 times by default
  retryOnPermissionDenied: true, // Retry even on permission errors
}

async function copyFileWithRetry(
  src: string,
  dest: string,
  options: RetryOptions = {},
): Promise<boolean> {
  const opts = { ...defaultRetryOptions, ...options }
  let delay = opts.initialDelay!

  for (let attempt = 1; attempt <= opts.maxRetries!; attempt++) {
    try {
      // First check if we have necessary permissions
      if (!(await checkFilePermissions(src))) {
        if (!opts.retryOnPermissionDenied) {
          console.error(chalk.yellow(`[emit:${NAME}] Permission denied for source file: ${src}`))
          return false
        }
      }

      // Ensure the destination directory exists
      const destDir = path.dirname(dest)
      if (!(await ensureDirectoryExists(destDir))) {
        return false
      }

      // Attempt to copy the file
      await fs.promises.copyFile(src, dest)

      // If this wasn't our first attempt, log the success after retry
      if (attempt > 1) {
        console.log(
          chalk.green(`[emit:${NAME}] Successfully copied file after ${attempt} attempts: ${src}`),
        )
      }

      return true
    } catch (error) {
      const lastAttempt = attempt === opts.maxRetries

      if (lastAttempt) {
        console.error(
          chalk.red(`[emit:${NAME}] Failed to copy file after ${opts.maxRetries} attempts:`),
        )
        console.error(chalk.gray(`\tFrom: ${src}`))
        console.error(chalk.gray(`\tTo: ${dest}`))
        console.error(chalk.gray(`\tError: ${error}`))
        return false
      }

      // Log retry attempt
      console.warn(
        chalk.yellow(
          `Attempt ${attempt}/${opts.maxRetries} failed, retrying in ${delay}ms: ${src}`,
        ),
      )

      // Wait with exponential backoff
      await new Promise((resolve) => setTimeout(resolve, delay))

      // Exponential backoff with jitter
      // Add random jitter of ±10% to help avoid thundering herd problems
      const jitter = delay * (0.9 + Math.random() * 0.2)
      delay = (Math.min(delay * 2, opts.maxDelay!) * jitter) / delay
    }
  }
  return false
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: NAME,
    getQuartzComponents() {
      return []
    },
    async getDependencyGraph(ctx, _content, _resources) {
      const { argv, cfg } = ctx
      const graph = new DepGraph<FilePath>()

      const fps = await filesToCopy(argv, cfg)

      for (const fp of fps) {
        const ext = path.extname(fp)
        const src = joinSegments(argv.directory, fp) as FilePath
        const name = (slugifyFilePath(fp as FilePath, true) + ext) as FilePath

        const dest = joinSegments(argv.output, name) as FilePath

        graph.addEdge(src, dest)

        if (await checkFilePermissions(src)) {
          graph.addEdge(src, dest)
        } else {
          console.warn(chalk.yellow(`[emit:${NAME}] Skipping file due to permissions: ${src}`))
        }
      }
      return graph
    },
    async emit({ argv, cfg }, _content, _resources): Promise<FilePath[]> {
      const assetsPath = argv.output
      const fps = await filesToCopy(argv, cfg)
      const res: FilePath[] = []

      let succeeded = 0
      let failed = 0
      const failedFiles: string[] = []

      for (const fp of fps) {
        const ext = path.extname(fp)
        const src = joinSegments(argv.directory, fp) as FilePath
        const name = (slugifyFilePath(fp as FilePath, true) + ext) as FilePath
        const dest = joinSegments(assetsPath, name) as FilePath
        // check if file exists, then continue
        if (fs.existsSync(dest)) continue
        if (await copyFileWithRetry(src, dest)) {
          res.push(dest)
          succeeded++
        } else {
          failed++
          failedFiles.push(src)
        }
      }
      // Log summary
      if (argv.verbose) {
        console.log(
          chalk.blue(`[emit:${NAME}] Assets copied: ${succeeded} succeeded, ${failed} failed`),
        )
        if (failed > 0) {
          console.log(chalk.yellow(`\n[emit:${NAME}] Failed files:`))
          failedFiles.forEach((file) => {
            console.log(chalk.gray(`\t- ${file}`))
          })
        }
      }

      return res
    },
  }
}
