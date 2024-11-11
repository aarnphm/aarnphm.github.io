import { QuartzEmitterPlugin } from "../types"
import path from "path"
import fs from "fs"
import sharp from "sharp"
import { FilePath, joinSegments, slugifyFilePath } from "../../util/path"
import { glob } from "../../util/glob"
import DepGraph from "../../depgraph"
import { Argv } from "../../util/ctx"
import chalk from "chalk"
import { BuildCtx } from "../../util/ctx"
import { QuartzConfig } from "../../cfg"

interface Options {
  /**
   * Quality of WebP output (0-100)
   * @default 80
   */
  quality?: number
  /**
   * Extensions to transform
   * @default ['.jpg', '.jpeg']
   */
  extensions?: string[]
}

const defaultOptions: Options = {
  quality: 80,
  extensions: [".jpg", ".jpeg"],
}

interface ConversionTask {
  inputPath: FilePath
  outputPath: FilePath
}

const NAME = "Image"

async function processBatch(
  ctx: BuildCtx,
  tasks: ConversionTask[],
  opts: Options,
  onProgress?: (completed: number, total: number) => void,
): Promise<FilePath[]> {
  const fps: FilePath[] = []
  let completed = 0
  const total = tasks.length

  // Process concurrently in batches
  const batchSize = ctx.argv.concurrency ?? 4
  for (let i = 0; i < tasks.length; i += batchSize) {
    const batch = tasks.slice(i, i + batchSize)
    const batchPromises = batch.map(async ({ inputPath, outputPath }) => {
      try {
        // Ensure output directory exists
        const outputDir = path.dirname(outputPath)
        if (!fs.existsSync(outputDir)) {
          await fs.promises.mkdir(outputDir, { recursive: true })
        }

        // Convert image
        await sharp(inputPath).webp({ quality: opts.quality }).toFile(outputPath)

        fps.push(outputPath)
        completed++
        onProgress?.(completed, total)
      } catch (error) {
        console.error(chalk.red(`[${NAME}] Error processing ${inputPath}:`), error)
      }
    })

    // Wait for current batch to complete before starting next batch
    await Promise.all(batchPromises)
  }

  return fps
}

const filesToCopy = async (argv: Argv, opts: Options, cfg: QuartzConfig) => {
  return await glob(
    `**/*{${opts.extensions?.join(",")}}`,
    argv.directory,
    cfg.configuration.ignorePatterns,
  )
}

export const Image: QuartzEmitterPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: NAME,
    async getDependencyGraph({ argv, cfg }, _content, _resources) {
      const graph = new DepGraph<FilePath>()
      const imageFiles = await filesToCopy(argv, opts, cfg)
      // Find all image files in output directory matching our extensions
      for (const fp of imageFiles) {
        const src = joinSegments(argv.directory, fp) as FilePath
        const name = (slugifyFilePath(fp as FilePath, true) + ".webp") as FilePath
        const dest = joinSegments(argv.output, name) as FilePath
        graph.addEdge(src, dest)
      }

      return graph
    },

    async emit(ctx, _content, _resources) {
      const { argv, cfg } = ctx

      // Find all image files in output directory matching our extensions
      const fps = await filesToCopy(argv, opts, cfg)
      if (fps.length === 0) return []

      // Create task list
      const tasks: ConversionTask[] = fps.map((file) => ({
        inputPath: joinSegments(argv.directory, file) as FilePath,
        outputPath: joinSegments(
          argv.output,
          (slugifyFilePath(file as FilePath, true) + ".webp") as FilePath,
        ) as FilePath,
      }))

      // Progress tracking
      let progressBar = ""
      const total = tasks.length
      const updateProgress = (completed: number, total: number) => {
        const percent = Math.round((completed / total) * 100)
        progressBar = `[emit:${NAME}] Converting images: ${completed}/${total} (${percent}%)`
        process.stdout.write(`\r${progressBar}`)
      }

      console.log(chalk.blue(`[emit:${NAME}] Converting ${total} images to webp format...`))
      const res = await processBatch(ctx, tasks, opts, updateProgress)

      if (progressBar) {
        process.stdout.write("\n") // New line after progress bar
      }
      console.log(chalk.green(`[emit:${NAME}] Successfully converted ${fps.length} images`))

      return res
    },
    getQuartzComponents() {
      return []
    },
  }
}
