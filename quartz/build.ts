import sourceMapSupport from "source-map-support"
sourceMapSupport.install(options)
import path from "path"
import fs from "node:fs/promises"
import { PerfTimer } from "./util/perf"
import { GlobbyFilterFunction, isGitIgnored } from "globby"
import { parseMarkdown } from "./processors/parse"
import { filterContent } from "./processors/filter"
import { emitContent } from "./processors/emit"
import cfg from "../quartz.config"
import { FilePath, FullSlug, joinSegments, slugifyFilePath } from "./util/path"
import chokidar from "chokidar"
import { HtmlContent } from "./plugins/vfile"
import { Argv, BuildCtx } from "./util/ctx"
import { glob, toPosixPath } from "./util/glob"
import { trace } from "./util/trace"
import { options } from "./util/sourcemap"
import { Mutex } from "async-mutex"
import { styleText } from "node:util"

type TrackedFile = {
  type: "content" | "asset"
  content?: HtmlContent
}

type BuildData = {
  ctx: BuildCtx
  ignored: GlobbyFilterFunction
  mut: Mutex
  initialSlugs: FullSlug[]
  trackedFiles: Map<FilePath, TrackedFile>
  toRebuild: Set<FilePath>
  toRemove: Set<FilePath>
  lastBuildMs: number
}

type FileEvent = "add" | "change" | "delete"

function newBuildId() {
  return Math.random().toString(36).substring(2, 8)
}

async function buildQuartz(argv: Argv, mut: Mutex, clientRefresh: () => void) {
  const ctx: BuildCtx = {
    buildId: newBuildId(),
    argv,
    cfg,
    allSlugs: [],
    allAssets: [],
  }

  const perf = new PerfTimer()
  const output = argv.output

  const pluginCount = Object.values(cfg.plugins).flat().length
  const pluginNames = (key: "transformers" | "filters" | "emitters") =>
    cfg.plugins[key].map((plugin) => plugin.name)
  if (argv.verbose) {
    console.log(`[process] Loaded ${pluginCount} plugins`)
    console.log(`[process] ├── Transformers: [${pluginNames("transformers").join(",")}]`)
    console.log(`[process] ├── Filters: [${pluginNames("filters").join(",")}]`)
    console.log(`[process] └── Emitters: [${pluginNames("emitters").join(",")}]`)
  }

  const release = await mut.acquire()
  perf.addEvent("clean")

  try {
    await fs.access(path.join(output, "/"))
    await fs.rm(path.join(output, "/"), { recursive: true })
    console.log(`[process] Cleaned output directory \`${output}\` in ${perf.timeSince("clean")}`)
  } catch {}

  perf.addEvent("glob")
  const allFiles = await glob("**/*.*", argv.directory, cfg.configuration.ignorePatterns)
  const fps = allFiles.filter((fp) => fp.endsWith(".md")).sort()
  console.log(
    `[process] Found ${fps.length} input files from \`${argv.directory}\` in ${perf.timeSince("glob")}`,
  )

  const filePaths = fps.map((fp) => joinSegments(argv.directory, fp) as FilePath)
  ctx.allSlugs = allFiles.map((fp) => slugifyFilePath(fp as FilePath))

  const parsedFiles = await parseMarkdown(ctx, filePaths)
  const filteredContent = filterContent(ctx, parsedFiles)

  await emitContent(ctx, filteredContent)
  console.log(
    styleText("green", `[build] Done processing ${fps.length} files in ${perf.timeSince()}`),
  )
  release()

  if (argv.serve) {
    return startServing(ctx, mut, parsedFiles, clientRefresh)
  }
}

// setup watcher for rebuilds
async function startServing(
  ctx: BuildCtx,
  mut: Mutex,
  initialContent: HtmlContent[],
  clientRefresh: () => void,
) {
  const { argv } = ctx

  // cache file parse results
  const trackedFiles = new Map<FilePath, TrackedFile>()
  for (const content of initialContent) {
    const [_tree, vfile] = content
    trackedFiles.set(vfile.data.filePath!, { type: "content", content })
  }

  const buildData: BuildData = {
    ctx,
    mut,
    trackedFiles,
    ignored: await isGitIgnored(),
    initialSlugs: ctx.allSlugs,
    toRebuild: new Set<FilePath>(),
    toRemove: new Set<FilePath>(),
    lastBuildMs: 0,
  }

  const watcher = chokidar.watch(".", {
    persistent: true,
    cwd: argv.directory,
    ignoreInitial: true,
  })

  watcher
    .on("add", (fp) => rebuildFromEntrypoint(fp, "add", clientRefresh, buildData))
    .on("change", (fp) => rebuildFromEntrypoint(fp, "change", clientRefresh, buildData))
    .on("unlink", (fp) => rebuildFromEntrypoint(fp, "delete", clientRefresh, buildData))

  return async () => {
    await watcher.close()
  }
}

async function rebuildFromEntrypoint(
  fp: string,
  action: FileEvent,
  clientRefresh: () => void,
  buildData: BuildData, // note: this function mutates buildData
) {
  const { ctx, ignored, mut, initialSlugs, trackedFiles, toRebuild, toRemove } = buildData

  const { argv } = ctx

  // don't do anything for gitignored files
  if (ignored(fp)) {
    return
  }

  // dont bother rebuilding for non-content files, just track and refresh
  fp = toPosixPath(fp)
  const filePath = joinSegments(argv.directory, fp) as FilePath
  if (path.extname(fp) !== ".md") {
    if (action === "add" || action === "change") {
      trackedFiles.set(filePath, { type: "asset" })
    } else if (action === "delete") {
      trackedFiles.delete(filePath)
    }
    clientRefresh()
    return
  }

  if (action === "add" || action === "change") {
    toRebuild.add(filePath)
  } else if (action === "delete") {
    toRemove.add(filePath)
  }
  ctx.allAssets = [...trackedFiles.keys()].filter((k) => path.extname(k) !== ".md")

  const buildId = newBuildId()
  ctx.buildId = buildId
  buildData.lastBuildMs = new Date().getTime()
  const release = await mut.acquire()

  // there's another build after us, release and let them do it
  if (ctx.buildId !== buildId) {
    release()
    return
  }

  const perf = new PerfTimer()
  console.log(styleText("yellow", "Detected change, rebuilding..."))

  try {
    const filesToRebuild = [...toRebuild].filter((fp) => !toRemove.has(fp))
    const parsedContent = await parseMarkdown(ctx, filesToRebuild)
    for (const [tree, vfile] of parsedContent) {
      trackedFiles.set(vfile.data.filePath!, { type: "content", content: [tree, vfile] })
    }

    for (const fp of toRemove) {
      trackedFiles.delete(fp)
    }

    const parsedFiles = [...trackedFiles.values()]
      .filter((file) => file.type === "content" && file.content)
      .map((file) => file.content!)
    const filteredContent = filterContent(ctx, parsedFiles)

    // re-update slugs
    const trackedSlugs = [...new Set([...trackedFiles.keys(), ...toRebuild])]
      .filter((fp) => !toRemove.has(fp))
      .map((fp) => slugifyFilePath(path.posix.relative(argv.directory, fp) as FilePath))

    ctx.allSlugs = [...new Set([...initialSlugs, ...trackedSlugs])]

    // Determine which files need to be rebuilt
    const filesToDelete = new Set<string>()

    // Add changed/removed files to deletion set
    for (const file of [...new Set([...toRebuild, ...toRemove])]) {
      const relativePath = path.relative(argv.directory, file)
      const outputPath = path.join(argv.output, relativePath)
      filesToDelete.add(outputPath.replace(/\.md$/, ".html"))
    }

    // Delete only the necessary files
    for (const file of filesToDelete) {
      try {
        await fs.rm(file, { force: true })
        if (argv.verbose) {
          console.log(styleText("yellow", `[build] Deleted ${file}`))
        }
      } catch {}
    }

    // Filter content to only include changed/rebuilt files
    const changedFiles = new Set([...toRebuild].map((fp) => path.relative(argv.directory, fp)))
    const contentToEmit = filteredContent.filter(([_tree, vfile]) => {
      const relativePath = path.relative(argv.directory, vfile.data.filePath!)
      return changedFiles.has(relativePath) || toRemove.has(vfile.data.filePath!)
    })

    if (argv.verbose) {
      console.log(styleText("yellow", `[rebuild] Emitting ${contentToEmit.length} changed files`))
    }
    await emitContent(ctx, contentToEmit, true)

    console.log(styleText("green", `[rebuild] Done rebuilding in ${perf.timeSince()}`))
  } catch (err: any) {
    console.log(
      styleText("yellow", `[rebuild] Rebuild failed. Waiting on a change to fix the error...`),
    )
    if (argv.verbose) {
      console.log(styleText("red", err.toString()))
    }
  }

  clientRefresh()
  toRebuild.clear()
  toRemove.clear()
  release()
}

export default async (argv: Argv, mut: Mutex, clientRefresh: () => void) => {
  try {
    return await buildQuartz(argv, mut, clientRefresh)
  } catch (err) {
    trace("\nExiting Quartz due to a fatal error", err as Error)
  }
}
