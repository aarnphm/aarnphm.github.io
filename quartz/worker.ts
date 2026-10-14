import sourceMapSupport from 'source-map-support'
sourceMapSupport.install(options)

import cfg from '../quartz.config'
import { MarkdownContent, ProcessedContent } from './plugins/vfile'
import {
  cachedProcessors,
  createFileParser,
  createHtmlProcessor,
  createMarkdownParser,
  createMdProcessor,
} from './processors/parse'
import { BuildCtx, WorkerSerializableBuildCtx } from './util/ctx'
import { FilePath } from './util/path'
import { options } from './util/sourcemap'

let workerCtx: BuildCtx | undefined

function resolveWorkerCtx(partialCtx: WorkerSerializableBuildCtx): BuildCtx {
  if (!workerCtx || workerCtx.buildId !== partialCtx.buildId) {
    workerCtx = { ...partialCtx, cfg }
  } else {
    workerCtx.allSlugs = partialCtx.allSlugs
    workerCtx.allFiles = partialCtx.allFiles
    workerCtx.incremental = partialCtx.incremental
  }
  return workerCtx
}

// only called from worker thread
export async function parseFiles(
  partialCtx: WorkerSerializableBuildCtx,
  fps: FilePath[],
): Promise<ProcessedContent[]> {
  const ctx = resolveWorkerCtx(partialCtx)
  const processors = cachedProcessors(ctx)
  const mds = await createFileParser(ctx, fps)(processors.md)
  return createMarkdownParser(ctx, mds)(processors.html)
}

// only called from worker thread
export async function parseMarkdown(
  partialCtx: WorkerSerializableBuildCtx,
  fps: FilePath[],
): Promise<MarkdownContent[]> {
  const ctx = resolveWorkerCtx(partialCtx)
  return await createFileParser(ctx, fps)(createMdProcessor(ctx))
}

// only called from worker thread
export function processHtml(
  partialCtx: WorkerSerializableBuildCtx,
  mds: MarkdownContent[],
): Promise<ProcessedContent[]> {
  const ctx = resolveWorkerCtx(partialCtx)
  return createMarkdownParser(ctx, mds)(createHtmlProcessor(ctx))
}
