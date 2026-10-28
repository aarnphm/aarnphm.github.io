import path from 'path'
import { ProcessedContent } from '../plugins/vfile'
import { ChangeEvent } from '../types/plugin'
import { courseDirOf } from '../util/course'
import { Argv, BuildCtx } from '../util/ctx'
import { flashcardsSlug, isFlashcardPath } from '../util/flashcards-path'
import { toPosixPath } from '../util/glob'
import { FilePath, joinSegments, slugifyFilePath } from '../util/path'

const markdownExtensions = new Set(['.md', '.base', '.canvas'])
const testPathPattern =
  /(?:^|\/)(?:__(?:tests|specs)__|tests?|specs?)(?:\/|$)|(?:^|\/)(?:tests?|specs?|(?:test|spec)_[^/]+|[^/]+_(?:test|spec)|[^/]+\.(?:test|spec))\.[^/]+$/i

export const isMarkdownPath = (fp: string): boolean =>
  markdownExtensions.has(path.extname(fp)) || isFlashcardPath(fp)

export const isTestPath = (fp: string): boolean => testPathPattern.test(toPosixPath(fp))

let derivedPaths: { source: FilePath[]; markdown: FilePath[]; files: FilePath[] } | undefined

export const derivePaths = (argv: Argv, allFiles: FilePath[]) => {
  if (derivedPaths?.source === allFiles) return derivedPaths
  const markdown = allFiles.filter(isMarkdownPath).sort()
  const files = markdown.map(fp => joinSegments(argv.directory, fp) as FilePath)
  derivedPaths = { source: allFiles, markdown, files }
  return derivedPaths
}

export const syncCtxFiles = (ctx: BuildCtx, allFiles: FilePath[]) => {
  ctx.allFiles = allFiles
  ctx.allSlugs = allFiles.filter(fp => !isFlashcardPath(fp)).map(fp => slugifyFilePath(fp))
  delete ctx.trie
  delete ctx.renderData
}

export type PendingChanges = Map<FilePath, ChangeEvent['type']>

const contentFilePath = (content: ProcessedContent): FilePath | undefined =>
  content[1].data.filePath as FilePath | undefined

export function indexContent(content: ProcessedContent[]): Map<FilePath, ProcessedContent> {
  const map = new Map<FilePath, ProcessedContent>()
  for (const entry of content) {
    const fp = contentFilePath(entry)
    if (fp) map.set(fp, entry)
  }
  return map
}

export function buildChangeEvents(
  ctx: BuildCtx,
  pending: PendingChanges,
  previous: ReadonlyMap<FilePath, ProcessedContent>,
  current: ReadonlyMap<FilePath, ProcessedContent>,
): ChangeEvent[] {
  const events: ChangeEvent[] = []
  for (const [relPath, fsType] of pending) {
    const fullPath = joinSegments(ctx.argv.directory, relPath) as FilePath
    const prev = previous.get(fullPath)
    const cur = fsType === 'delete' ? undefined : current.get(fullPath)
    if (prev && !cur) {
      events.push({ type: 'delete', path: relPath, file: prev[1], previousFile: prev[1] })
    } else if (!prev && cur) {
      events.push({ type: 'add', path: relPath, file: cur[1] })
    } else if (prev && cur) {
      events.push({ type: 'change', path: relPath, file: cur[1], previousFile: prev[1] })
    } else {
      events.push({ type: fsType, path: relPath })
    }
  }
  for (const [fullPath, content] of current) {
    const relPath = path.relative(ctx.argv.directory, fullPath) as FilePath
    if (pending.has(relPath) || previous.get(fullPath) === content) continue
    const prev = previous.get(fullPath)
    events.push({
      type: prev ? 'change' : 'add',
      path: relPath,
      file: content[1],
      previousFile: prev?.[1],
    })
  }
  return events
}

/**
 * Output slugs a deleted course page leaves in `public`: its page (ContentPage, FolderPage or PdfReader
 * writes it at the page slug), and the deck page of a `.fc`.
 */
export function deletedCourseOutputSlugs(event: ChangeEvent): string[] | undefined {
  const data = event.file?.data
  if (!data?.slug || courseDirOf(data.slug) === null) return undefined
  const sourceSlug = data.flashcards?.sourceSlug
  return sourceSlug ? [data.slug, flashcardsSlug(sourceSlug)] : [data.slug]
}

export function describeBuildError(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  try {
    return JSON.stringify(err)
  } catch {
    return String(err)
  }
}
