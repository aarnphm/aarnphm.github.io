import path from 'path'
import type { ChangeEvent } from '../../../types/plugin'
import type { BuildCtx } from '../../../util/ctx'
import {
  pageListingChanged,
  pageNavigationChanged,
  pageSearchChanged,
  pageSitemapChanged,
} from '../../../util/listing-signature'
import { FilePath, FullSlug, getAllSegmentPrefixes, slugifyFilePath } from '../../../util/path'
import { isSearchExcluded } from '../../../util/search-exclude'
import { pageTitlePatchEvents } from '../../../util/title-patch'
import { ProcessedContent } from '../../vfile'
import {
  addFilePageToIndex,
  addProcessedContentToIndex,
  contentDataBySlug,
  isContentPageFile,
  isGraphFilePage,
  normalizedContentSlug,
  processedContentBySlug,
  type ContentIndexMap,
  type Options,
} from './link-index'
import { serializeContentDetails } from './search-index'

function isSourcePath(path: FilePath): boolean {
  return (
    path.startsWith('quartz/') ||
    path.startsWith('worker/') ||
    path.startsWith('.github/') ||
    path === ('package.json' as FilePath) ||
    path === ('pnpm-lock.yaml' as FilePath) ||
    path === ('tsconfig.json' as FilePath)
  )
}

function changeSlug(changeEvent: ChangeEvent): FullSlug | undefined {
  const slug = changeEvent.file?.data.slug
  if (typeof slug === 'string') {
    return slug as FullSlug
  }
  if (isContentPageFile(changeEvent.path) || isGraphFilePage(changeEvent.path)) {
    return slugifyFilePath(changeEvent.path, path.extname(changeEvent.path) === '.ipynb')
  }
  return undefined
}

function affectedFolderFeedSlugs(changeEvents: readonly ChangeEvent[]): Set<string> {
  const slugs = new Set<string>()
  for (const changeEvent of changeEvents) {
    const slug = changeSlug(changeEvent)
    if (!slug) continue
    const prefixes = getAllSegmentPrefixes(slug)
    prefixes.pop()
    for (const prefix of prefixes) {
      const normalizedPrefix = prefix.replace(/^\/+|\/+$/g, '')
      if (normalizedPrefix.length > 0) {
        slugs.add(normalizedPrefix)
      }
    }
  }

  return slugs
}

function filePageSlug(file: FilePath): FullSlug {
  return slugifyFilePath(file, path.extname(file) === '.ipynb')
}

function isSecurityPolicyChange(changeEvent: ChangeEvent): boolean {
  if (changeEvent.file?.data.slug === 'security-policy') {
    return true
  }
  const normalized = changeEvent.path.replace(/\\/g, '/')
  return normalized === 'security policy.md' || normalized.endsWith('/security policy.md')
}

export function buildSearchIndexPatch(
  ctx: BuildCtx,
  content: ProcessedContent[],
  opts: Options,
  changeEvents: readonly ChangeEvent[],
): { updates: Map<FullSlug, string>; deletes: Set<FullSlug> } | undefined {
  const contentBySlug = contentDataBySlug(content)
  const processedBySlug = processedContentBySlug(content)
  const updates = new Map<FullSlug, string>()
  const deletes = new Set<FullSlug>()

  for (const changeEvent of changeEvents) {
    if (isSourcePath(changeEvent.path)) {
      continue
    }

    if (changeEvent.file) {
      const slug = normalizedContentSlug(changeEvent.file.data)
      if (!slug || slug === 'arena') {
        return undefined
      }
      if (changeEvent.type === 'delete' || isSearchExcluded(slug)) {
        deletes.add(slug)
        continue
      }

      const processed = processedBySlug.get(slug)
      if (!processed) {
        deletes.add(slug)
        continue
      }

      const entryIndex: ContentIndexMap = new Map()
      addProcessedContentToIndex(
        ctx,
        entryIndex,
        processed[0],
        processed[1],
        opts,
        contentBySlug,
        false,
      )
      if (entryIndex.size !== 1) {
        return undefined
      }
      for (const [entrySlug, details] of entryIndex) {
        if (isSearchExcluded(entrySlug) || isSearchExcluded(details.fileName)) {
          deletes.add(entrySlug)
          continue
        }
        updates.set(entrySlug, serializeContentDetails(details, true))
      }
      continue
    }

    if (isGraphFilePage(changeEvent.path)) {
      return undefined
    }
    if (isContentPageFile(changeEvent.path)) {
      continue
    }

    const slug = filePageSlug(changeEvent.path)
    if (
      changeEvent.type === 'delete' ||
      isSearchExcluded(changeEvent.path) ||
      isSearchExcluded(slug)
    ) {
      deletes.add(slug)
      continue
    }

    const fileIndex: ContentIndexMap = new Map()
    addFilePageToIndex(ctx, fileIndex, changeEvent.path)
    const details = fileIndex.get(slug)
    if (details) {
      updates.set(slug, serializeContentDetails(details, true))
    }
  }

  return { updates, deletes }
}

export function titlePatchContentIndexUpdates(
  previous: ContentIndexMap,
  changeEvents: readonly ChangeEvent[],
):
  | {
      linkIndex: ContentIndexMap
      changedSlugs: Set<FullSlug>
      contentUpdates: Map<FullSlug, string>
      searchUpdates: Map<FullSlug, string>
    }
  | undefined {
  const titlePatches = pageTitlePatchEvents(changeEvents)
  if (!titlePatches) return undefined
  const linkIndex = new Map(previous)
  const changedSlugs = new Set<FullSlug>()
  const contentUpdates = new Map<FullSlug, string>()
  const searchUpdates = new Map<FullSlug, string>()

  for (let i = 0; i < changeEvents.length; i += 1) {
    const changeEvent = changeEvents[i]
    const fileData = changeEvent.file?.data
    if (!fileData) return undefined
    const slug = normalizedContentSlug(fileData)
    if (!slug || slug === 'arena') return undefined
    const previousDetails = linkIndex.get(slug)
    if (!previousDetails) return undefined
    const nextDetails = { ...previousDetails, title: titlePatches[i].currentTitle, fileData }
    linkIndex.set(slug, nextDetails)
    changedSlugs.add(slug)
    contentUpdates.set(slug, serializeContentDetails(nextDetails, false))
    searchUpdates.set(slug, serializeContentDetails(nextDetails, true))
  }

  return { linkIndex, changedSlugs, contentUpdates, searchUpdates }
}

type ContentIndexPartialPlan = {
  content: boolean
  graph: boolean
  search: boolean
  security: boolean
  listing: boolean
  sitemap: boolean
  changedSlugs: Set<FullSlug>
  affectedFolders: Set<string>
}

export function planPartialContentIndex(
  changeEvents: readonly ChangeEvent[],
): ContentIndexPartialPlan {
  let content = false
  let graph = false
  let search = false
  let security = false
  let listing = false
  let sitemap = false
  const changedSlugs = new Set<FullSlug>()

  for (const changeEvent of changeEvents) {
    if (isSourcePath(changeEvent.path)) {
      continue
    }

    const file = changeEvent.file
    if (file) {
      search ||= changeEvent.type !== 'change'
      search ||= pageSearchChanged(file.data, changeEvent.previousFile?.data)
      const slug = normalizedContentSlug(file.data)
      if (slug) {
        changedSlugs.add(slug)
      }
      content ||= changeEvent.type !== 'change'
      content ||= pageNavigationChanged(file.data, changeEvent.previousFile?.data)
      listing ||= changeEvent.type !== 'change'
      listing ||= pageListingChanged(file.data, changeEvent.previousFile?.data)
      sitemap ||= changeEvent.type !== 'change'
      sitemap ||= pageSitemapChanged(file.data, changeEvent.previousFile?.data)
      security ||= isSecurityPolicyChange(changeEvent)
      continue
    }

    if (isGraphFilePage(changeEvent.path)) {
      graph ||= changeEvent.type !== 'change'
      search ||= changeEvent.type !== 'change'
      changedSlugs.add(filePageSlug(changeEvent.path))
      continue
    }

    if (!isContentPageFile(changeEvent.path) && changeEvent.type !== 'change') {
      search = true
      changedSlugs.add(filePageSlug(changeEvent.path))
    }
  }

  return {
    content,
    graph,
    search,
    security,
    listing,
    sitemap,
    changedSlugs,
    affectedFolders: affectedFolderFeedSlugs(changeEvents),
  }
}
