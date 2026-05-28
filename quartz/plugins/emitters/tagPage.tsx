import { readFile, rm } from 'node:fs/promises'
import { defaultListPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout, GlobalConfiguration } from '../../cfg'
import { TagContent } from '../../components'
import HeaderConstructor from '../../components/Header'
import { byDateAndAlphabetical, type SortFn } from '../../components/PageList'
import { pageResources, renderPage } from '../../components/renderPage'
import { i18n } from '../../i18n'
import { QuartzComponentProps } from '../../types/component'
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { escapeHTML } from '../../util/escape'
import { pageListingChanged } from '../../util/listing-signature'
import {
  FilePath,
  FullSlug,
  getAllSegmentPrefixes,
  joinSegments,
  pathToRoot,
  resolveRelative,
} from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { StaticResources } from '../../util/resources'
import { PageTitlePatch, pageTitlePatchEvents } from '../../util/title-patch'
import { ProcessedContent, QuartzPluginData, defaultProcessedContent } from '../vfile'
import { write, writeKnownChanged } from './helpers'

interface TagPageOptions extends FullPageLayout {
  sort?: SortFn
}

type TagIndexData = {
  tagItemMap: Map<string, QuartzPluginData[]>
  tagContentMap: Map<string, QuartzPluginData>
}

type TagInfo = {
  tags: Set<string>
  tagDescriptions: Record<string, ProcessedContent>
  tagItemMap: Map<string, QuartzPluginData[]>
  tagContentMap: Map<string, QuartzPluginData>
}

type HtmlPatchResult = { html: string; patched: boolean }

function defaultTagContent(tag: string, cfg: GlobalConfiguration): ProcessedContent {
  const title =
    tag === 'index'
      ? i18n(cfg.locale).pages.tagContent.tagIndex
      : `${i18n(cfg.locale).pages.tagContent.tag}: ${tag}`
  return defaultProcessedContent({
    slug: joinSegments('tags', tag) as FullSlug,
    frontmatter: { title, tags: [], pageLayout: 'default', description: `generated tags/${tag}` },
  })
}

function tagPrefixes(fileData: QuartzPluginData | undefined): string[] {
  return (fileData?.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes)
}

function tagPageTag(fileData: QuartzPluginData | undefined): string | undefined {
  const slug = fileData?.slug
  return slug?.startsWith('tags/') ? slug.slice('tags/'.length) : undefined
}

function tagItemSlugs(info: TagInfo, tag: string): string[] {
  return (info.tagItemMap.get(tag) ?? []).map(file => file.slug ?? '')
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) return false
  for (let i = 0; i < left.length; i += 1) {
    if (left[i] !== right[i]) return false
  }
  return true
}

function patchPageListTitleHtml(
  html: string,
  href: string,
  previousTitle: string,
  currentTitle: string,
): HtmlPatchResult | undefined {
  const escapedHref = escapeHTML(href)
  const escapedPreviousTitle = escapeHTML(previousTitle)
  const escapedCurrentTitle = escapeHTML(currentTitle)
  let next = ''
  let cursor = 0
  let patched = false

  while (true) {
    const hrefIndex = html.indexOf(`href="${escapedHref}"`, cursor)
    if (hrefIndex === -1) {
      next += html.slice(cursor)
      break
    }

    const liStart = html.lastIndexOf('<li', hrefIndex)
    const liEnd = html.indexOf('</li>', hrefIndex)
    if (liStart === -1 || liEnd === -1) return undefined

    const blockEnd = liEnd + '</li>'.length
    const before = html.slice(cursor, liStart)
    const block = html.slice(liStart, blockEnd)
    const withTitleAttr = block.replace(
      `data-title="${escapedPreviousTitle}"`,
      `data-title="${escapedCurrentTitle}"`,
    )
    const patchedBlock = withTitleAttr.replace(
      `<div class="desc">${escapedPreviousTitle}`,
      `<div class="desc">${escapedCurrentTitle}`,
    )
    if (patchedBlock === block) return undefined
    next += before + patchedBlock
    cursor = blockEnd
    patched = true
  }

  return { html: next, patched }
}

function patchTagPageTitles(
  html: string,
  tagSlug: FullSlug,
  tag: string,
  patches: readonly PageTitlePatch[],
): string | undefined {
  let next = html
  for (const patch of patches) {
    const shouldPatch = tag === 'index' || patch.tags.has(tag)
    if (!shouldPatch) continue
    const result = patchPageListTitleHtml(
      next,
      resolveRelative(tagSlug, patch.slug),
      patch.previousTitle,
      patch.currentTitle,
    )
    if (!result || (tag !== 'index' && !result.patched)) return undefined
    next = result.html
  }
  return next
}

function removeTaggedFile(tagItemMap: Map<string, QuartzPluginData[]>, fileData: QuartzPluginData) {
  const slug = fileData.slug
  if (!slug) return
  for (const tag of tagPrefixes(fileData)) {
    const pages = tagItemMap.get(tag)
    if (!pages) continue
    const kept = pages.filter(page => page.slug !== slug)
    if (kept.length > 0) {
      tagItemMap.set(tag, kept)
    } else {
      tagItemMap.delete(tag)
    }
  }
}

function addTaggedFile(tagItemMap: Map<string, QuartzPluginData[]>, fileData: QuartzPluginData) {
  if (!fileData.slug) return
  for (const tag of tagPrefixes(fileData)) {
    const pages = tagItemMap.get(tag)
    if (pages) {
      pages.push(fileData)
    } else {
      tagItemMap.set(tag, [fileData])
    }
  }
}

function processedBySlug(content: ProcessedContent[]): Map<string, ProcessedContent> {
  const processed = new Map<string, ProcessedContent>()
  for (const entry of content) {
    const slug = entry[1].data.slug
    if (slug) processed.set(slug, entry)
  }
  return processed
}

async function deleteTagPage(ctx: BuildCtx, tag: string): Promise<void> {
  await rm(joinSegments(ctx.argv.output, 'tags', `${tag}.html`) as FilePath, { force: true })
}

function computeTagInfo(
  allFiles: QuartzPluginData[],
  content: ProcessedContent[],
  cfg: GlobalConfiguration,
  sort: SortFn | undefined,
): TagInfo {
  const tagItemMap = new Map<string, QuartzPluginData[]>()
  const tagContentMap = new Map<string, QuartzPluginData>()
  for (const data of allFiles) {
    for (const tag of (data.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes)) {
      const pages = tagItemMap.get(tag)
      if (pages) {
        pages.push(data)
      } else {
        tagItemMap.set(tag, [data])
      }
    }
  }
  const sorter = sort ?? byDateAndAlphabetical(cfg)
  for (const [tag, pages] of tagItemMap) {
    tagItemMap.set(tag, [...pages].sort(sorter))
  }
  const tags = new Set(tagItemMap.keys())

  tags.add('index')

  const tagDescriptions: Record<string, ProcessedContent> = Object.fromEntries(
    [...tags].map(tag => [tag, defaultTagContent(tag, cfg)]),
  )

  for (const [tree, file] of content) {
    const slug = file.data.slug!
    if (slug.startsWith('tags/')) {
      const tag = slug.slice('tags/'.length)
      if (tags.has(tag)) {
        tagDescriptions[tag] = [tree, file]
        tagContentMap.set(tag, file.data)
        if (file.data.frontmatter?.title === tag) {
          file.data.frontmatter.title = `${i18n(cfg.locale).pages.tagContent.tag}: ${tag}`
        }
      }
    }
  }

  return { tags, tagDescriptions, tagItemMap, tagContentMap }
}

function isTagRelevantChange(changeEvent: ChangeEvent): boolean {
  const current = changeEvent.file?.data
  const previous = changeEvent.previousFile?.data
  if (tagPageTag(current) || tagPageTag(previous)) return true
  if (changeEvent.type !== 'change') return true
  return pageListingChanged(current, previous)
}

function affectedTagsForEvents(changeEvents: readonly ChangeEvent[]): Set<string> {
  const affectedTags = new Set<string>()
  for (const changeEvent of changeEvents) {
    const current = changeEvent.file?.data
    const previous = changeEvent.previousFile?.data
    tagPrefixes(previous).forEach(tag => affectedTags.add(tag))
    tagPrefixes(current).forEach(tag => affectedTags.add(tag))
    const currentTagPage = tagPageTag(current)
    const previousTagPage = tagPageTag(previous)
    if (currentTagPage) affectedTags.add(currentTagPage)
    if (previousTagPage) affectedTags.add(previousTagPage)
  }
  if (affectedTags.size > 0) affectedTags.add('index')
  return affectedTags
}

function updateTagInfo(
  info: TagInfo,
  content: ProcessedContent[],
  cfg: GlobalConfiguration,
  sort: SortFn | undefined,
  changeEvents: readonly ChangeEvent[],
): Set<string> {
  const affectedTags = new Set<string>()
  const processed = processedBySlug(content)

  for (const changeEvent of changeEvents) {
    const current = changeEvent.file?.data
    const previous = changeEvent.previousFile?.data
    const currentTagPage = tagPageTag(current)
    const previousTagPage = tagPageTag(previous)

    if (previous) {
      tagPrefixes(previous).forEach(tag => affectedTags.add(tag))
      removeTaggedFile(info.tagItemMap, previous)
    }

    if (current && changeEvent.type !== 'delete') {
      tagPrefixes(current).forEach(tag => affectedTags.add(tag))
      addTaggedFile(info.tagItemMap, current)
    }

    if (previousTagPage) {
      affectedTags.add(previousTagPage)
      info.tagContentMap.delete(previousTagPage)
    }

    if (currentTagPage && current && changeEvent.type !== 'delete') {
      affectedTags.add(currentTagPage)
      info.tagContentMap.set(currentTagPage, current)
    }
  }

  const sorter = sort ?? byDateAndAlphabetical(cfg)
  for (const [tag, pages] of info.tagItemMap) {
    if (!affectedTags.has(tag)) continue
    info.tagItemMap.set(tag, [...pages].sort(sorter))
  }

  info.tags = new Set(info.tagItemMap.keys())
  info.tags.add('index')
  affectedTags.add('index')

  for (const tag of affectedTags) {
    if (!info.tags.has(tag)) {
      delete info.tagDescriptions[tag]
      continue
    }
    const contentPage = processed.get(joinSegments('tags', tag))
    if (contentPage) {
      info.tagDescriptions[tag] = contentPage
      info.tagContentMap.set(tag, contentPage[1].data)
      if (contentPage[1].data.frontmatter?.title === tag) {
        contentPage[1].data.frontmatter.title = `${i18n(cfg.locale).pages.tagContent.tag}: ${tag}`
      }
    } else {
      info.tagDescriptions[tag] = defaultTagContent(tag, cfg)
    }
  }

  return affectedTags
}

async function processTagPage(
  ctx: BuildCtx,
  tag: string,
  tagContent: ProcessedContent,
  allFiles: QuartzPluginData[],
  tagPageFiles: QuartzPluginData[] | undefined,
  tagIndexData: TagIndexData | undefined,
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = joinSegments('tags', tag) as FullSlug
  const [tree, file] = tagContent
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: file.data,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
    tagPageFiles,
    tagIndexData,
  }

  const content = renderPage(ctx, slug, componentData, opts, externalResources, true)
  return write({ ctx, content, slug: file.data.slug!, ext: '.html' })
}

export const TagPage: QuartzEmitterPlugin<Partial<TagPageOptions>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    pageBody: TagContent({ sort: userOpts?.sort }),
    header: [...defaultListPageLayout.beforeBody],
    beforeBody: [],
    sidebar: [],
    afterBody: [],
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()
  let cachedTagInfo: TagInfo | undefined

  return {
    name: 'TagPage',
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)
      const cfg = ctx.cfg.configuration
      cachedTagInfo = computeTagInfo(allFiles, content, cfg, userOpts?.sort)
      const { tags, tagDescriptions, tagItemMap, tagContentMap } = cachedTagInfo
      const tagIndexData = { tagItemMap, tagContentMap }

      const files = await mapConcurrent([...tags], defaultIoConcurrency, tag => {
        const tagPageFiles = tag === 'index' ? undefined : (tagItemMap.get(tag) ?? [])
        return processTagPage(
          ctx,
          tag,
          tagDescriptions[tag],
          allFiles,
          tagPageFiles,
          tag === 'index' ? tagIndexData : undefined,
          opts,
          resources,
        )
      })

      yield* files
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const relevantEvents = changeEvents.filter(isTagRelevantChange)
      if (relevantEvents.length === 0) return
      const canTryTitlePatch = !relevantEvents.some(changeEvent =>
        tagPageTag(changeEvent.file?.data),
      )

      const allFiles = contentDataFor(content)
      const cfg = ctx.cfg.configuration

      let affectedTags: Set<string>
      if (cachedTagInfo) {
        const previousTagOrder = new Map<string, string[]>()
        const titlePatches = canTryTitlePatch
          ? pageTitlePatchEvents(relevantEvents, { requireEquivalentTags: true })
          : undefined
        if (titlePatches) {
          for (const tag of affectedTagsForEvents(relevantEvents)) {
            previousTagOrder.set(tag, tagItemSlugs(cachedTagInfo, tag))
          }
        }
        affectedTags = updateTagInfo(cachedTagInfo, content, cfg, userOpts?.sort, relevantEvents)
        if (titlePatches) {
          const canPatchTitles = [...affectedTags].every(tag =>
            sameStringArray(previousTagOrder.get(tag) ?? [], tagItemSlugs(cachedTagInfo!, tag)),
          )
          if (canPatchTitles) {
            const perf = new PerfTimer()
            const files = await mapConcurrent(
              [...affectedTags],
              defaultIoConcurrency,
              async tag => {
                if (!cachedTagInfo!.tags.has(tag)) return undefined
                const slug = joinSegments('tags', tag) as FullSlug
                const pathToPage = joinSegments(ctx.argv.output, `${slug}.html`) as FilePath
                const currentHtml = await readFile(pathToPage, 'utf8')
                const patchedHtml = patchTagPageTitles(currentHtml, slug, tag, titlePatches)
                if (patchedHtml === undefined || patchedHtml === currentHtml) return undefined
                return writeKnownChanged({ ctx, content: patchedHtml, slug, ext: '.html' })
              },
            )

            logBuildSpan(ctx.argv, 'tagPage:titlePatch', `${files.length} tags`, perf.elapsedMs())
            yield* files.filter(file => file !== undefined)
            return
          }
        }
      } else {
        cachedTagInfo = computeTagInfo(allFiles, content, cfg, userOpts?.sort)
        affectedTags = affectedTagsForEvents(relevantEvents)
      }

      const { tagDescriptions, tagItemMap, tagContentMap, tags } = cachedTagInfo
      const tagIndexData = { tagItemMap, tagContentMap }

      const files = await mapConcurrent([...affectedTags], defaultIoConcurrency, async tag => {
        if (!tags.has(tag)) {
          await deleteTagPage(ctx, tag)
          return undefined
        }

        const tagDescription = tagDescriptions[tag]
        if (!tagDescription) return undefined

        const tagPageFiles = tag === 'index' ? undefined : (tagItemMap.get(tag) ?? [])
        return processTagPage(
          ctx,
          tag,
          tagDescription,
          allFiles,
          tagPageFiles,
          tag === 'index' ? tagIndexData : undefined,
          opts,
          resources,
        )
      })

      yield* files.filter(file => file !== undefined)
    },
  }
}
