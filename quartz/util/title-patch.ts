import type { QuartzPluginData } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import { FullSlug, getAllSegmentPrefixes } from './path'

export type PageTitlePatch = {
  slug: FullSlug
  previousTitle: string
  currentTitle: string
  tags: Set<string>
}

function tagPrefixes(fileData: QuartzPluginData | undefined): string[] {
  return (fileData?.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes)
}

function equivalentTagPrefixes(
  current: QuartzPluginData | undefined,
  previous: QuartzPluginData | undefined,
): boolean {
  const currentTags = tagPrefixes(current)
  const previousTags = tagPrefixes(previous)
  if (currentTags.length !== previousTags.length) return false
  const currentSet = new Set(currentTags)
  return previousTags.every(tag => currentSet.has(tag))
}

function frontmatterWithoutTitle(data: QuartzPluginData | undefined): string {
  const frontmatter = data?.frontmatter
  if (!frontmatter) return ''
  const entries = Object.entries(frontmatter)
    .filter(([key]) => key !== 'title')
    .sort(([left], [right]) => left.localeCompare(right))
  return JSON.stringify(entries)
}

export function pageTitlePatchForEvent(
  changeEvent: ChangeEvent,
  options?: { requireEquivalentTags?: boolean },
): PageTitlePatch | undefined {
  if (changeEvent.type !== 'change') return undefined
  const current = changeEvent.file?.data
  const previous = changeEvent.previousFile?.data
  const slug = current?.slug
  if (!current || !previous || typeof slug !== 'string') return undefined
  if (options?.requireEquivalentTags && !equivalentTagPrefixes(current, previous)) return undefined
  if (frontmatterWithoutTitle(current) !== frontmatterWithoutTitle(previous)) return undefined
  if (JSON.stringify(current.dates ?? null) !== JSON.stringify(previous.dates ?? null))
    return undefined
  if ((current.description ?? '') !== (previous.description ?? '')) return undefined
  const currentTitle = current.frontmatter?.title
  const previousTitle = previous.frontmatter?.title
  if (typeof currentTitle !== 'string' || typeof previousTitle !== 'string') return undefined
  if (currentTitle === previousTitle) return undefined
  return {
    slug: slug as FullSlug,
    previousTitle,
    currentTitle,
    tags: new Set(tagPrefixes(current)),
  }
}

export function pageTitlePatchEvents(
  changeEvents: readonly ChangeEvent[],
  options?: { requireEquivalentTags?: boolean },
): PageTitlePatch[] | undefined {
  const patches: PageTitlePatch[] = []
  for (const changeEvent of changeEvents) {
    const patch = pageTitlePatchForEvent(changeEvent, options)
    if (!patch) return undefined
    patches.push(patch)
  }
  return patches.length > 0 ? patches : undefined
}
