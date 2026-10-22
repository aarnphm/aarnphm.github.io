import { QuartzPluginData } from '../../../../plugins/vfile'
import { ReloadableWeakMap } from '../../../module-caches'
import { FilePath, FullSlug, simplifySlug, slugifyFilePath, splitAnchor } from '../../../path'
import { isRecord } from '../../../type-guards'
import { parseWikilink, resolveWikilinkTarget } from '../../../wikilinks'
import { toValue, valueToString } from './value-conversion'
import {
  isDateValue,
  isFileValue,
  isLinkValue,
  isListValue,
  isObjectValue,
  isStringValue,
  isValue,
  makeDate,
  makeFile,
  makeLink,
  makeList,
  makeNull,
  makeNumber,
  makeObject,
  makeString,
  type EvalContext,
  type ObjectValue,
  type Value,
} from './value-model'

type FrontmatterValueCache = { full?: ObjectValue; props: Map<string, Value> }

const frontmatterValueCache = new ReloadableWeakMap<
  Record<string, unknown>,
  FrontmatterValueCache
>()

const getFrontmatterCacheEntry = (frontmatter: Record<string, unknown>): FrontmatterValueCache => {
  let entry = frontmatterValueCache.get(frontmatter)
  if (!entry) {
    entry = { props: new Map() }
    frontmatterValueCache.set(frontmatter, entry)
  }
  return entry
}

export const getFrontmatterPropValue = (
  frontmatter: Record<string, unknown> | undefined,
  prop: string,
): Value => {
  if (!frontmatter || !isRecord(frontmatter)) return makeNull()
  const entry = getFrontmatterCacheEntry(frontmatter)
  const cached = entry.props.get(prop)
  if (cached !== undefined) return cached
  const converted = toValue(frontmatter[prop])
  entry.props.set(prop, converted)
  return converted
}

const getFrontmatterObjectValue = (frontmatter: Record<string, unknown> | undefined): Value => {
  if (!frontmatter || !isRecord(frontmatter)) return makeNull()
  const entry = getFrontmatterCacheEntry(frontmatter)
  if (entry.full) return entry.full
  const obj: Record<string, Value> = {}
  for (const key of Object.keys(frontmatter)) {
    obj[key] = getFrontmatterPropValue(frontmatter, key)
  }
  const full = makeObject(obj)
  entry.full = full
  return full
}

export const resolveIdentifier = (name: string, ctx: EvalContext): Value => {
  if (name === 'this') return ctx.thisFile ? makeFile(ctx.thisFile) : makeNull()
  if (ctx.locals && name in ctx.locals) {
    const local = ctx.locals[name]
    if (isValue(local)) return local
  }
  if (name === 'file') return makeFile(ctx.file)
  if (name === 'note') {
    return getFrontmatterObjectValue(ctx.file.frontmatter as Record<string, unknown> | undefined)
  }
  if (name === 'values' && ctx.values) {
    return makeList(ctx.values)
  }
  if (name === 'rows' && ctx.rows) {
    return makeList(ctx.rows.map(row => makeFile(row)))
  }
  if (name === 'formula') {
    return makeObject({})
  }
  return getFrontmatterPropValue(ctx.file.frontmatter as Record<string, unknown> | undefined, name)
}

// Obsidian resolves file.path, folder, name and inFolder() against the vault path. A slug can
// differ from it: spaces become dashes and a flashcard deck `a/b.fc` publishes at a/b/flashcards.
export const vaultPath = (file: QuartzPluginData): string => {
  const rel = file.relativePath
  if (typeof rel === 'string' && rel.length > 0) {
    return rel.startsWith('content/') ? rel.slice(8) : rel
  }
  const fp = file.filePath
  if (typeof fp === 'string' && fp.length > 0) {
    return fp.startsWith('content/') ? fp.slice(8) : fp
  }
  return file.slug ? `${String(file.slug)}.md` : ''
}

const backlinksIndexCache = new ReloadableWeakMap<QuartzPluginData[], Map<string, string[]>>()

const getBacklinksIndex = (allFiles: QuartzPluginData[]): Map<string, string[]> => {
  let index = backlinksIndexCache.get(allFiles)
  if (!index) {
    const backlinkSets = new Map<string, Set<string>>()
    for (const entry of allFiles) {
      if (!entry.slug) continue
      const sourceSlug = simplifySlug(entry.slug as FullSlug)
      const links = Array.isArray(entry.links) ? entry.links : []
      for (const rawLink of links) {
        const link = String(rawLink)
        if (!link) continue
        let set = backlinkSets.get(link)
        if (!set) {
          set = new Set()
          backlinkSets.set(link, set)
        }
        set.add(sourceSlug)
      }
    }
    index = new Map<string, string[]>()
    for (const [key, set] of backlinkSets) {
      index.set(key, [...set])
    }
    backlinksIndexCache.set(allFiles, index)
  }
  return index
}

export const resolveFileProperty = (
  file: QuartzPluginData,
  property: string,
  ctx: EvalContext,
): Value => {
  if (property === 'file') return makeFile(file)
  if (property === 'name' || property === 'basename') {
    const vp = vaultPath(file)
    const slash = vp.lastIndexOf('/')
    const segment = slash >= 0 ? vp.slice(slash + 1) : vp
    if (property === 'name') return makeString(segment)
    const dot = segment.lastIndexOf('.')
    return makeString(dot > 0 ? segment.slice(0, dot) : dot === 0 ? '' : segment)
  }
  if (property === 'title') {
    const title = typeof file.frontmatter?.title === 'string' ? file.frontmatter.title : ''
    if (title.length > 0) return makeString(title)
    return resolveFileProperty(file, 'basename', ctx)
  }
  if (property === 'path') return makeString(vaultPath(file))
  if (property === 'folder') {
    const vp = vaultPath(file)
    const slash = vp.lastIndexOf('/')
    return makeString(slash >= 0 ? vp.slice(0, slash) : '')
  }
  if (property === 'ext') {
    const vp = vaultPath(file)
    const slash = vp.lastIndexOf('/')
    const dot = vp.lastIndexOf('.')
    return makeString(dot > slash ? vp.slice(dot + 1) : 'md')
  }
  if (property === 'size') {
    if (isRecord(file) && typeof file.size === 'number') {
      return makeNumber(file.size)
    }
    return makeNull()
  }
  if (property === 'ctime') {
    const created = file.dates?.created
    if (!created) return makeNull()
    return makeDate(new Date(created))
  }
  if (property === 'mtime') {
    const modified = file.dates?.modified
    if (!modified) return makeNull()
    return makeDate(new Date(modified))
  }
  if (property === 'tags') {
    const rawTags = file.frontmatter?.tags
    const tags = Array.isArray(rawTags) ? rawTags : typeof rawTags === 'string' ? [rawTags] : []
    return makeList(tags.map(tag => makeString(String(tag))))
  }
  if (property === 'aliases') {
    const aliases = file.frontmatter?.aliases
    if (!aliases) return makeList([])
    const list = Array.isArray(aliases) ? aliases : [aliases]
    return makeList(list.map(alias => makeString(String(alias))))
  }
  if (property === 'links' || property === 'outlinks') {
    const links = Array.isArray(file.links) ? file.links : []
    return makeList(links.map(link => makeLink(String(link))))
  }
  if (property === 'backlinks' || property === 'inlinks') {
    const slug = file.slug ? String(file.slug) : ''
    const key = simplifySlug(slug as FullSlug)
    const index = ctx.backlinksIndex ?? getBacklinksIndex(ctx.allFiles)
    const backlinks = index.get(key) ?? []
    return makeList(backlinks.map(link => makeLink(String(link))))
  }
  if (property === 'embeds') {
    const embeds = isRecord(file) && Array.isArray(file.embeds) ? file.embeds : []
    return makeList(embeds.map(entry => makeString(String(entry))))
  }
  if (property === 'properties') {
    return getFrontmatterObjectValue(file.frontmatter as Record<string, unknown> | undefined)
  }
  if (property === 'link') {
    const slug = file.slug ? String(file.slug) : ''
    return makeLink(slug)
  }
  return getFrontmatterPropValue(file.frontmatter as Record<string, unknown> | undefined, property)
}

export const accessProperty = (value: Value, property: string, ctx: EvalContext): Value => {
  if (isStringValue(value) && property === 'length') return makeNumber(value.value.length)
  if (isListValue(value) && property === 'length') return makeNumber(value.value.length)
  if (isDateValue(value)) {
    if (property === 'year') return makeNumber(value.value.getUTCFullYear())
    if (property === 'month') return makeNumber(value.value.getUTCMonth() + 1)
    if (property === 'day') return makeNumber(value.value.getUTCDate())
    if (property === 'hour') return makeNumber(value.value.getUTCHours())
    if (property === 'minute') return makeNumber(value.value.getUTCMinutes())
    if (property === 'second') return makeNumber(value.value.getUTCSeconds())
    if (property === 'millisecond') return makeNumber(value.value.getUTCMilliseconds())
  }
  if (isObjectValue(value)) {
    return value.value[property] ?? makeNull()
  }
  if (isFileValue(value)) {
    return resolveFileProperty(value.value, property, ctx)
  }
  if (isLinkValue(value)) {
    if (property === 'value') return makeString(value.value)
  }
  return makeNull()
}

const resolveFileSlug = (file: QuartzPluginData): string | undefined => {
  if (!file.slug) return undefined
  return simplifySlug(file.slug as FullSlug)
}

const normalizeLinkText = (value: string): string => value.trim()

const LINK_SLUG_CACHE_MAX = 4096
const linkSlugCache = new Map<string, string | null>()

const computeLinkSlugFromTrimmed = (
  trimmed: string,
  currentSlug: string | undefined,
): string | undefined => {
  if (/^[a-z][a-z0-9+.-]*:/.test(trimmed)) return undefined
  const parsed = parseWikilink(trimmed)
  if (parsed) {
    if (/^[a-z][a-z0-9+.-]*:/.test(parsed.target)) return undefined
    if (currentSlug) {
      const resolved = resolveWikilinkTarget(parsed, currentSlug as FullSlug)
      return resolved ? simplifySlug(resolved.slug) : undefined
    }
    const parsedTarget = parsed.target.trim()
    if (!parsedTarget) return undefined
    const slug = slugifyFilePath(parsedTarget as FilePath)
    return simplifySlug(slug)
  }
  const [target, anchor] = splitAnchor(trimmed)
  if (currentSlug) {
    const resolved = resolveWikilinkTarget(
      {
        raw: trimmed,
        target,
        anchor: anchor.length > 0 ? anchor : undefined,
        alias: undefined,
        embed: trimmed.startsWith('!'),
      },
      currentSlug as FullSlug,
    )
    if (resolved) return simplifySlug(resolved.slug)
  }
  if (!target) return undefined
  const normalized = target.replace(/\\/g, '/').replace(/^\/+/, '')
  if (!normalized) return undefined
  const slug = slugifyFilePath(normalized as FilePath)
  return simplifySlug(slug)
}

export const resolveLinkSlugFromText = (raw: string, ctx: EvalContext): string | undefined => {
  const trimmed = raw.trim()
  if (!trimmed) return undefined
  const currentSlug = ctx.file.slug
    ? String(ctx.file.slug)
    : ctx.thisFile?.slug
      ? String(ctx.thisFile.slug)
      : undefined
  const needsCurrentSlug =
    trimmed.startsWith('#') ||
    trimmed.startsWith('[[#') ||
    trimmed.startsWith('![[#') ||
    trimmed === '[[]]' ||
    trimmed === '![[]]'
  const cacheKey = needsCurrentSlug ? `${currentSlug ?? ''}\0${trimmed}` : trimmed
  const cached = linkSlugCache.get(cacheKey)
  if (cached !== undefined) {
    return cached ?? undefined
  }
  const result = computeLinkSlugFromTrimmed(trimmed, currentSlug)
  if (linkSlugCache.size >= LINK_SLUG_CACHE_MAX) {
    const oldestKey = linkSlugCache.keys().next().value
    if (oldestKey !== undefined) {
      linkSlugCache.delete(oldestKey)
    }
  }
  linkSlugCache.set(cacheKey, result ?? null)
  return result
}

export const resolveLinkSlugFromValue = (value: Value, ctx: EvalContext): string | undefined => {
  if (isFileValue(value)) return resolveFileSlug(value.value)
  if (isLinkValue(value)) return resolveLinkSlugFromText(value.value, ctx)
  if (isStringValue(value)) return resolveLinkSlugFromText(value.value, ctx)
  return undefined
}

export const resolveLinkComparisonKey = (
  value: Value,
  ctx: EvalContext,
): { slug?: string; text: string } => {
  if (isFileValue(value)) {
    const slug = resolveFileSlug(value.value)
    return { slug, text: slug ?? '' }
  }
  if (isLinkValue(value)) {
    const slug = resolveLinkSlugFromText(value.value, ctx)
    const text = value.display && value.display.length > 0 ? value.display : value.value
    return { slug, text: normalizeLinkText(text) }
  }
  if (isStringValue(value)) {
    const slug = resolveLinkSlugFromText(value.value, ctx)
    return { slug, text: normalizeLinkText(value.value) }
  }
  return { text: normalizeLinkText(valueToString(value)) }
}

export const findFileByTarget = (
  target: string,
  ctx: EvalContext,
): QuartzPluginData | undefined => {
  const slug = resolveLinkSlugFromText(target, ctx)
  if (!slug) return undefined
  const indexed = ctx.fileIndex?.get(slug)
  if (indexed) return indexed
  return ctx.allFiles.find(entry => resolveFileSlug(entry) === slug)
}
