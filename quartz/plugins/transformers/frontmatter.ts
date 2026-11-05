import type { VFile } from 'vfile'
import matter from 'gray-matter'
import yaml from 'js-yaml'
import remarkFrontmatter from 'remark-frontmatter'
import type { BuildCtx } from '../../util/ctx'
import type { ReferenceSyntax } from '../../util/references/model'
import { i18n } from '../../i18n'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { FullSlug } from '../../util/path'
import { referenceCatalogFor, referenceSourcePath } from '../../util/references/context'
import {
  extractLiteralReferences,
  resolveLiteralReference,
  type LiteralReferenceResolutionOptions,
} from '../../util/references/literal'
import { recordReferenceDependency } from '../../util/references/markdown'
import { referenceFields, referenceRequest } from '../../util/references/request'
import { missingReferenceRoute } from '../../util/references/route'
import { tagMetadata } from '../../util/tags'
import { isRecord } from '../../util/type-guards'
import { ContentLayout } from '../emitters/contentIndex'
import { QuartzPluginData } from '../vfile'

function getAliasSlugs(aliases: string[]): FullSlug[] {
  return aliases.flatMap(alias => {
    const route = missingReferenceRoute(alias, 'index')
    return route ? [route.slug] : []
  })
}

function coalesceAliases(data: Record<string, unknown>, aliases: string[]) {
  for (const alias of aliases) {
    if (data[alias] !== undefined && data[alias] !== null) return data[alias]
  }
}

function coerceToArray(input: unknown): string[] | undefined {
  if (input === undefined || input === null) return undefined
  const values: unknown[] = Array.isArray(input)
    ? input
    : String(input)
        .split(',')
        .map(value => value.trim())
  return values
    .filter(
      (value): value is string | number => typeof value === 'string' || typeof value === 'number',
    )
    .map(String)
}

function coerceToBoolean(input: unknown): boolean | undefined {
  if (input === undefined || input === null) return undefined
  if (typeof input === 'boolean') return input
  if (typeof input === 'number') return input !== 0

  if (Array.isArray(input)) {
    for (const value of input) {
      const coerced = coerceToBoolean(value)
      if (coerced !== undefined) return coerced
    }
    return undefined
  }

  if (typeof input === 'string') {
    const normalized = input.trim().toLowerCase()
    if (['true', 't', 'yes', 'y', '1', 'on'].includes(normalized)) return true
    if (['false', 'f', 'no', 'n', '0', 'off'].includes(normalized)) return false
  }

  return undefined
}

export interface FrontmatterLink {
  raw: string
  slug: FullSlug
  anchor?: string
  alias?: string
}

function collectValueLinks(
  value: unknown,
  currentSlug: FullSlug,
  options: LiteralReferenceResolutionOptions,
  record?: (link: ReferenceSyntax) => void,
): FrontmatterLink[] {
  const results: FrontmatterLink[] = []
  const seen = new Set<string>()

  const visitValue = (val: unknown) => {
    if (typeof val === 'string') {
      for (const link of extractLiteralReferences(val)) {
        record?.(link)
        const resolved = resolveLiteralReference(link, currentSlug, options)
        if (!resolved) continue

        const fingerprint = `${resolved.slug}${resolved.anchor ?? ''}`
        if (seen.has(fingerprint)) continue
        seen.add(fingerprint)

        results.push({
          raw: link.raw,
          slug: resolved.slug,
          anchor: resolved.anchor,
          alias: referenceFields(link).label,
        })
      }
    } else if (Array.isArray(val)) {
      for (const item of val) {
        visitValue(item)
      }
    } else if (isRecord(val)) {
      for (const inner of Object.values(val)) {
        visitValue(inner)
      }
    }
  }

  visitValue(value)
  return results
}

function collectFrontmatterLinks(
  data: Record<string, unknown>,
  currentSlug: FullSlug,
  options: LiteralReferenceResolutionOptions,
  record?: (link: ReferenceSyntax) => void,
): Record<string, FrontmatterLink[]> | undefined {
  const result: Record<string, FrontmatterLink[]> = {}

  for (const [key, value] of Object.entries(data)) {
    const links = collectValueLinks(value, currentSlug, options, record)
    if (links.length > 0) {
      result[key] = links
    }
  }

  return Object.keys(result).length > 0 ? result : undefined
}

function buildFrontmatterLinkLookup(
  links: Record<string, FrontmatterLink[]> | undefined,
): Record<string, FrontmatterLink> | undefined {
  if (!links) return undefined

  const lookup: Record<string, FrontmatterLink> = {}

  for (const group of Object.values(links)) {
    for (const link of group) {
      if (!lookup[link.raw]) {
        lookup[link.raw] = link
      }
    }
  }

  return Object.keys(lookup).length > 0 ? lookup : undefined
}

function refreshFrontmatterLinks(ctx: BuildCtx, file: VFile, record: boolean): void {
  const currentSlug = file.data.slug
  const data = file.data.authoredProperties ?? file.data.frontmatter
  if (!currentSlug || !isRecord(data)) return
  const options = { catalog: referenceCatalogFor(ctx), origin: referenceSourcePath(file.data) }
  const links = collectFrontmatterLinks(
    data,
    currentSlug,
    options,
    record
      ? link => {
          recordReferenceDependency(ctx, file, referenceRequest(link, options.origin))
        }
      : undefined,
  )
  if (links) file.data.frontmatterLinks = links
  else delete file.data.frontmatterLinks
  const lookup = buildFrontmatterLinkLookup(links)
  if (lookup) file.data.frontmatterLinkLookup = lookup
  else delete file.data.frontmatterLinkLookup
}

export const FrontMatter: QuartzTransformerPlugin = () => ({
  name: 'FrontMatter',
  markdownPlugins: ctx => {
    const { cfg } = ctx
    return [
      [remarkFrontmatter, ['yaml', 'toml']],
      () => {
        return (_, file) => {
          const { data } = matter(Buffer.from(file.value), {
            delimiters: '---',
            language: 'yaml',
            engines: { yaml: s => yaml.load(s, { schema: yaml.JSON_SCHEMA }) as object },
          })
          const authoredProperties: Record<string, unknown> = structuredClone(data)
          file.data.authoredProperties = authoredProperties

          if (data.title != null && data.title.toString() !== '') {
            data.title = data.title.toString()
          } else {
            data.title = file.stem ?? i18n(cfg.configuration.locale).propertyDefaults.title
          }

          const tags = coerceToArray(coalesceAliases(data, ['tags']))
          if (tags) {
            const normalized = tagMetadata(tags)
            data.tags = normalized.tags
            file.data.tagDisplay = normalized.display
            file.data.tagAliases = normalized.aliases
          }

          const aliases = coerceToArray(coalesceAliases(data, ['aliases', 'alias']))
          if (aliases) {
            data.aliases = aliases
            file.data.aliases = getAliasSlugs(aliases)
          }
          const permalinks = coerceToArray(coalesceAliases(data, ['permalink', 'permalinks']))

          if (permalinks) {
            data.permalinks = permalinks
            const aliases = file.data.aliases ?? []
            aliases.push(...permalinks)
            file.data.aliases = aliases
          }

          const cssclasses = coerceToArray(coalesceAliases(data, ['cssclasses']))
          if (cssclasses) data.cssclasses = cssclasses

          const noindex = coerceToBoolean(coalesceAliases(data, ['noindex', 'unlisted']))
          if (noindex !== undefined) data.noindex = noindex

          const settings: unknown = data.settings
          if (settings && typeof settings === 'object' && !Array.isArray(settings)) {
            data.settings = {
              ...settings,
              'heading-toggle': coerceToBoolean(
                'heading-toggle' in settings ? settings['heading-toggle'] : undefined,
              ),
            }
          } else {
            delete data.settings
          }

          const currentSlug = file.data.slug
          const referenceOptions = {
            catalog: referenceCatalogFor(ctx),
            origin: referenceSourcePath(file.data),
          }

          const socialImage = coalesceAliases(data, ['socialImage', 'image', 'cover'])
          if (typeof socialImage === 'string') {
            // Check if socialImage contains wikilinks
            const wikilinks = extractLiteralReferences(socialImage)
            if (wikilinks.length > 0 && currentSlug) {
              const resolved = resolveLiteralReference(wikilinks[0], currentSlug, referenceOptions)
              if (resolved) {
                data.socialImage = `https://${cfg.configuration.baseUrl}/${resolved.slug}`
              } else {
                data.socialImage = socialImage
              }
            } else {
              data.socialImage = socialImage
            }
          } else if (socialImage !== undefined && socialImage !== null) {
            delete data.socialImage
            file.message('Social image metadata must be a string; quote wikilinks in YAML.', {
              source: 'frontmatter',
              ruleId: 'social-image',
            })
          }

          const description = coalesceAliases(data, ['description', 'socialDescription'])
          if (description) data.description = description

          const transclude = coalesceAliases(data, ['transclude', 'transclusion'])
          if (transclude) data.transclude = transclude

          const socials = coalesceAliases(data, ['social', 'socials'])
          if (socials) data.socials = socials

          const authors = coalesceAliases(data, ['author', 'authors'])
          if (authors) data.authors = authors

          const slides = coalesceAliases(data, ['slides', 'slide', 'ppt', 'powerpoint'])
          if (slides) data.slides = slides

          const created = coalesceAliases(data, ['date', 'created'])
          if (created) {
            data.created = created
          }
          const modified = coalesceAliases(data, [
            'lastmod',
            'updated',
            'last-modified',
            'modified',
          ])
          if (modified) data.modified = modified
          data.modified ||= created // if modified is not set, use created

          const published = coalesceAliases(data, ['publishDate', 'published', 'date'])
          if (published) data.published = published

          let layout = coalesceAliases(data, ['pageLayout', 'folderLayout', 'layout'])
          layout ||= 'default'
          data.pageLayout = layout

          // fill in frontmatter
          file.data.frontmatter = data as QuartzPluginData['frontmatter']
          refreshFrontmatterLinks(ctx, file, false)
        }
      },
    ]
  },
  htmlPlugins: ctx => [
    () => (_tree, file) => {
      refreshFrontmatterLinks(ctx, file, true)
    },
  ],
})

export type TranscludeOptions = {
  dynalist: boolean
  title: boolean
  headings: boolean
  skipTranscludes: boolean
}

declare module 'vfile' {
  interface DataMap {
    aliases: string[]
    frontmatter: { [key: string]: unknown } & {
      title: string
      pageLayout: ContentLayout
    } & Partial<{
        settings: { 'heading-toggle'?: boolean }
        flashcards: { default?: boolean }
        masonry: string[]
        priority: number | undefined
        permalinks: string[]
        tags: string[]
        aliases: string[]
        abstract: string
        created: string
        modified: string
        published: string
        description: string
        publish: boolean
        draft: boolean
        private: boolean
        lang: string
        enableToc: string
        cssclasses: string[]
        socialImage: string
        socialDescription: string
        noindex: boolean
        comments: boolean
        slides: boolean
        collapseHeadings: boolean
        folderSort: string
        transclude: Partial<TranscludeOptions>
        signature: string
        socials: Record<string, string>
        authors: string[]
        email: boolean
      }>
    frontmatterLinks?: Record<string, FrontmatterLink[]>
    frontmatterLinkLookup?: Record<string, FrontmatterLink>
  }
}
