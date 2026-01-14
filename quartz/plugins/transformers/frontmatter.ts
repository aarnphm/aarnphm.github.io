import matter from "gray-matter"
import remarkFrontmatter from "remark-frontmatter"
import { QuartzTransformerPlugin } from "../../types/plugin"
import yaml from "js-yaml"
import { FilePath, FullSlug, slugifyFilePath, slugTag, getFileExtension } from "../../util/path"
import { extractWikilinks, resolveWikilinkTarget } from "../../util/wikilinks"
import { QuartzPluginData } from "../vfile"
import { i18n } from "../../i18n"
import { ContentLayout } from "../emitters/contentIndex"

function getAliasSlugs(aliases: string[]): FullSlug[] {
  const res: FullSlug[] = []
  for (const alias of aliases) {
    const isMd = getFileExtension(alias) === "md"
    const mockFp = isMd ? alias : alias + ".md"
    const slug = slugifyFilePath(mockFp as FilePath)
    res.push(slug)
  }
  return res
}

function coalesceAliases(data: { [key: string]: any }, aliases: string[]) {
  for (const alias of aliases) {
    if (data[alias] !== undefined && data[alias] !== null) return data[alias]
  }
}

function coerceToArray(input: string | string[]): string[] | undefined {
  if (input === undefined || input === null) return undefined

  // coerce to array
  if (!Array.isArray(input)) {
    input = input
      .toString()
      .split(",")
      .map((tag: string) => tag.trim())
  }

  // remove all non-strings
  return input
    .filter((tag: unknown) => typeof tag === "string" || typeof tag === "number")
    .map((tag: string | number) => tag.toString())
}

export interface FrontmatterLink {
  raw: string
  slug: FullSlug
  anchor?: string
  alias?: string
}

function collectValueLinks(value: unknown, currentSlug: FullSlug): FrontmatterLink[] {
  const results: FrontmatterLink[] = []
  const seen = new Set<string>()

  const visitValue = (val: unknown) => {
    if (typeof val === "string") {
      for (const link of extractWikilinks(val)) {
        const resolved = resolveWikilinkTarget(link, currentSlug)
        if (!resolved) continue

        const fingerprint = `${resolved.slug}${resolved.anchor ?? ""}`
        if (seen.has(fingerprint)) continue
        seen.add(fingerprint)

        results.push({
          raw: link.raw,
          slug: resolved.slug,
          anchor: resolved.anchor,
          alias: link.alias,
        })
      }
    } else if (Array.isArray(val)) {
      for (const item of val) {
        visitValue(item)
      }
    } else if (val && typeof val === "object" && val.constructor === Object) {
      for (const inner of Object.values(val as Record<string, unknown>)) {
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
): Record<string, FrontmatterLink[]> | undefined {
  const result: Record<string, FrontmatterLink[]> = {}

  for (const [key, value] of Object.entries(data)) {
    const links = collectValueLinks(value, currentSlug)
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

export const FrontMatter: QuartzTransformerPlugin = () => ({
  name: "FrontMatter",
  markdownPlugins: ({ cfg, allSlugs }) => [
    [remarkFrontmatter, ["yaml", "toml"]],
    () => {
      return (_, file) => {
        const { data } = matter(Buffer.from(file.value), {
          delimiters: "---",
          language: "yaml",
          engines: {
            yaml: (s) => yaml.load(s, { schema: yaml.JSON_SCHEMA }) as object,
          },
        })

        if (data.title != null && data.title.toString() !== "") {
          data.title = data.title.toString()
        } else {
          data.title = file.stem ?? i18n(cfg.configuration.locale).propertyDefaults.title
        }

        const tags = coerceToArray(coalesceAliases(data, ["tags"]))
        if (tags) data.tags = [...new Set(tags.map((tag: string) => slugTag(tag)))]

        const aliases = coerceToArray(coalesceAliases(data, ["aliases", "alias"]))
        if (aliases) {
          data.aliases = aliases
          file.data.aliases = getAliasSlugs(aliases)
          allSlugs.push(...file.data.aliases)
        }
        const permalinks = coerceToArray(coalesceAliases(data, ["permalink", "permalinks"]))

        if (permalinks) {
          data.permalinks = permalinks as FullSlug[]
          const aliases = file.data.aliases ?? []
          aliases.push(...data.permalinks)
          file.data.aliases = aliases
          allSlugs.push(data.permalinks)
        }

        const cssclasses = coerceToArray(coalesceAliases(data, ["cssclasses"]))
        if (cssclasses) data.cssclasses = cssclasses

        const noindex = coerceToArray(coalesceAliases(data, ["noindex", "unlisted"]))
        if (noindex) data.noindex = noindex

        const currentSlug = file.data.slug as FullSlug | undefined

        const socialImage = coalesceAliases(data, ["socialImage", "image", "cover"])
        if (socialImage) {
          // Check if socialImage contains wikilinks
          const wikilinks = extractWikilinks(socialImage)
          if (wikilinks.length > 0 && currentSlug) {
            const resolved = resolveWikilinkTarget(wikilinks[0], currentSlug)
            if (resolved) {
              data.socialImage = `https://${cfg.configuration.baseUrl}/${resolved.slug}`
            } else {
              data.socialImage = socialImage
            }
          } else {
            data.socialImage = socialImage
          }
        }

        const description = coalesceAliases(data, ["description", "socialDescription"])
        if (description) data.description = description

        const transclude = coalesceAliases(data, ["transclude", "transclusion"])
        if (transclude) data.transclude = transclude

        const socials = coalesceAliases(data, ["social", "socials"])
        if (socials) data.socials = socials

        const authors = coalesceAliases(data, ["author", "authors"])
        if (authors) data.authors = authors

        const slides = coalesceAliases(data, ["slides", "slide", "ppt", "powerpoint"])
        if (slides) data.slides = slides

        const created = coalesceAliases(data, ["date", "created"])
        if (created) {
          data.created = created
        }
        const modified = coalesceAliases(data, ["lastmod", "updated", "last-modified", "modified"])
        if (modified) data.modified = modified
        data.modified ||= created // if modified is not set, use created

        const published = coalesceAliases(data, ["publishDate", "published", "date"])
        if (published) data.published = published

        let layout = coalesceAliases(data, ["pageLayout", "folderLayout", "layout"])
        layout ||= "default"
        data.pageLayout = layout

        if (currentSlug) {
          const frontmatterLinks = collectFrontmatterLinks(
            data as Record<string, unknown>,
            currentSlug,
          )
          if (frontmatterLinks) {
            file.data.frontmatterLinks = frontmatterLinks
            const lookup = buildFrontmatterLinkLookup(frontmatterLinks)
            if (lookup) {
              file.data.frontmatterLinkLookup = lookup
            }
          }
        }

        // fill in frontmatter
        file.data.frontmatter = data as QuartzPluginData["frontmatter"]
      }
    },
  ],
})

export type TranscludeOptions = {
  dynalist: boolean
  title: boolean
  headings: boolean
  skipTranscludes: boolean
}

declare module "vfile" {
  interface DataMap {
    aliases: FullSlug[]
    frontmatter: { [key: string]: unknown } & {
      title: string
      pageLayout: ContentLayout
    } & Partial<{
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
        transclude: Partial<TranscludeOptions>
        signature: string
        socials: Record<string, string>
        authors: string[]
      }>
    frontmatterLinks?: Record<string, FrontmatterLink[]>
    frontmatterLinkLookup?: Record<string, FrontmatterLink>
  }
}
