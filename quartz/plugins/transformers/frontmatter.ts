import matter from "gray-matter"
import remarkFrontmatter from "remark-frontmatter"
import { QuartzTransformerPlugin } from "../types"
import yaml from "js-yaml"
import toml from "toml"
import { slugTag } from "../../util/path"
import { QuartzPluginData } from "../vfile"
import { i18n } from "../../i18n"

export interface Options {
  delimiters: string | [string, string]
  language: "yaml" | "toml"
}

const defaultOptions: Options = {
  delimiters: "---",
  language: "yaml",
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

export const FrontMatter: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "FrontMatter",
    markdownPlugins({ cfg }) {
      return [
        [remarkFrontmatter, ["yaml", "toml"]],
        () => {
          return (_, file) => {
            const { data } = matter(Buffer.from(file.value), {
              ...opts,
              engines: {
                yaml: (s) => yaml.load(s, { schema: yaml.JSON_SCHEMA }) as object,
                toml: (s) => toml.parse(s) as object,
              },
            })

            if (data.title != null && data.title.toString() !== "") {
              data.title = data.title.toString()
            } else {
              data.title = file.stem ?? i18n(cfg.configuration.locale).propertyDefaults.title
            }

            const tags = coerceToArray(coalesceAliases(data, ["tags", "tag"]))
            if (tags) data.tags = [...new Set(tags.map((tag: string) => slugTag(tag)))]

            const aliases = coerceToArray(coalesceAliases(data, ["aliases", "alias"]))
            if (aliases) data.aliases = aliases
            const cssclasses = coerceToArray(coalesceAliases(data, ["cssclasses", "cssclass"]))
            if (cssclasses) data.cssclasses = cssclasses

            const permalinks = coerceToArray(coalesceAliases(data, ["permalinks", "permalink"]))
            if (permalinks) data.permalinks = permalinks

            const socialImage = coalesceAliases(data, ["socialImage", "image", "cover"])
            if (socialImage) data.socialImage = socialImage

            const description = coalesceAliases(data, ["description", "socialDescription"])
            if (description) data.description = description

            const transclude = coalesceAliases(data, ["transclude", "transclusion"])
            if (transclude) data.transclude = transclude

            const socials = coalesceAliases(data, ["social", "socials"])
            if (socials) data.socials = socials

            const created = coalesceAliases(data, ["date", "created"])
            if (created) data.created = created
            const modified = coalesceAliases(data, [
              "lastmod",
              "updated",
              "last-modified",
              "modified",
            ])
            if (modified) data.modified = modified
            const published = coalesceAliases(data, ["publishDate", "published", "date"])
            if (published) data.published = published

            // fill in frontmatter
            file.data.frontmatter = data as QuartzPluginData["frontmatter"]
          }
        },
      ]
    },
  }
}

export type TranscludeOptions = {
  dynalist: boolean
  title: boolean
}

declare module "vfile" {
  interface DataMap {
    frontmatter: { [key: string]: unknown } & {
      title: string
    } & Partial<{
        priority: number | undefined
        permalinks: string[]
        tags: string[]
        aliases: string[]
        created: string
        modified: string
        published: string
        description: string
        publish: boolean
        draft: boolean
        lang: string
        enableToc: string
        cssclasses: string[]
        socialImage: string
        noindex: boolean
        comments: boolean
        transclude: Partial<TranscludeOptions>
        signature: string
        socials: Record<string, string>
        pageLayout: "default" | "letters" | "technical" | "reflection"
      }>
  }
}
