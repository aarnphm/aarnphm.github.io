import { Element } from "hast"
import { Html, Link, Paragraph, PhrasingContent } from "mdast"
import { Parent } from "unist"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../../types/plugin"
import { unescapeHTML } from "../../util/escape"
import { wikiTextTransform } from "./ofm"

export const twitterUrlRegex = /^.*(twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/(status)\/(\d{19}).*/

export function filterEmbedTwitter(node: Element): boolean {
  const href = node.properties.href
  if (href === undefined || typeof href !== "string") return false
  return node.children.length !== 0 && twitterUrlRegex.test(href)
}

type TwitterEmbed = {
  url: string
  author_name: string
  author_url: string
  html: string
  width: number
  height: null
  type: "rich"
  cache_age: number
  provider_name: "Twitter"
  provider_url: "https://twitter.com"
  version: "1.0"
}

const cache = new Map<string, string>()

const fallbackHtml = (url: string) =>
  `<p class="twitter-fallback">Link to original <a href="${url}">tweet</a>.</p>`

const isWhitespaceNode = (node: PhrasingContent) => {
  if (node.type !== "text") return false
  return node.value.trim() === ""
}

const isNakedLink = (parent: Paragraph, child: Link) => {
  const meaningfulChildren = parent.children.filter((node) => !isWhitespaceNode(node))
  if (meaningfulChildren.length !== 1 || meaningfulChildren[0] !== child) {
    return false
  }

  const linkText = child.children
    .map((node) => (node.type === "text" ? node.value : ""))
    .join("")
    .trim()

  return linkText.length === 0 || linkText === child.url
}

export async function fetchTwitterEmbed(url: string, locale: string): Promise<string> {
  const cacheKey = `twitter:${locale}:${url}`
  const cached = cache.get(cacheKey)
  if (cached) {
    return cached
  }

  let value = fallbackHtml(url)

  try {
    const res = await fetch(
      `https://publish.twitter.com/oembed?url=${url}&dnt=true&omit_script=true&lang=${locale}`,
    )
    if (!res.ok) {
      throw new Error(`twitter oEmbed request failed with status ${res.status}`)
    }
    const data = (await res.json()) as TwitterEmbed
    value = unescapeHTML(data.html)
  } catch {
    // swallow network failures and fall back to a simple link
    value = fallbackHtml(url)
  }

  cache.set(cacheKey, value)
  return value
}

export const Twitter: QuartzTransformerPlugin = () => ({
  name: "Twitter",
  textTransform(_, src) {
    src = wikiTextTransform(src)

    return src
  },
  markdownPlugins({ cfg }) {
    const locale = cfg.configuration.locale.split("-")[0] ?? "en"
    return [
      () => async (tree, file) => {
        const fileData = file.data
        if (fileData.slug === "are.na") return

        const promises: Promise<void>[] = []

        const fetchEmbedded = async (
          parent: Parent,
          index: number,
          url: string,
          locale: string,
        ) => {
          const value = await fetchTwitterEmbed(url, locale)
          parent.children.splice(index, 1, { type: "html", value } as Html)
        }

        visit(tree, "paragraph", (node: Paragraph) => {
          for (let i = 0; i < node.children.length; i++) {
            const child = node.children[i]
            if (
              child.type === "link" &&
              twitterUrlRegex.test(child.url) &&
              isNakedLink(node, child)
            ) {
              promises.push(fetchEmbedded(node, i, child.url, locale))
            }
          }
        })

        if (promises.length > 0) await Promise.all(promises)
      },
    ]
  },
})
