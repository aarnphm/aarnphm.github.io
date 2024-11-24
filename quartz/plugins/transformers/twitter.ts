import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { Html, Root } from "mdast"
import { SKIP, visit } from "unist-util-visit"
import { unescapeHTML } from "../../util/escape"

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

const cache = new Map()

export const Twitter: QuartzTransformerPlugin = () => ({
  name: "Twitter",
  markdownPlugins(ctx) {
    const locale = ctx.cfg.configuration.locale.split("-")[0] ?? "en"
    return [
      () => async (tree: Root, _file) => {
        const promises: Promise<void>[] = []

        const fetchEmbedded = async (parent: Root, index: number, url: string, locale: string) => {
          let value: string

          const cacheKey = `twitter:${url}`
          let htmlString = cache.get(cacheKey)
          if (!htmlString) {
            try {
              const data: TwitterEmbed = await fetch(
                `https://publish.twitter.com/oembed?url=${url}&dnt=false&omit_script=true&lang=${locale}`,
              ).then((res) => res.json())
              value = unescapeHTML(data.html)
              cache.set(cacheKey, value)
            } catch (error) {
              value = `<p>Link to original <a href="${url}">tweet</a>.</p>`
            }
          }

          const node: Html = { type: "html", value }
          parent!.children.splice(index, 1, node)
        }

        visit(tree, "paragraph", (node, index, parent) => {
          if (node.children.length === 0) return SKIP

          // find first line and callout content
          const [firstChild] = node.children
          if (firstChild.type !== "link" || !twitterUrlRegex.test(firstChild.url)) return SKIP

          promises.push(fetchEmbedded(parent as Root, index!, firstChild.url, locale))
        })

        if (promises.length > 0) await Promise.all(promises)
      },
    ]
  },
})
