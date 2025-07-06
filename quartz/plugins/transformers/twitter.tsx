import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { Root } from "mdast"
import { visit } from "unist-util-visit"
import { unescapeHTML } from "../../util/escape"
// @ts-ignore
import script from "../../components/scripts/twitter.inline"

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
  markdownPlugins: ({ cfg }) => {
    const locale = cfg.configuration.locale.split("-")[0] ?? "en"
    return [
      () => async (tree) => {
        const promises: Promise<void>[] = []

        const fetchEmbedded = async (parent: Root, index: number, url: string, locale: string) => {
          let value = `<p>Link to original <a href="${url}">tweet</a>.</p>`

          const cacheKey = `twitter:${url}`
          let htmlString = cache.get(cacheKey)
          if (!htmlString) {
            await fetch(
              `https://publish.twitter.com/oembed?url=${url}&dnt=true&omit_script=true&lang=${locale}`,
            )
              .then((res) => res.json())
              .catch((error) => {
                console.error(`Failed to fetch Twitter embed for ${url}:`, error)
                return value
              })
              .then((data: TwitterEmbed) => {
                value = unescapeHTML(data.html)
                cache.set(cacheKey, value)
                return value
              })
          }
          parent!.children.splice(index, 1, { type: "html", value })
        }

        visit(tree, "paragraph", (node, index, parent) => {
          // find first line and callout content
          const [firstChild] = node.children
          if (firstChild.type === "link" && twitterUrlRegex.test(firstChild.url)) {
            promises.push(fetchEmbedded(parent as Root, index!, firstChild.url, locale))
          }
        })

        if (promises.length > 0) await Promise.all(promises)
      },
    ]
  },
})
