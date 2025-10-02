import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { Html } from "mdast"
import { Parent } from "unist"
import { visit } from "unist-util-visit"
import { unescapeHTML } from "../../util/escape"
// @ts-ignore
import script from "../../components/scripts/twitter.inline"
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

const cache = new Map()

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
          let value = `<p>Link to original <a href="${url}">tweet</a>.</p>`

          const cacheKey = `twitter:${url}`
          let htmlString = cache.get(cacheKey)
          if (!htmlString) {
            await fetch(
              `https://publish.twitter.com/oembed?url=${url}&dnt=true&omit_script=true&lang=${locale}`,
            )
              .then((res) => res.json())
              .then((data) => {
                value = unescapeHTML((data as TwitterEmbed).html)
                cache.set(cacheKey, value)
                return value
              })
              .catch(() => value)
          }
          parent.children.splice(index, 1, { type: "html", value } as Html)
        }

        visit(tree, "paragraph", (node) => {
          for (let i = 0; i < node.children.length; i++) {
            const child = node.children[i]
            if (child.type === "link" && twitterUrlRegex.test(child.url)) {
              promises.push(fetchEmbedded(node, i, child.url, locale))
            }
          }
        })

        if (promises.length > 0) await Promise.all(promises)
      },
    ]
  },
})
