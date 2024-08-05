import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

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

export const Twitter: QuartzTransformerPlugin = () => ({
  name: "Twitter",
  markdownPlugins(ctx) {
    const locale = ctx.cfg.configuration.locale.split("-")[0] ?? "en"
    return [
      () => async (tree: Root, _file) => {
        const promises: Promise<void>[] = []

        visit(tree, "paragraph", (node, index, parent) => {
          if (node.children.length === 0) return

          // find first line and callout content
          const [firstChild] = node.children
          if (firstChild.type !== "link" || !twitterUrlRegex.test(firstChild.url)) return

          promises.push(
            fetch(
              `https://publish.twitter.com/oembed?url=${firstChild.url}&dnt=false&omit_script=true&lang=${locale}`,
            )
              .then((res) => res.json())
              .then((data: TwitterEmbed) => {
                parent!.children.splice(index!, 1, {
                  type: "html",
                  value: data.html
                    .replace(/\?ref_src=twsrc.*?fw/g, "")
                    .replace(/<br>/g, "<br />")
                    .trim(),
                })
              }),
          )
        })

        await Promise.all(promises)
      },
    ]
  },
})
