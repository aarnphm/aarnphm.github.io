import { google } from "googleapis"
import { QuartzTransformerPlugin } from "../types"
import { Root, Link, PhrasingContent } from "mdast"
import { findAndReplace } from "mdast-util-find-and-replace"
import { RegExpMatchObject } from "mdast-util-find-and-replace"
import { visit } from "unist-util-visit"

// example docs url: https://docs.google.com/document/d/1X-VQD2U0E2Jb0ncmjxCruyQO02Z_cgB46sinpVk97-A/edit
// We will need to get the document_id here
const docsRegex = /https?:\/\/docs\.google\.com\/document\/d\/([a-zA-Z0-9_-]+)(?:\/[^\/\s]*)?/gi

const docs = google.docs({ version: "v1", auth: "GOOGLECLOUD_API_KEY" })

async function getDocsName(id: string): Promise<string> {
  try {
    const item = await docs.documents.get({ documentId: id })
    return item.data.title || "Untitled Document"
  } catch (error) {
    console.error(`Failed to fetch document title for ${id}:`, error)
    return "Google Doc"
  }
}

export const GoogleDocs: QuartzTransformerPlugin = () => ({
  name: "GoogleDocs",
  markdownPlugins() {
    return [
      () => {
        return async (tree: Root, file) => {
          const docsLinks: { node: Link; documentId: string }[] = []

          visit(tree, "link", (node: Link) => {
            const match = node.url.match(
              /https?:\/\/docs\.google\.com\/document\/d\/([a-zA-Z0-9_-]+)/,
            )
            if (match) {
              docsLinks.push({ node, documentId: match[1] })
            }
          })

          await Promise.all(
            docsLinks.map(async ({ node, documentId }) => {
              const title = await getDocsName(documentId)
              node.children = [
                {
                  type: "text",
                  value: `📄 ${title}`,
                },
              ]
            }),
          )

          findAndReplace(
            tree,
            [
              [
                docsRegex,
                (
                  value: string,
                  documentId: string,
                  match: RegExpMatchObject,
                ): PhrasingContent | false => {
                  if (match.index > 0 && /[<"]/.test(match.input.charAt(match.index - 1))) {
                    return false
                  }

                  return {
                    type: "link",
                    url: value,
                    children: [
                      {
                        type: "text",
                        value: `Google Doc (${documentId.substring(0, 8)}...)`,
                      },
                    ],
                  }
                },
              ],
            ],
            { ignore: ["link", "linkReference"] },
          )
        }
      },
    ]
  },
})
