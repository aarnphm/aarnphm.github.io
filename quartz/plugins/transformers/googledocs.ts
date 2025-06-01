import path from "path"
import os from "os"
import fs from "fs/promises"
import { google } from "googleapis"
import { authenticate } from "@google-cloud/local-auth"
import { QuartzTransformerPlugin } from "../types"
import { Root, Link } from "mdast"
import { visit } from "unist-util-visit"

const docs = google.docs({ version: "v1" })

async function getDocsName(id: string) {
  try {
    const item = await docs.documents.get({ documentId: id })
    return item.data.title
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
        return async (tree: Root, _file) => {
          const docsLinks: { node: Link; documentId: string }[] = []
          const tmpKeyfile = path.join(os.tmpdir(), "google-cloud-key.json")
          await fs.writeFile(tmpKeyfile, process.env.GOOGLECLOUD_API_KEY, "utf-8")
          const auth = await authenticate({
            keyfilePath: tmpKeyfile,
            scopes: ["https://www.googleapis.com/auth/documents"],
          })
          google.options({ auth })

          visit(tree, "link", (node: Link) => {
            const match = node.url.match(
              /https?:\/\/docs\.google\.com\/document(?:\/u\/\d+)?\/d\/([a-zA-Z0-9_-]+)/,
            )
            console.log(match)
            if (match) {
              docsLinks.push({ node, documentId: match[1] })
            }
          })

          if (docsLinks.length > 0) {
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
          }
        }
      },
    ]
  },
})
