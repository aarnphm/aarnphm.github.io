import fs from "node:fs/promises"
import path from "node:path"
import rehypeParse from "rehype-parse"
import rehypeRemark from "rehype-remark"
import remarkGfm from "remark-gfm"
import remarkStringify from "remark-stringify"
import { unified } from "unified"

async function htmlToMarkdown(htmlString) {
  // Build a pipeline that:
  // 1. Parses HTML
  // 2. Converts the HTML AST (hAST) to a Markdown AST (mdAST)
  // 3. Stringifies the mdAST into Markdown text
  const file = await unified()
    .use(rehypeParse, { fragment: true }) // parse HTML
    .use(rehypeRemark) // convert to remark (Markdown AST)
    .use(remarkGfm)
    .use(remarkStringify) // stringify to Markdown
    .process(htmlString)

  return String(file)
}

async function readBaseTemplate() {
  try {
    const currentDir = path.dirname(new URL(import.meta.url).pathname)
    const templatePath = path.join(currentDir, "base.html")
    return await fs.readFile(templatePath, "utf8")
  } catch (error) {
    console.error("Error reading base template:", error)
    throw error
  }
}

const docs = await readBaseTemplate()
const md = await htmlToMarkdown(docs)
console.log(md)
