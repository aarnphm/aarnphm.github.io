import { Element, Root, Node } from "hast"
import { fromHtml } from "hast-util-from-html"
import { toHtml } from "hast-util-to-html"
import { existsSync, promises as fs } from "node:fs"
import path from "path"
import { EXIT, visit } from "unist-util-visit"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { FullPageLayout } from "../../cfg"
import { Content } from "../../components"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../types/component"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { BuildCtx } from "../../util/ctx"
import {
  FilePath,
  FullSlug,
  getFileExtension,
  joinSegments,
  pathToRoot,
  QUARTZ,
} from "../../util/path"
import { StaticResources } from "../../util/resources"
import { extractWikilinksWithPositions, resolveWikilinkTarget } from "../../util/wikilinks"
import { QuartzPluginData } from "../vfile"

const name = "EmailEmitter"
const emailsPath = path.join(QUARTZ, "static", "emails.txt")
const imageExtensions = new Set([".avif", ".gif", ".jpg", ".jpeg", ".png", ".svg", ".webp"])

const renderEmailRoot = (
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
): Root => {
  const slug = fileData.slug!
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }
  const emailLayout: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: Content(),
    afterBody: [],
    sidebar: [],
  }

  const rendered = renderPage(ctx, slug, componentData, emailLayout, externalResources, false)
  const doc = fromHtml(rendered)
  let body: Element | undefined
  visit(doc, { tagName: "body" }, (node: Element) => {
    body = node
    return EXIT
  })
  if (body) {
    const root: Root = { type: "root", children: body.children }
    return root
  }
  return doc
}

const formatPlainText = (body: string, slug: FullSlug, baseUrl?: string): string => {
  const ranges = extractWikilinksWithPositions(body)
  if (ranges.length === 0) return body
  let result = ""
  let lastIndex = 0
  for (const range of ranges) {
    if (range.start > lastIndex) {
      result += body.slice(lastIndex, range.start)
    }
    const link = range.wikilink
    const raw = link.raw ?? body.slice(range.start, range.end)
    const target = link.target ?? ""
    const ext = getFileExtension(target as FilePath) ?? ""
    const isImage = link.embed && ext.length > 0 && imageExtensions.has(ext)
    if (isImage) {
      const cleaned = target.split(/[?#]/)[0] ?? ""
      const filename = cleaned ? path.basename(cleaned) : "image"
      result += `[image: ${filename}]`
    } else {
      const resolved = resolveWikilinkTarget(link, slug)
      if (!resolved) {
        result += link.alias ?? link.anchorText ?? link.target ?? raw
      } else {
        const base = baseUrl ? `https://${baseUrl}` : ""
        const path = base ? joinSegments(base, resolved.slug) : `/${resolved.slug}`
        const url = resolved.anchor ? `${path}${resolved.anchor}` : path
        const label = link.alias ?? link.anchorText ?? link.target ?? resolved.slug
        result += `${label} <${url}>`
      }
    }
    lastIndex = range.end
  }
  if (lastIndex < body.length) {
    result += body.slice(lastIndex)
  }
  return result
}

export const EmailEmitter: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, content, resources) {
      if (ctx.argv.watch && !ctx.argv.force) return []
      if (process.env.EMAIL_EMITTER_ENABLED !== "1") return []

      const secret = process.env.EMAIL_EMITTER_SECRET
      if (!secret) throw new Error("missing EMAIL_EMITTER_SECRET")

      const recipients = new Set<string>()
      for (const line of (await fs.readFile(emailsPath, "utf8")).split(/\r?\n/)) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith("#")) continue
        const match = trimmed.match(/<([^>]+)>/)
        const candidate = (match ? match[1] : trimmed).trim()
        if (!candidate || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(candidate)) continue
        recipients.add(candidate.toLowerCase())
      }
      const recipientList = [...recipients]
      if (recipientList.length === 0) return []

      const allFiles = content.map((c) => c[1].data)
      const contentRoot = ctx.argv.directory
      const staticRoot = path.join(QUARTZ, "static")
      const contentTypes: Record<string, string> = {
        ".avif": "image/avif",
        ".gif": "image/gif",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
      }

      for (const [tree, file] of content) {
        const fileData: QuartzPluginData = file.data
        if (fileData.frontmatter?.email !== true || fileData.frontmatter?.emailSent !== false)
          continue

        const slug = fileData.slug!
        const relativePath = fileData.relativePath ?? fileData.filePath ?? ""
        const baseName = path.basename(relativePath, path.extname(relativePath))
        const [monthToken, yearToken] = baseName.split("-")
        const monthKey = (monthToken || "").slice(0, 3).toLowerCase()
        const monthIndex =
          {
            jan: 0,
            feb: 1,
            mar: 2,
            apr: 3,
            may: 4,
            jun: 5,
            jul: 6,
            aug: 7,
            sep: 8,
            oct: 9,
            nov: 10,
            dec: 11,
          }[monthKey] ?? Number.parseInt(monthToken ?? "", 10) - 1
        const year = Number.parseInt(yearToken ?? "", 10)
        const emailSuffix = String(monthIndex + 1).padStart(3, "0")
        const subject = `GET updates/${year}/${emailSuffix}`
        const baseUrl = ctx.cfg.configuration.baseUrl

        const root = renderEmailRoot(ctx, tree, fileData, allFiles, resources)
        const attachments: {
          contentId: string
          filename: string
          contentType: string
          content: string
        }[] = []
        const replacements = new Map<string, string>()
        const pending = new Map<string, { nodes: Element[]; sourcePath: string }>()
        const filePath = fileData.filePath!
        const fileDir = path.dirname(filePath)
        const resolveAttachmentPath = (src: string) => {
          const cleaned = decodeURIComponent(src.split(/[?#]/)[0] ?? "")
          if (
            !cleaned ||
            /^([a-z]+:)?\/\//i.test(cleaned) ||
            cleaned.startsWith("data:") ||
            cleaned.startsWith("cid:") ||
            cleaned.startsWith("mailto:")
          ) {
            return null
          }
          const normalized = cleaned.replace(/^\.\//, "")
          if (normalized.startsWith("/static/")) {
            const candidate = path.join(QUARTZ, normalized.slice(1))
            return existsSync(candidate) ? candidate : null
          }
          if (normalized.startsWith("static/")) {
            const candidate = path.join(staticRoot, normalized.slice("static/".length))
            if (existsSync(candidate)) return candidate
          }
          const staticMatch = normalized.match(/^(\.\.\/)+static\/(.+)/)
          if (staticMatch) {
            const candidate = path.join(staticRoot, staticMatch[2])
            if (existsSync(candidate)) return candidate
          }
          if (normalized.startsWith("/")) {
            const candidate = path.join(contentRoot, normalized.slice(1))
            return existsSync(candidate) ? candidate : null
          }
          const relativeCandidate = path.join(fileDir, normalized)
          if (existsSync(relativeCandidate)) return relativeCandidate
          const rootCandidate = path.join(contentRoot, normalized)
          return existsSync(rootCandidate) ? rootCandidate : null
        }
        visit(root, "element", (node: Element) => {
          if (node.tagName !== "img" || !node.properties?.src) return
          const src = String(node.properties.src)
          if (replacements.has(src)) {
            node.properties.src = replacements.get(src)
            return
          }
          const sourcePath = resolveAttachmentPath(src)
          if (!sourcePath) return
          const existing = pending.get(src)
          if (existing) {
            existing.nodes.push(node)
            return
          }
          pending.set(src, { nodes: [node], sourcePath })
        })
        for (const [src, entry] of pending) {
          const ext = path.extname(entry.sourcePath).toLowerCase()
          const contentType = contentTypes[ext] ?? "application/octet-stream"
          const contentId = `${slug.replace(/[^a-z0-9]/gi, "-")}-${replacements.size + 1}`
          const cid = `cid:${contentId}`
          attachments.push({
            contentId,
            filename: path.basename(entry.sourcePath),
            contentType,
            content: (await fs.readFile(entry.sourcePath)).toString("base64"),
          })
          replacements.set(src, cid)
          for (const node of entry.nodes) {
            node.properties.src = cid
          }
        }
        if (baseUrl) {
          const basePath = slug === "index" ? "" : `${slug}/`
          const base = `https://${baseUrl}/${basePath}`
          visit(root, "element", (node: Element) => {
            if (!node.properties) return
            const src = node.properties.src
            const href = node.properties.href
            const normalize = (value: string) => {
              if (
                value.startsWith("http://") ||
                value.startsWith("https://") ||
                value.startsWith("mailto:") ||
                value.startsWith("data:") ||
                value.startsWith("cid:") ||
                value.startsWith("#") ||
                value.startsWith("//") ||
                value.startsWith("javascript:")
              ) {
                return value
              }
              return new URL(value, base).toString()
            }
            if (typeof src === "string") {
              node.properties.src = normalize(src)
            }
            if (typeof href === "string") {
              node.properties.href = normalize(href)
            }
          })
        }
        let html = toHtml(root)
        html = `<div dir="ltr">${html}</div>`

        const raw = await fs.readFile(filePath, "utf8")
        const newline = raw.includes("\r\n") ? "\r\n" : "\n"
        const normalized = raw.replace(/\r\n/g, "\n")
        const marker = "\n---\n"
        const endIndex = normalized.indexOf(marker, 4)
        const body = endIndex === -1 ? normalized : normalized.slice(endIndex + marker.length)
        let text = body
          .replace(/```[^\n]*\n([\s\S]*?)```/g, "$1")
          .replace(/\n{3,}/g, "\n\n")
          .trim()
        text = formatPlainText(text, slug as FullSlug, baseUrl)
        text = text.replace(/!\[[^\]]*\]\(([^)]+)\)/g, (_match, target) => {
          const cleaned = String(target).split(/[?#]/)[0] ?? ""
          return `[image: ${path.basename(cleaned)}]`
        })

        const endpoint =
          process.env.EMAIL_EMITTER_ENDPOINT ??
          `https://${ctx.cfg.configuration.baseUrl}/internal/email/emit`
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-email-secret": secret },
          body: JSON.stringify({ subject, text, html, recipients: recipientList, attachments }),
        })

        if (!response.ok) {
          const body = await response.text()
          throw new Error(`email send failed: ${response.status} ${body}`)
        }

        const frontmatterLines = normalized.slice(4, endIndex).split("\n")
        const index = frontmatterLines.findIndex((line) => line.trim().startsWith("emailSent:"))
        const match = frontmatterLines[index].match(/^(\s*emailSent:\s*).+$/)
        frontmatterLines[index] = match ? `${match[1]}true` : "emailSent: true"
        const updated = `---\n${frontmatterLines.join("\n")}\n---\n${normalized.slice(
          endIndex + marker.length,
        )}`
        await fs.writeFile(
          filePath,
          newline === "\n" ? updated : updated.replace(/\n/g, newline),
          "utf8",
        )
      }
      yield "" as FilePath
    },
  }
}
