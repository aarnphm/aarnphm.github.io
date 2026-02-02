import { toString } from "hast-util-to-string"
import { existsSync, promises as fs } from "node:fs"
import path from "path"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { FullPageLayout } from "../../cfg"
import { Content } from "../../components"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { FilePath, pathToRoot, QUARTZ } from "../../util/path"

const name = "EmailEmitter"
const emailsPath = path.join(QUARTZ, "static", "emails.txt")

export const EmailEmitter: QuartzEmitterPlugin = () => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: Content(),
  }

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

      const allFiles = content.map((item) => item[1].data)
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
        if (file.data.frontmatter?.email !== true || file.data.frontmatter?.emailSent !== false) continue

        const slug = file.data.slug!
        const relativePath = file.data.relativePath ?? file.data.filePath ?? ""
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
        const url =
          baseUrl && slug === "index"
            ? `https://${baseUrl}/`
            : baseUrl
              ? `https://${baseUrl}/${slug}`
              : slug

        const externalResources = pageResources(pathToRoot(slug), resources, ctx)
        const bannerHtml =
          `<div style="border:1px solid #e0e0e0;padding:12px;margin-bottom:16px;font-size:14px;">` +
          `<p style="margin:0;"><a href="${url}">rendered</a></p>` +
          `</div>`
        let html = renderPage(
          ctx,
          slug,
          {
            ctx,
            fileData: file.data,
            externalResources,
            cfg: ctx.cfg.configuration,
            children: [],
            tree,
            allFiles,
          },
          opts,
          externalResources,
          false,
        )
        html = html.includes("<body")
          ? html.replace(/<body([^>]*)>/i, `<body$1>${bannerHtml}`)
          : bannerHtml + html
        const attachments: {
          contentId: string
          filename: string
          contentType: string
          content: string
        }[] = []
        const replacements = new Map<string, string>()
        const imgPattern = /<img\b[^>]*\bsrc=(["'])(.*?)\1/gi
        const filePath = file.data.filePath!
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
          if (cleaned.startsWith("/static/")) {
            const candidate = path.join(QUARTZ, cleaned.slice(1))
            return existsSync(candidate) ? candidate : null
          }
          if (cleaned.startsWith("/")) {
            const candidate = path.join(contentRoot, cleaned.slice(1))
            return existsSync(candidate) ? candidate : null
          }
          if (cleaned.startsWith("static/")) {
            const candidate = path.join(staticRoot, cleaned.slice("static/".length))
            if (existsSync(candidate)) return candidate
          }
          const relativeCandidate = path.join(fileDir, cleaned)
          if (existsSync(relativeCandidate)) return relativeCandidate
          const rootCandidate = path.join(contentRoot, cleaned)
          return existsSync(rootCandidate) ? rootCandidate : null
        }
        for (const match of html.matchAll(imgPattern)) {
          const src = match[2]?.trim()
          if (!src || replacements.has(src)) continue
          const sourcePath = resolveAttachmentPath(src)
          if (!sourcePath) continue
          const ext = path.extname(sourcePath).toLowerCase()
          const contentType = contentTypes[ext] ?? "application/octet-stream"
          const contentId = `${slug.replace(/[^a-z0-9]/gi, "-")}-${replacements.size + 1}`
          attachments.push({
            contentId,
            filename: path.basename(sourcePath),
            contentType,
            content: (await fs.readFile(sourcePath)).toString("base64"),
          })
          replacements.set(src, `cid:${contentId}`)
        }
        for (const [src, cid] of replacements) {
          const escaped = src.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
          html = html.replace(
            new RegExp(`(<img\\b[^>]*\\bsrc=)(["'])${escaped}\\2`, "gi"),
            `$1"$cid"`,
          )
        }
        if (baseUrl) {
          html = html.replace(/(src|href)="\/(?!\/)/g, `$1="https://${baseUrl}/`)
        }

        const contentText = toString(tree)
          .replace(/\n{3,}/g, "\n\n")
          .trim()
        const text = [
          "----------------------------------------",
          `rendered: ${url}`,
          "----------------------------------------",
          contentText
        ]
          .join("\n")
          .trim()

        const response = await fetch(
          process.env.EMAIL_EMITTER_ENDPOINT ??
            `https://${ctx.cfg.configuration.baseUrl}/internal/email/emit`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json", "x-email-secret": secret },
            body: JSON.stringify({
              subject,
              text,
              html,
              recipients,
              attachments,
            }),
          },
        )

        if (!response.ok) {
          const body = await response.text()
          throw new Error(`email send failed: ${response.status} ${body}`)
        }

        const raw = await fs.readFile(filePath, "utf8")
        const newline = raw.includes("\r\n") ? "\r\n" : "\n"
        const normalized = raw.replace(/\r\n/g, "\n")
        const marker = "\n---\n"
        const endIndex = normalized.indexOf(marker, 4)
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
