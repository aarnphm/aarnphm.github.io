import { Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { existsSync, promises as fs } from 'node:fs'
import path from 'path'
import sharp from 'sharp'
import { visit } from 'unist-util-visit'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { contentDataFor } from '../../util/ctx'
import { FullSlug, QUARTZ } from '../../util/path'
import { getFontSpecificationName } from '../../util/theme'
import { QuartzPluginData } from '../vfile'
import {
  applySignatureStyles,
  attachSignatureImage,
  EmailAttachment,
  extractEmailRoot,
  formatPlainText,
  readEmailSignatureAsset,
  renderEmail,
  stripEmailNodes,
} from './email/document'
import {
  buildFontFaceCss,
  buildFontStack,
  DEFAULT_MONO,
  DEFAULT_SANS_SERIF,
  loadFontFaceEntries,
} from './email/fonts'
import { applyEmailStyles, buildEmailHtml } from './email/styles'

const name = 'EmailEmitter'
const emailsPath = path.join(QUARTZ, 'static', 'emails.txt')

const preferCompatibleImage = (
  sourcePath: string,
  fileDir: string,
  contentRoot: string,
  staticRoot: string,
): string => {
  const ext = path.extname(sourcePath).toLowerCase()
  if (ext !== '.webp' && ext !== '.avif') return sourcePath
  const baseName = path.basename(sourcePath, ext)
  const candidates = [path.dirname(sourcePath), fileDir, contentRoot, staticRoot]
  const preferredExts = ['.png', '.jpg', '.jpeg', '.gif']
  for (const dir of candidates) {
    for (const preferredExt of preferredExts) {
      const candidate = path.join(dir, `${baseName}${preferredExt}`)
      if (existsSync(candidate)) return candidate
    }
  }
  return sourcePath
}

export const EmailEmitter: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, content, resources) {
      if (process.env.EMAIL_EMITTER_ENABLED !== '1') return []

      const secret = process.env.EMAIL_EMITTER_SECRET
      if (!secret) throw new Error('missing EMAIL_EMITTER_SECRET')

      const recipients = new Set<string>()
      for (const line of (await fs.readFile(emailsPath, 'utf8')).split(/\r?\n/)) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith('#')) continue
        const match = trimmed.match(/<([^>]+)>/)
        const candidate = (match ? match[1] : trimmed).trim()
        if (!candidate || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(candidate)) continue
        recipients.add(candidate.toLowerCase())
      }
      const recipientList = [...recipients]
      if (recipientList.length === 0) return []

      const cfg = ctx.cfg.configuration
      const baseUrl = cfg.baseUrl
      const typography = cfg.theme.typography
      const fontFamilies = new Set<string>()
      if (typography.title) fontFamilies.add(getFontSpecificationName(typography.title))
      fontFamilies.add(getFontSpecificationName(typography.header))
      fontFamilies.add(getFontSpecificationName(typography.body))
      fontFamilies.add(getFontSpecificationName(typography.code))
      const fontEntries = cfg.theme.fontOrigin === 'local' ? await loadFontFaceEntries() : []
      const fontCss = buildFontFaceCss(fontEntries, baseUrl, fontFamilies)
      const bodyFont = buildFontStack(getFontSpecificationName(typography.body), DEFAULT_SANS_SERIF)
      const headerFont = buildFontStack(
        getFontSpecificationName(typography.header),
        DEFAULT_SANS_SERIF,
      )
      const codeFont = buildFontStack(getFontSpecificationName(typography.code), DEFAULT_MONO)
      const colors = cfg.theme.colors.lightMode

      const allFiles = contentDataFor(content)
      const contentRoot = ctx.argv.directory
      const staticRoot = path.join(QUARTZ, 'static')
      const contentTypes: Record<string, string> = {
        '.avif': 'image/avif',
        '.gif': 'image/gif',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp',
      }

      for (const [tree, file] of content) {
        const fileData: QuartzPluginData = file.data
        if (fileData.frontmatter?.email !== true || fileData.frontmatter?.emailSent !== false)
          continue

        const slug = fileData.slug!
        const relativePath = fileData.relativePath ?? fileData.filePath ?? ''
        const baseName = path.basename(relativePath, path.extname(relativePath))
        const [monthToken, yearToken] = baseName.split('-')
        const monthKey = (monthToken || '').slice(0, 3).toLowerCase()
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
          }[monthKey] ?? Number.parseInt(monthToken ?? '', 10) - 1
        const year = Number.parseInt(yearToken ?? '', 10)
        const emailSuffix = String(monthIndex + 1).padStart(3, '0')
        const subject = `GET updates/${year}/${emailSuffix}`

        const renderedRoot = await renderEmail(ctx, tree, fileData, allFiles, resources)
        const root = extractEmailRoot(renderedRoot)
        stripEmailNodes(root)
        applySignatureStyles(root, colors.dark)
        const attachments: EmailAttachment[] = []
        attachSignatureImage(root, slug, readEmailSignatureAsset(fileData), attachments)
        const replacements = new Map<string, string>()
        const pending = new Map<string, { nodes: Element[]; sourcePath: string }>()
        const filePath = fileData.filePath!
        const fileDir = path.dirname(filePath)
        const hasUrlScheme = (value: string) => /^[A-Za-z][A-Za-z0-9+.-]*:/.test(value)
        const resolveAttachmentPath = (src: string) => {
          const cleaned = decodeURIComponent(src.split(/[?#]/)[0] ?? '')
          if (!cleaned || cleaned.startsWith('//') || hasUrlScheme(cleaned)) {
            return null
          }
          const normalized = cleaned.replace(/^\.\//, '')
          if (normalized.startsWith('/static/')) {
            const candidate = path.join(QUARTZ, normalized.slice(1))
            return existsSync(candidate) ? candidate : null
          }
          if (normalized.startsWith('static/')) {
            const candidate = path.join(staticRoot, normalized.slice('static/'.length))
            if (existsSync(candidate)) return candidate
          }
          const staticMatch = normalized.match(/^(\.\.\/)+static\/(.+)/)
          if (staticMatch) {
            const candidate = path.join(staticRoot, staticMatch[2])
            if (existsSync(candidate)) return candidate
          }
          if (normalized.startsWith('/')) {
            const candidate = path.join(contentRoot, normalized.slice(1))
            return existsSync(candidate) ? candidate : null
          }
          const relativeCandidate = path.join(fileDir, normalized)
          if (existsSync(relativeCandidate)) return relativeCandidate
          const rootCandidate = path.join(contentRoot, normalized)
          return existsSync(rootCandidate) ? rootCandidate : null
        }
        visit(root, 'element', (node: Element) => {
          if (node.tagName !== 'img' || !node.properties?.src) return
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
          const preferredPath = preferCompatibleImage(
            entry.sourcePath,
            fileDir,
            contentRoot,
            staticRoot,
          )
          const ext = path.extname(preferredPath).toLowerCase()
          let contentType = contentTypes[ext] ?? 'application/octet-stream'
          let filename = path.basename(preferredPath)
          let contentBuffer: Buffer<ArrayBufferLike> = await fs.readFile(preferredPath)
          if (ext === '.webp' || ext === '.avif') {
            try {
              contentBuffer = await sharp(contentBuffer).png().toBuffer()
              contentType = 'image/png'
              filename = `${path.basename(preferredPath, ext)}.png`
            } catch {}
          }
          const contentId = `${slug.replace(/[^a-z0-9]/gi, '-')}-${replacements.size + 1}`
          const cid = `cid:${contentId}`
          attachments.push({
            contentId,
            filename,
            contentType,
            content: contentBuffer.toString('base64'),
          })
          replacements.set(src, cid)
          for (const node of entry.nodes) {
            node.properties.src = cid
          }
        }
        if (baseUrl) {
          const basePath = slug === 'index' ? '' : `${slug}/`
          const base = `https://${baseUrl}/${basePath}`
          visit(root, 'element', (node: Element) => {
            if (!node.properties) return
            const src = node.properties.src
            const href = node.properties.href
            const normalize = (value: string) => {
              const trimmed = value.trim()
              if (trimmed.startsWith('#') || trimmed.startsWith('//')) return trimmed
              try {
                const absolute = new URL(trimmed)
                const protocol = absolute.protocol.toLowerCase()
                if (
                  protocol === 'http:' ||
                  protocol === 'https:' ||
                  protocol === 'mailto:' ||
                  protocol === 'data:' ||
                  protocol === 'cid:'
                ) {
                  return absolute.toString()
                }
                return ''
              } catch {
                const resolved = new URL(trimmed, base)
                const protocol = resolved.protocol.toLowerCase()
                return protocol === 'javascript:' || protocol === 'vbscript:'
                  ? ''
                  : resolved.toString()
              }
            }
            if (typeof src === 'string') {
              node.properties.src = normalize(src)
            }
            if (typeof href === 'string') {
              node.properties.href = normalize(href)
            }
          })
        }
        applyEmailStyles(root, { headerFont, codeFont, colors })
        let html = toHtml(root, { allowDangerousHtml: true })
        html = buildEmailHtml(html, { fontCss, colors, bodyFont })

        const raw = await fs.readFile(filePath, 'utf8')
        const newline = raw.includes('\r\n') ? '\r\n' : '\n'
        const normalized = raw.replace(/\r\n/g, '\n')
        const marker = '\n---\n'
        const endIndex = normalized.indexOf(marker, 4)
        const body = endIndex === -1 ? normalized : normalized.slice(endIndex + marker.length)
        let text = body
          .replace(/```[^\n]*\n([\s\S]*?)```/g, '$1')
          .replace(/\n{3,}/g, '\n\n')
          .trim()
        text = formatPlainText(text, slug as FullSlug, baseUrl)
        text = text.replace(/!\[[^\]]*\]\(([^)]+)\)/g, (_match, target) => {
          const cleaned = String(target).split(/[?#]/)[0] ?? ''
          return `[image: ${path.basename(cleaned)}]`
        })

        const endpoint =
          process.env.EMAIL_EMITTER_ENDPOINT ?? `https://${baseUrl}/internal/email/emit`
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-email-secret': secret },
          body: JSON.stringify({ subject, text, html, recipients: recipientList, attachments }),
        })

        if (!response.ok) {
          const body = await response.text()
          throw new Error(`email send failed: ${response.status} ${body}`)
        }

        const frontmatterLines = normalized.slice(4, endIndex).split('\n')
        const index = frontmatterLines.findIndex(line => line.trim().startsWith('emailSent:'))
        const match = frontmatterLines[index].match(/^(\s*emailSent:\s*).+$/)
        frontmatterLines[index] = match ? `${match[1]}true` : 'emailSent: true'
        const updated = `---\n${frontmatterLines.join('\n')}\n---\n${normalized.slice(
          endIndex + marker.length,
        )}`
        await fs.writeFile(
          filePath,
          newline === '\n' ? updated : updated.replace(/\n/g, newline),
          'utf8',
        )
        yield filePath
      }
    },
  }
}
