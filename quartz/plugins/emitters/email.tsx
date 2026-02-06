import { Element, Root, Node } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { existsSync, promises as fs } from 'node:fs'
import path from 'path'
import sharp from 'sharp'
import { EXIT, visit } from 'unist-util-visit'
import { FullPageLayout } from '../../cfg'
import { Content, Head } from '../../components'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponent, QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import {
  FilePath,
  FullSlug,
  getFileExtension,
  joinSegments,
  pathToRoot,
  QUARTZ,
} from '../../util/path'
import { StaticResources } from '../../util/resources'
import { ColorScheme, getFontSpecificationName } from '../../util/theme'
import { extractWikilinksWithPositions, resolveWikilinkTarget } from '../../util/wikilinks'
import { QuartzPluginData } from '../vfile'

const name = 'EmailEmitter'
const emailsPath = path.join(QUARTZ, 'static', 'emails.txt')
const imageExtensions = new Set(['.avif', '.gif', '.jpg', '.jpeg', '.png', '.svg', '.webp'])
const EmptyFooter: QuartzComponent = () => null
const DEFAULT_SANS_SERIF =
  'system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"'
const DEFAULT_MONO = 'ui-monospace, SFMono-Regular, SF Mono, Menlo, monospace'

type FontFaceEntry = {
  family: string
  filename: string
  format: string
  style: string
  weight: string
  path: string
}

const normalizeFontFormat = (format: string): string => {
  const normalized = format.toLowerCase()
  if (normalized === 'ttf') return 'truetype'
  if (normalized === 'otf') return 'opentype'
  return normalized
}

const splitFontFaceArgs = (value: string): string[] => {
  const args: string[] = []
  let current = ''
  let inString = false
  let quote = ''
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index]
    if (inString) {
      current += char
      if (char === quote && value[index - 1] !== '\\') {
        inString = false
        quote = ''
      }
      continue
    }
    if (char === '"' || char === "'") {
      inString = true
      quote = char
      current += char
      continue
    }
    if (char === ',') {
      args.push(current.trim())
      current = ''
      continue
    }
    current += char
  }
  const trimmed = current.trim()
  if (trimmed.length > 0) args.push(trimmed)
  return args
}

const unquote = (value: string): string => value.replace(/^["']|["']$/g, '')

const parseFontFaceIncludes = (scss: string): FontFaceEntry[] => {
  const entries: FontFaceEntry[] = []
  const regex = /@include font-face\(([\s\S]*?)\);/g
  let match = regex.exec(scss)
  while (match) {
    const args = splitFontFaceArgs(match[1] ?? '')
    if (args.length >= 3) {
      const family = unquote(args[0] ?? '')
      const filename = unquote(args[1] ?? '')
      const format = unquote(args[2] ?? '')
      const style = unquote(args[3] ?? 'normal')
      const weight = unquote(args[4] ?? 'normal')
      const pathArg = unquote(args[5] ?? '/fonts/')
      if (family && filename && format) {
        entries.push({ family, filename, format, style, weight, path: pathArg })
      }
    }
    match = regex.exec(scss)
  }
  return entries
}

const normalizeFontPath = (value: string): string => {
  let normalized = value.trim()
  if (!normalized.startsWith('/')) normalized = `/${normalized}`
  if (!normalized.endsWith('/')) normalized = `${normalized}/`
  return normalized
}

const buildFontFaceCss = (
  entries: FontFaceEntry[],
  baseUrl: string | undefined,
  families: Set<string>,
): string => {
  const base = baseUrl ? (baseUrl.startsWith('http') ? baseUrl : `https://${baseUrl}`) : ''
  return entries
    .filter(entry => families.has(entry.family))
    .map(entry => {
      const pathPrefix = normalizeFontPath(entry.path)
      const url = base ? `${base}${pathPrefix}${entry.filename}` : `${pathPrefix}${entry.filename}`
      const format = normalizeFontFormat(entry.format)
      return `@font-face{font-family:"${entry.family}";font-style:${entry.style};font-weight:${entry.weight};font-display:swap;src:url("${url}") format("${format}")}`
    })
    .join('')
}

const buildFontStack = (fontName: string, fallback: string): string => `"${fontName}", ${fallback}`

const loadFontFaceEntries = async (): Promise<FontFaceEntry[]> => {
  const fontsPath = path.join(QUARTZ, 'styles', 'fonts.scss')
  try {
    const scss = await fs.readFile(fontsPath, 'utf8')
    return parseFontFaceIncludes(scss)
  } catch {
    return []
  }
}

const renderEmail = async (
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
): Promise<Root> => {
  const slug = fileData.slug!
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const emailFileData: QuartzPluginData = {
    ...fileData,
    frontmatter: fileData.frontmatter ? { ...fileData.frontmatter } : undefined,
  }
  if (emailFileData.frontmatter?.protected === true) {
    emailFileData.frontmatter.protected = false
  }
  delete emailFileData.protectedPassword

  const componentData: QuartzComponentProps = {
    ctx,
    fileData: emailFileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }
  const emailLayout: FullPageLayout = {
    head: Head(),
    header: [],
    beforeBody: [],
    pageBody: Content(),
    afterBody: [],
    sidebar: [],
    footer: EmptyFooter,
  }
  const rendered = renderPage(
    ctx,
    slug,
    componentData,
    emailLayout,
    externalResources,
    false,
    false,
    { forEmail: true },
  )
  const doc = fromHtml(rendered)
  let body: Element | undefined
  visit(doc, { tagName: 'body' }, (node: Element) => {
    body = node
    return EXIT
  })
  return body ? ({ type: 'root', children: body.children } as Root) : doc
}

const extractEmailRoot = (root: Root): Root => {
  let article: Element | undefined
  const footnotes: Element[] = []
  visit(root, 'element', (node: Element) => {
    if (!article && node.tagName === 'article') {
      article = node
    }
    if (node.properties?.dataFootnotes === '') {
      footnotes.push(node)
    }
  })
  if (!article && footnotes.length === 0) return root
  const children: Root['children'] = []
  if (article) children.push(article)
  children.push(...footnotes)
  return { type: 'root', children }
}

const classList = (node: Element): string[] => {
  const classes = node.properties?.className
  return Array.isArray(classes) ? classes.map(String) : classes ? [String(classes)] : []
}

const buildParentMap = (root: Root): Map<Node, Element | Root> => {
  const parentMap = new Map<Node, Element | Root>()
  visit(root, (_node, _index, parent) => {
    if (parent) parentMap.set(_node, parent as Element | Root)
  })
  return parentMap
}

const hasAncestorWithClass = (
  node: Node,
  className: string,
  parentMap: Map<Node, Element | Root>,
): boolean => {
  let current: Element | Root | undefined = parentMap.get(node)
  while (current && current.type === 'element') {
    if (classList(current).includes(className)) return true
    current = parentMap.get(current)
  }
  return false
}

const stripEmailNodes = (root: Root) => {
  const parentMap = buildParentMap(root)
  const removals: Array<{ parent: Element | Root; index: number }> = []
  visit(root, 'element', (node: Element, index, parent) => {
    if (!parent || index === null) return
    if (node.tagName === 'script' || node.tagName === 'style') {
      removals.push({ parent: parent as Element | Root, index })
      return
    }
    if (node.tagName === 'span' && classList(node).includes('indicator-hook')) {
      removals.push({ parent: parent as Element | Root, index })
      return
    }
    if (node.tagName === 'svg' && !hasAncestorWithClass(node, 'signature', parentMap)) {
      removals.push({ parent: parent as Element | Root, index })
    }
  })
  removals
    .sort((a, b) => b.index - a.index)
    .forEach(({ parent, index }) => parent.children.splice(index, 1))
}

const applySignatureStyles = (root: Root, stroke: string) => {
  visit(root, 'element', (node: Element) => {
    if (node.tagName !== 'path') return
    node.properties = node.properties ?? {}
    node.properties.stroke = node.properties.stroke ?? stroke
    node.properties['stroke-width'] = node.properties['stroke-width'] ?? '1'
    node.properties['stroke-linecap'] = node.properties['stroke-linecap'] ?? 'round'
    node.properties['stroke-linejoin'] = node.properties['stroke-linejoin'] ?? 'round'
  })
}

const formatPlainText = (body: string, slug: FullSlug, baseUrl?: string): string => {
  const ranges = extractWikilinksWithPositions(body)
  if (ranges.length === 0) return body
  let result = ''
  let lastIndex = 0
  for (const range of ranges) {
    if (range.start > lastIndex) {
      result += body.slice(lastIndex, range.start)
    }
    const link = range.wikilink
    const raw = link.raw ?? body.slice(range.start, range.end)
    const target = link.target ?? ''
    const ext = getFileExtension(target as FilePath) ?? ''
    const isImage = link.embed && ext.length > 0 && imageExtensions.has(ext)
    if (isImage) {
      const cleaned = target.split(/[?#]/)[0] ?? ''
      const filename = cleaned ? path.basename(cleaned) : 'image'
      result += `[image: ${filename}]`
    } else {
      const resolved = resolveWikilinkTarget(link, slug)
      if (!resolved) {
        result += link.alias ?? link.anchorText ?? link.target ?? raw
      } else {
        const base = baseUrl ? `https://${baseUrl}` : ''
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

const applyEmailStyles = (
  root: Root,
  options: { headerFont: string; codeFont: string; colors: ColorScheme },
) => {
  const { headerFont, codeFont, colors } = options
  const codeBackground = colors.highlight || colors.lightgray
  const mergeStyle = (node: Element, style: string) => {
    const current = typeof node.properties?.style === 'string' ? node.properties.style : ''
    node.properties = node.properties ?? {}
    node.properties.style = current ? `${current}; ${style}` : style
  }
  visit(root, 'element', (node: Element, _index, parent) => {
    const dataCodeblock = node.properties?.dataCodeblock
    const classes = classList(node)
    const isSignature = classes.includes('signature')
    const isSms = dataCodeblock === 'sms' || classes.includes('text')
    if (node.tagName === 'p' || node.tagName === 'div') {
      if (isSms) {
        mergeStyle(
          node,
          [
            'padding-top: 24px',
            'padding-bottom: 24px',
            'padding-left: 16px',
            'font-style: italic',
            'border: 0',
            `border-top: 1px solid ${colors.lightgray}`,
            `border-bottom: 1px solid ${colors.lightgray}`,
          ].join('; '),
        )
      } else if (typeof dataCodeblock === 'string') {
        mergeStyle(
          node,
          [
            `background: ${codeBackground}`,
            `border: 1px solid ${colors.lightgray}`,
            'border-radius: 6px',
            'padding: 12px 14px',
            'margin: 0 0 18px 0',
            `font-family: ${codeFont}`,
            'font-size: 13px',
            'line-height: 1.6',
            'white-space: pre-wrap',
          ].join('; '),
        )
      }
    }
    if (node.tagName === 'p' && isSignature) {
      mergeStyle(
        node,
        [
          'display: flex',
          'flex-wrap: wrap',
          'justify-content: flex-end',
          'min-height: 51px',
          'margin: 0 0 18px 0',
        ].join('; '),
      )
    }
    switch (node.tagName) {
      case 'article':
        mergeStyle(node, 'margin: 0;')
        break
      case 'section':
        mergeStyle(node, 'margin: 0;')
        break
      case 'p':
        mergeStyle(node, 'margin: 0 0 18px 0;')
        break
      case 'h1':
        mergeStyle(
          node,
          `margin: 0 0 18px 0; font-size: 30px; line-height: 1.25; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'h2':
        mergeStyle(
          node,
          `margin: 28px 0 14px 0; font-size: 22px; line-height: 1.3; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'h3':
        mergeStyle(
          node,
          `margin: 22px 0 12px 0; font-size: 19px; line-height: 1.35; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'h4':
        mergeStyle(
          node,
          `margin: 18px 0 10px 0; font-size: 17px; line-height: 1.35; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'ul':
      case 'ol':
        mergeStyle(node, 'margin: 0 0 18px 0; padding: 0 0 0 24px;')
        break
      case 'li':
        mergeStyle(node, 'margin: 6px 0;')
        break
      case 'blockquote':
        mergeStyle(
          node,
          `margin: 22px 0; padding: 0 0 0 16px; border-left: 3px solid ${colors.lightgray}; color: ${colors.darkgray}; font-style: italic;`,
        )
        break
      case 'hr':
        mergeStyle(node, `border: 0; border-top: 1px solid ${colors.lightgray}; margin: 28px 0;`)
        break
      case 'img':
        mergeStyle(
          node,
          'max-width: 100%; height: auto; display: block; margin: 24px auto; border-radius: 6px;',
        )
        break
      case 'figure':
        mergeStyle(node, 'margin: 24px 0;')
        break
      case 'figcaption':
        mergeStyle(
          node,
          `margin-top: 10px; font-size: 14px; color: ${colors.darkgray}; text-align: center;`,
        )
        break
      case 'pre':
        mergeStyle(
          node,
          [
            `background: ${codeBackground}`,
            `border: 1px solid ${colors.lightgray}`,
            'border-radius: 6px',
            'padding: 12px 14px',
            'margin: 0 0 18px 0',
            'overflow-x: auto',
            'white-space: pre-wrap',
            `font-family: ${codeFont}`,
            'font-size: 13px',
            'line-height: 1.6',
          ].join('; '),
        )
        break
      case 'code': {
        const isBlock = parent?.type === 'element' && (parent as Element).tagName === 'pre'
        if (isBlock) {
          mergeStyle(node, 'background: transparent; padding: 0;')
        } else {
          mergeStyle(
            node,
            [
              `background: ${codeBackground}`,
              `border: 1px solid ${colors.lightgray}`,
              'border-radius: 4px',
              'padding: 0 3px',
              `font-family: ${codeFont}`,
              'font-size: 0.95em',
            ].join('; '),
          )
        }
        break
      }
      case 'a':
        mergeStyle(node, `color: ${colors.dark}; text-decoration: underline;`)
        break
      case 'table':
        mergeStyle(node, 'width: 100%; border-collapse: collapse; margin: 24px 0;')
        break
      case 'thead':
        mergeStyle(node, `background: ${colors.light};`)
        break
      case 'th':
        mergeStyle(
          node,
          `text-align: left; border-bottom: 1px solid ${colors.lightgray}; padding: 8px 6px;`,
        )
        break
      case 'td':
        mergeStyle(node, `border-bottom: 1px solid ${colors.lightgray}; padding: 8px 6px;`)
        break
      case 'sup':
        mergeStyle(node, 'font-size: 0.75em; line-height: 0;')
        break
      case 'small':
        mergeStyle(node, `font-size: 14px; color: ${colors.darkgray};`)
        break
      default:
        break
    }
  })
}

const buildEmailHtml = (
  content: string,
  options: { fontCss: string; colors: ColorScheme; bodyFont: string },
): string => {
  const { fontCss, colors, bodyFont } = options
  const bodyStyle = [
    'margin: 0',
    'padding: 0',
    `background: ${colors.light}`,
    `color: ${colors.dark}`,
    '-webkit-text-size-adjust: 100%',
  ].join('; ')
  const outerTableStyle = ['width: 100%', `background: ${colors.light}`].join('; ')
  const containerStyle = [
    'width: 100%',
    'max-width: 640px',
    'background: #ffffff',
    `border: 1px solid ${colors.lightgray}`,
    'border-radius: 12px',
    'overflow: hidden',
  ].join('; ')
  const innerStyle = [
    'padding: 32px 36px 40px',
    `font-family: ${bodyFont}`,
    'font-size: 17px',
    'line-height: 1.75',
    `color: ${colors.dark}`,
    'text-align: left',
  ].join('; ')
  const headParts = [
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
    fontCss ? `<style>${fontCss}</style>` : '',
  ].filter(Boolean)
  return [
    '<!doctype html>',
    '<html>',
    '<head>',
    ...headParts,
    '</head>',
    `<body dir="ltr" style="${bodyStyle}">`,
    `<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="${outerTableStyle}">`,
    '<tr>',
    '<td align="center" style="padding: 24px 12px;">',
    `<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="${containerStyle}">`,
    '<tr>',
    `<td style="${innerStyle}">`,
    content,
    '</td>',
    '</tr>',
    '</table>',
    '</td>',
    '</tr>',
    '</table>',
    '</body>',
    '</html>',
  ].join('')
}

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
      if (ctx.argv.watch && !ctx.argv.force) return []
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

      const allFiles = content.map(c => c[1].data)
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
          const cleaned = decodeURIComponent(src.split(/[?#]/)[0] ?? '')
          if (
            !cleaned ||
            /^([a-z]+:)?\/\//i.test(cleaned) ||
            cleaned.startsWith('data:') ||
            cleaned.startsWith('cid:') ||
            cleaned.startsWith('mailto:')
          ) {
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
          let contentBuffer = await fs.readFile(preferredPath)
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
              if (
                value.startsWith('http://') ||
                value.startsWith('https://') ||
                value.startsWith('mailto:') ||
                value.startsWith('data:') ||
                value.startsWith('cid:') ||
                value.startsWith('#') ||
                value.startsWith('//') ||
                value.startsWith('javascript:')
              ) {
                return value
              }
              return new URL(value, base).toString()
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
