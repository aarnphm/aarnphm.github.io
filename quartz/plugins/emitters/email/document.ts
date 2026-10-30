import { Element, Root, Node } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import path from 'path'
import { EXIT, visit } from 'unist-util-visit'
import { FullPageLayout } from '../../../cfg'
import { Content, Head } from '../../../components'
import { pageResources, renderPage } from '../../../components/renderPage'
import { QuartzComponent, QuartzComponentProps } from '../../../types/component'
import { BuildCtx } from '../../../util/ctx'
import { FilePath, FullSlug, getFileExtension, joinSegments, pathToRoot } from '../../../util/path'
import { StaticResources } from '../../../util/resources'
import { extractWikilinksWithPositions, resolveWikilinkTarget } from '../../../util/wikilinks'
import { QuartzPluginData } from '../../vfile'

const imageExtensions = new Set(['.avif', '.gif', '.jpg', '.jpeg', '.png', '.svg', '.webp'])
export type EmailAttachment = {
  contentId: string
  filename: string
  contentType: string
  content: string
}
export type EmailSignatureAsset = {
  contentType: string
  content: string
  width: number
  height: number
}
const EmptyFooter: QuartzComponent = () => null

export const renderEmail = async (
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

export const extractEmailRoot = (root: Root): Root => {
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

export const classList = (node: Element): string[] => {
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

export const stripEmailNodes = (root: Root) => {
  const parentMap = buildParentMap(root)
  const removals: Array<{ parent: Element | Root; index: number }> = []
  visit(root, 'element', (node: Element, index, parent) => {
    if (!parent || index == null) return
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

export const readEmailSignatureAsset = (fileData: QuartzPluginData): EmailSignatureAsset | null => {
  const value = fileData.emailSignature
  if (!value || typeof value !== 'object') return null
  const candidate = value as Record<string, unknown>
  if (typeof candidate.contentType !== 'string' || candidate.contentType.length === 0) return null
  if (typeof candidate.content !== 'string' || candidate.content.length === 0) return null
  if (typeof candidate.width !== 'number' || !Number.isFinite(candidate.width)) return null
  if (typeof candidate.height !== 'number' || !Number.isFinite(candidate.height)) return null
  return {
    contentType: candidate.contentType,
    content: candidate.content,
    width: Math.max(1, Math.ceil(candidate.width)),
    height: Math.max(1, Math.ceil(candidate.height)),
  }
}

export const attachSignatureImage = (
  root: Root,
  slug: string,
  signature: EmailSignatureAsset | null,
  attachments: EmailAttachment[],
) => {
  if (!signature) return
  visit(root, 'element', (node: Element, index, parent) => {
    if (!parent || index == null) return
    if (node.tagName !== 'p' || !classList(node).includes('signature')) return
    const contentId = `${slug.replace(/[^a-z0-9]/gi, '-')}-signature-${attachments.length + 1}`
    attachments.push({
      contentId,
      filename: 'signature.png',
      contentType: signature.contentType,
      content: signature.content,
    })
    const imgNode: Element = {
      type: 'element',
      tagName: 'img',
      properties: {
        src: `cid:${contentId}`,
        alt: 'signature',
        width: String(signature.width),
        height: String(signature.height),
        style: 'display:block;margin:0 0 18px auto;max-width:100%;height:auto;',
      },
      children: [],
    }
    const container = parent as Element | Root
    container.children.splice(index, 1, imgNode)
    return EXIT
  })
}

export const applySignatureStyles = (root: Root, stroke: string) => {
  visit(root, 'element', (node: Element) => {
    if (node.tagName !== 'path') return
    node.properties = node.properties ?? {}
    node.properties.stroke = node.properties.stroke ?? stroke
    node.properties['stroke-width'] = node.properties['stroke-width'] ?? '1'
    node.properties['stroke-linecap'] = node.properties['stroke-linecap'] ?? 'round'
    node.properties['stroke-linejoin'] = node.properties['stroke-linejoin'] ?? 'round'
  })
}

export const formatPlainText = (body: string, slug: FullSlug, baseUrl?: string): string => {
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
