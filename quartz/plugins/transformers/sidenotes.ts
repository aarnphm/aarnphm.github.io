import type { Node } from 'unist'
import { Element as HastElement, Text as HastText, ElementContent, Root as HastRoot } from 'hast'
import { h } from 'hastscript'
import isAbsoluteUrl from 'is-absolute-url'
import { Root as MdastRoot } from 'mdast'
import path from 'path'
import { visit } from 'unist-util-visit'
import { VFile } from 'vfile'
import type {
  Sidenote,
  SidenoteReference,
} from '../../extensions/micromark-extension-ofm-sidenotes'
// @ts-ignore
import script from '../../components/scripts/sidenotes.inline'
import content from '../../components/styles/sidenotes.inline.scss'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { FullSlug, transformLink } from '../../util/path'
import { extractWikilinks, resolveWikilinkTarget } from '../../util/wikilinks'

const isSidenoteNode = (node: Node): node is Sidenote | SidenoteReference =>
  node.type === 'sidenote' || node.type === 'sidenoteReference'

const isHastElement = (node: Node): node is HastElement => node.type === 'element'

const readString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined

const readNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

const readBoolean = (value: unknown): boolean | undefined =>
  typeof value === 'boolean' ? value : undefined

const readStringArray = (value: unknown): string[] | undefined =>
  Array.isArray(value) && value.every(item => typeof item === 'string') ? value : undefined

const getClassNameList = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.filter((entry): entry is string => typeof entry === 'string')
  }
  if (typeof value === 'string') return [value]
  return []
}

const findChildByClass = (
  children: ElementContent[],
  className: string,
): HastElement | undefined => {
  for (const child of children) {
    if (!isHastElement(child)) continue
    const classList = getClassNameList(child.properties?.className)
    if (classList.includes(className)) return child
  }
  return undefined
}

export const Sidenotes: QuartzTransformerPlugin = () => {
  return {
    name: 'Sidenotes',
    externalResources() {
      return {
        js: [{ script, contentType: 'inline', loadTime: 'afterDOMReady' }],
        css: [{ content, spaPreserve: true, inline: true }],
      }
    },
    markdownPlugins() {
      return [
        () => (tree: MdastRoot, file: VFile) => {
          let counter = 0

          visit(tree, (node: Node) => {
            if (!isSidenoteNode(node)) return

            const sidenoteId = ++counter
            const baseId = buildSidenoteDomId(file, sidenoteId)

            if (node.type === 'sidenote') {
              const parsed = node.data?.sidenoteParsed
              if (!parsed) return

              const props = parsed.properties || {}
              const forceInline = props.dropdown === 'true' || props.inline === 'true'
              const allowLeft = props.left !== 'false'
              const allowRight = props.right !== 'false'

              node.data = {
                ...node.data,
                sidenoteId,
                baseId,
                forceInline,
                allowLeft,
                allowRight,
                internal: props.internal,
                label: parsed.label || '',
              }
            } else {
              node.data = { ...node.data, sidenoteId, baseId }
            }
          })
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => {
          return (tree: HastRoot, file: VFile) => {
            const definitions = new Map<string, ElementContent[]>()

            visit(tree, (node, index, parent) => {
              if (!isHastElement(node)) return

              const dataType = readString(node.properties?.dataType)
              if (dataType !== 'sidenote-def') return

              const label = readString(node.properties?.label)
              const contentDiv = node.children[1]
              if (label && contentDiv && isHastElement(contentDiv)) {
                definitions.set(label, contentDiv.children)
              }

              if (parent && 'children' in parent && Array.isArray(parent.children)) {
                if (typeof index === 'number') {
                  parent.children.splice(index, 1)
                  return index
                }
              }
            })

            visit(tree, (node, index, parent) => {
              if (!isHastElement(node)) return
              if (!parent || typeof index !== 'number') return

              const dataType = readString(node.properties?.dataType)
              if (dataType !== 'sidenote' && dataType !== 'sidenote-ref') return

              const props = node.properties ?? {}
              const labelRaw = readString(props.label) ?? ''
              const labelContainer = findChildByClass(node.children, 'sidenote-label-hast')
              const contentContainer = findChildByClass(node.children, 'sidenote-content-hast')

              const labelHast = labelContainer?.children ?? []
              let contentHast: ElementContent[] = []

              if (dataType === 'sidenote-ref') {
                const defContent = definitions.get(labelRaw)
                if (defContent) {
                  contentHast = defContent
                } else {
                  contentHast = [{ type: 'text', value: '[Missing definition]' }]
                }
              } else {
                contentHast = contentContainer?.children ?? []
              }

              const internalLinks =
                readStringArray(props.internal) ??
                (typeof props.internal === 'string' ? [props.internal] : undefined)
              const internal = renderInternalLinks(internalLinks, file, ctx)
              const combinedContent = [...contentHast, ...internal]

              const finalLabel = labelHast.length > 0 ? labelHast : deriveLabel(labelRaw)
              const resolvedSidenoteId = readNumber(props.sidenoteId) ?? 0
              const resolvedBaseId =
                readString(props.baseId) ?? buildSidenoteDomId(file, resolvedSidenoteId)
              const forceInline = readBoolean(props.forceInline) ?? false
              const allowLeft = readBoolean(props.allowLeft)
              const allowRight = readBoolean(props.allowRight)

              if ('children' in parent && Array.isArray(parent.children)) {
                parent.children.splice(
                  index,
                  1,
                  ...buildSidenoteHast(
                    finalLabel,
                    forceInline === true,
                    allowLeft !== false,
                    allowRight !== false,
                    combinedContent,
                    resolvedSidenoteId,
                    resolvedBaseId,
                  ),
                )
                return index
              }
            })
          }
        },
      ]
    },
  }
}

function renderInternalLinks(
  wikilinks: string[] | undefined,
  file: VFile,
  ctx: BuildCtx,
): ElementContent[] {
  if (!wikilinks || wikilinks.length === 0) return []

  const links: HastElement[] = []

  for (const wl of wikilinks) {
    for (const parsed of extractWikilinks(wl)) {
      const resolved = resolveWikilinkTarget(parsed, file.data.slug as FullSlug)
      const anchor = parsed.anchor ?? ''
      let destination = parsed.target + anchor
      let dataSlug: string | undefined

      if (resolved) {
        dataSlug = resolved.slug
        const destWithAnchor = `${resolved.slug}${anchor}`
        destination = transformLink(file.data.slug as FullSlug, destWithAnchor, {
          allSlugs: ctx.allSlugs,
          strategy: 'absolute',
        })
      } else if (!isAbsoluteUrl(destination, { httpOnly: false })) {
        destination = transformLink(file.data.slug as FullSlug, destination, {
          allSlugs: ctx.allSlugs,
          strategy: 'absolute',
        })
      }

      const display =
        parsed.alias ||
        path.basename(parsed.target, path.extname(parsed.target)) ||
        parsed.target ||
        'link'

      const link = h(
        'a',
        {
          href: destination,
          className: ['internal'],
          ...(dataSlug ? { 'data-slug': dataSlug } : {}),
        },
        [h('span.indicator-hook'), display],
      )

      links.push(link)
    }
  }

  if (links.length === 0) return []

  const separator = h('span.sidenote-separator', { role: 'presentation' })

  const interleaved: ElementContent[] = []
  links.forEach((link, idx) => {
    if (idx > 0) {
      const comma: HastText = { type: 'text', value: ', ' }
      interleaved.push(comma)
    }
    interleaved.push(link)
  })

  const container = h('span.sidenote-linked-notes', ['linked notes: ', ...interleaved])

  return [separator, container]
}

function deriveLabel(rawLabel: string): ElementContent[] {
  let labelText = rawLabel.trim()
  if (!labelText) return []

  const parsed = extractWikilinks(labelText)[0]
  if (parsed) {
    labelText =
      parsed.alias || path.basename(parsed.target, path.extname(parsed.target)) || parsed.target
  }

  return [{ type: 'text', value: labelText }]
}

function buildSidenoteHast(
  label: ElementContent[],
  forceInline: boolean,
  allowLeft: boolean,
  allowRight: boolean,
  combinedContent: ElementContent[],
  sidenoteId: number,
  baseId: string,
): HastElement[] {
  const arrowDownSvg = h(
    'svg.sidenote-arrow.sidenote-arrow-down',
    {
      width: '8',
      height: '5',
      viewBox: '0 0 8 5',
      xmlns: 'http://www.w3.org/2000/svg',
      'aria-hidden': 'true',
    },
    [h('path', { d: 'M0 0L8 0L4 5Z', fill: 'currentColor' })],
  )

  const hasLabel = label.length > 0

  const labelProps: Record<string, string> = {
    id: `${baseId}-label`,
    'aria-controls': `${baseId}-content`,
  }

  const labelElement = h(
    'span.sidenote-label',
    hasLabel ? labelProps : { ...labelProps, 'data-auto': '' },
    hasLabel ? [...label, arrowDownSvg] : [{ type: 'text', value: '▪' }, arrowDownSvg],
  )

  const dataAttrs: Record<string, string> = { id: baseId, 'data-sidenote-id': String(sidenoteId) }

  if (forceInline) dataAttrs['data-force-inline'] = 'true'
  if (!allowLeft) dataAttrs['data-allow-left'] = 'false'
  if (!allowRight) dataAttrs['data-allow-right'] = 'false'

  const sidenoteElement = h('span.sidenote', dataAttrs, [labelElement])

  const contentElement = h(
    'span.sidenote-content',
    {
      id: `${baseId}-content`,
      'data-sidenote-id': String(sidenoteId),
      'data-sidenote-for': baseId,
      'aria-hidden': 'true',
    },
    combinedContent.length > 0 ? combinedContent : [{ type: 'text', value: '' }],
  )

  return [sidenoteElement, contentElement]
}

function buildSidenoteDomId(file: VFile, sidenoteId: number): string {
  const rawSlug = (file.data.slug as string | undefined) ?? 'note'
  const sanitized = rawSlug.replace(/[^A-Za-z0-9_-]/g, '-')
  return `sidenote-${sanitized}-${sidenoteId}`
}
