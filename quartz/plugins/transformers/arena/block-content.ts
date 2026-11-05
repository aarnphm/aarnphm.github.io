import type { Element, ElementContent } from 'hast'
import type { VFile } from 'vfile'
import { toString } from 'hast-util-to-string'
import type { BuildCtx } from '../../../util/ctx'
import type { ArenaBlock } from '../arena'
import { splitAnchor, stripSlashes } from '../../../util/path'
import { referenceSourcePath } from '../../../util/references/context'
import { labelToHast, parseReferenceLabel } from '../../../util/references/label'
import { extractLiteralReferences, referenceLabelSource } from '../../../util/references/literal'
import { recordReferenceDependency } from '../../../util/references/markdown'
import { referenceRequest } from '../../../util/references/request'
import { missingReferenceRoute } from '../../../util/references/route'
import { externalLinkRegex } from '../ofm'

const isElement = (node: ElementContent): node is Element => node.type === 'element'

const TRAILING_MARKERS_PATTERN = /(?:\s*\[(?:\*\*|--|—)\])+\s*$/
const HIGHLIGHT_MARKER = /\[\*\*\]/
const EMBED_DISABLED_MARKER = /\[(?:--|—)\]/
export const parseLinkTitle = (text: string): { url: string; title?: string } | undefined => {
  const match = text.match(/^(https?:\/\/[^\s]+)\s*(?:(?:--|—)\s*(.+))?$/)
  if (!match) {
    return undefined
  }

  return { url: match[1], title: match[2]?.trim() }
}

export const stripTrailingMarkers = (value: string): string =>
  value.replace(TRAILING_MARKERS_PATTERN, '').trim()

const getTextContentExcludingNestedUl = (li: Element): string => {
  let text = ''
  for (const child of li.children as ElementContent[]) {
    if (isElement(child) && child.tagName === 'ul') continue
    text += toString(child)
  }
  return text.trim()
}

export function readArenaBlockText(li: Element): {
  highlighted: boolean
  embedDisabledFromMarker: boolean
  strippedContent: string
} {
  const textContent = getTextContentExcludingNestedUl(li)
  const trailingSection = textContent.match(TRAILING_MARKERS_PATTERN)?.[0] ?? ''
  return {
    highlighted: HIGHLIGHT_MARKER.test(trailingSection),
    embedDisabledFromMarker: EMBED_DISABLED_MARKER.test(trailingSection),
    strippedContent: stripTrailingMarkers(textContent),
  }
}

export function populateArenaBlockContent(
  ctx: BuildCtx,
  file: VFile,
  block: ArenaBlock,
  li: Element,
  url: string | undefined,
): void {
  const origin = referenceSourcePath(file.data)
  const applyLinkProcessing = (node: ElementContent): ElementContent => {
    const processTextNode = (value: string): ElementContent[] => {
      const results: ElementContent[] = []
      const ranges = extractLiteralReferences(value)
      let lastIndex = 0
      for (const range of ranges) {
        const start = range.span.start
        if (start > lastIndex) {
          results.push({ type: 'text', value: value.slice(lastIndex, start) })
        }

        const parsed = range
        const request = referenceRequest(parsed, origin)
        const record = recordReferenceDependency(ctx, file, request)
        const route =
          record.route ??
          (record.status === 'missing'
            ? missingReferenceRoute(request.target, origin, request.fragments)
            : undefined)
        const label = labelToHast(
          parseReferenceLabel(referenceLabelSource(parsed, '', record.implicitLabel)),
        )

        if (route) {
          const href = `/${route.slug}${(route.anchor ?? '').replace(/^#\^/, '#')}`
          results.push({
            type: 'element',
            tagName: 'a',
            properties: {
              href,
              className: ['internal'],
              'data-slug': route.slug,
              'data-no-popover': true,
            },
            children: label,
          })
        } else {
          results.push(...label)
        }

        lastIndex = range.span.end
      }

      if (lastIndex < value.length) {
        results.push({ type: 'text', value: value.slice(lastIndex) })
      }

      return results
    }

    const visitEl = (el: ElementContent) => {
      if (el.type !== 'element') return
      const e = el
      if (e.tagName === 'code' || e.tagName === 'pre') return

      if (e.tagName === 'a' && typeof e.properties?.href === 'string') {
        const dest = e.properties.href
        const classes = Array.isArray(e.properties.className)
          ? e.properties.className.filter((value): value is string => typeof value === 'string')
          : typeof e.properties.className === 'string'
            ? [e.properties.className]
            : []

        const isExternal = externalLinkRegex.test(dest)

        if (!isExternal && !dest.startsWith('#')) {
          const knownSlug = e.properties['data-slug']
          if (typeof knownSlug === 'string') {
            const canonical = new URL(dest, `https://base.com/${file.data.slug ?? ''}`)
            e.properties.href = `/${knownSlug}${canonical.search}${canonical.hash}`
            if (!classes.includes('internal')) classes.push('internal')
          } else {
            const [target, ...fragments] = dest.split('#')
            const record = recordReferenceDependency(ctx, file, {
              target,
              origin,
              fragments,
              embed: false,
              decodeUri: true,
              implicitAlias: false,
            })
            const route =
              record.route ??
              (record.status === 'missing'
                ? missingReferenceRoute(target, origin, fragments)
                : undefined)
            if (route) {
              e.properties.href = `/${route.slug}${(route.anchor ?? '').replace(/^#\^/, '#')}`
              e.properties['data-slug'] = route.slug
              if (!classes.includes('internal')) classes.push('internal')
            }
          }
        } else if (isExternal) {
          if (block.url && dest === url) {
            e.properties.href = block.url
          }
          if (!classes.includes('external')) classes.push('external')
        }

        if (classes.length > 0) {
          e.properties.className = classes
        }
        return
      }

      if (el.children) {
        const newChildren: ElementContent[] = []
        for (const child of el.children) {
          if (child.type === 'text') {
            newChildren.push(...processTextNode(child.value))
          } else {
            visitEl(child)
            newChildren.push(child)
          }
        }
        el.children = newChildren
      }
    }

    const cloned = structuredClone(node)
    visitEl(cloned)
    return cloned
  }

  const buildTitleNode = (li: Element): ElementContent | undefined => {
    let contentChildren: ElementContent[] = li.children as ElementContent[]

    for (const child of contentChildren) {
      if (isElement(child) && child.tagName === 'p') {
        contentChildren = child.children as ElementContent[]
        break
      }
    }

    let linkHref: string | undefined
    let linkElement: Element | undefined

    for (const child of contentChildren) {
      if (isElement(child) && child.tagName === 'a') {
        linkElement = child
        const href = child.properties?.href
        if (typeof href === 'string') {
          linkHref = href
        }
        break
      }
    }

    let titleText: string | undefined
    let foundSeparator = false
    const titleElements: ElementContent[] = []

    for (const child of contentChildren) {
      if (isElement(child) && child.tagName === 'ul') break

      if (child.type === 'text') {
        const text = child.value.trim()
        if (text === '—' || text === '--') {
          foundSeparator = true
          continue
        }
        const match = text.match(/^(?:—|--)\s*(.+)$/)
        if (match && match[1]) {
          titleText = stripTrailingMarkers(match[1]).trim()
          foundSeparator = true
          continue
        }
        if (foundSeparator && text.length > 0) {
          titleElements.push(child)
        }
      }

      if (foundSeparator && isElement(child) && child.tagName === 'a' && child !== linkElement) {
        titleElements.push(child)
      }
    }

    if (linkHref && (titleText || titleElements.length > 0)) {
      const wrapper: Element = { type: 'element', tagName: 'span', properties: {}, children: [] }

      if (titleText) {
        wrapper.children.push({ type: 'text', value: titleText })
      }

      wrapper.children.push(...titleElements)

      return applyLinkProcessing(wrapper)
    }

    const collected: ElementContent[] = []
    for (const child of li.children as ElementContent[]) {
      if (isElement(child) && child.tagName === 'ul') break
      const cloned = structuredClone(child)
      collected.push(cloned)
    }

    if (collected.length === 0) return undefined

    const wrapper: Element = {
      type: 'element',
      tagName: 'span',
      properties: {},
      children: collected,
    }

    return applyLinkProcessing(wrapper)
  }

  block.titleHtmlNode = buildTitleNode(li)

  const aggregateListItem = (li: Element): ElementContent => {
    const collected: ElementContent[] = []
    for (const child of li.children as ElementContent[]) {
      if (isElement(child) && child.tagName === 'ul') continue
      const cloned = structuredClone(child)
      collected.push(cloned)
    }
    const wrapped: ElementContent = {
      type: 'element',
      tagName: 'div',
      properties: {},
      children: collected,
    }
    return applyLinkProcessing(wrapped)
  }

  block.htmlNode = aggregateListItem(li)

  const findInternalLink = (
    node?: ElementContent,
  ): { slug: string; href: string; title: string; hash?: string } | undefined => {
    if (!node || node.type !== 'element') return undefined
    const el = node as Element

    if (el.tagName === 'a') {
      const classes = Array.isArray(el.properties?.className) ? el.properties.className : []
      const hasInternal = classes.some(c => c === 'internal')
      if (hasInternal) {
        const slug = el.properties?.['data-slug']
        if (typeof slug === 'string' && slug.length > 0) {
          const rawHref = el.properties?.href
          const hrefString = typeof rawHref === 'string' ? rawHref : undefined
          const [, anchor] = hrefString ? splitAnchor(hrefString) : ['', '']
          const canonicalSlug = stripSlashes(slug, true)
          const canonicalHref = `/${canonicalSlug}${anchor ?? ''}`
          return {
            slug,
            href: canonicalHref,
            hash: anchor && anchor.length > 0 ? anchor : undefined,
            title: toString(el),
          }
        }
      }
    }

    if (el.children) {
      for (const child of el.children as ElementContent[]) {
        const found = findInternalLink(child)
        if (found) return found
      }
    }

    return undefined
  }

  const internalLinkInfo = findInternalLink(block.titleHtmlNode) ?? findInternalLink(block.htmlNode)
  if (internalLinkInfo) {
    block.internalSlug = internalLinkInfo.slug
    block.internalHref = internalLinkInfo.href
    block.internalHash = internalLinkInfo.hash
    block.internalTitle = internalLinkInfo.title
  }
}
