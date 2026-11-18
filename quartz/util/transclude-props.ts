import type { Element, RootContent } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import type { ReferenceSelector } from './references/catalog-types'
import type { ReferenceDependency } from './references/dependencies'
import { FullSlug, isFullSlug } from './path'

export type TranscludeTarget = {
  inner: Element
  targetSlug: FullSlug
  url: string
  alias: string
  blockRef?: string
  request: ReferenceDependency
  selector: ReferenceSelector
  fragments: readonly string[]
  metadata?: Readonly<Record<string, unknown>>
}

export type HeadingSectionBounds = { startIdx: number; endIdx?: number }

function readHastClassTokens(value: Element['properties'][string] | undefined): string[] {
  if (Array.isArray(value)) {
    return value.flatMap(token =>
      typeof token === 'string' || typeof token === 'number'
        ? `${token}`.split(/\s+/).filter(Boolean)
        : [],
    )
  }

  return typeof value === 'string' || typeof value === 'number'
    ? `${value}`.split(/\s+/).filter(Boolean)
    : []
}

export function getHastClassNames(node: Element): string[] {
  const properties = node.properties
  if (!properties) return []

  const seen = new Set<string>()
  return [
    ...readHastClassTokens(properties.className),
    ...readHastClassTokens(properties.class),
  ].filter(className => {
    if (seen.has(className)) return false
    seen.add(className)
    return true
  })
}

export function hasHastClass(node: Element, className: string): boolean {
  return getHastClassNames(node).includes(className)
}

export function readTranscludeTarget(node: Element): TranscludeTarget | undefined {
  const inner = node.children[0]
  const request = node.data?.referenceRequest
  if (
    inner?.type !== 'element' ||
    inner.tagName !== 'a' ||
    !request?.embed ||
    request.status !== 'found' ||
    !request.selector
  )
    return undefined

  const targetSlug = request.route?.slug ?? request.resolvedSlug
  if (!targetSlug || !isFullSlug(targetSlug)) return undefined
  const selector = request.selector
  const blockRef =
    selector.kind === 'whole' || selector.kind === 'base'
      ? undefined
      : selector.kind === 'heading' || selector.kind === 'block'
        ? selector.anchor
        : request.fragments.length
          ? `#${request.fragments.join('#')}`
          : undefined

  return {
    inner,
    targetSlug,
    request,
    selector,
    fragments: request.fragments,
    metadata:
      request.resolvedSettings?.kind === 'object' ? request.resolvedSettings.value : undefined,
    url: request.route?.slug ?? targetSlug,
    alias: request.label ?? request.implicitLabel ?? '',
    blockRef,
  }
}

export function transcludeVisitKey(target: TranscludeTarget): string {
  const { request, selector } = target
  if (selector.kind === 'whole') return request.resolvedSlug ?? target.targetSlug
  const source = request.resolvedPath ?? request.resolvedSlug ?? target.targetSlug
  const selection =
    selector.kind === 'heading' || selector.kind === 'block'
      ? [selector.kind, selector.anchor]
      : selector.kind === 'notebook'
        ? [selector.kind, selector.cells]
        : selector.kind === 'base'
          ? [selector.kind, request.route?.slug ?? selector.view]
          : [selector.kind, selector.raw]
  return JSON.stringify([source, selection])
}

function findHeadingElement(node: RootContent, id?: string): Element | undefined {
  if (node.type !== 'element') return undefined

  if (headingRank(node) && (id === undefined || node.properties?.id === id)) {
    return node
  }

  for (const child of node.children) {
    const heading = findHeadingElement(child, id)
    if (heading) return heading
  }

  return undefined
}

export function findHeadingSectionBounds(
  children: RootContent[],
  blockRef: string,
): HeadingSectionBounds | undefined {
  let startIdx: number | undefined = undefined
  let startDepth: number | undefined = undefined

  for (const [idx, child] of children.entries()) {
    if (startIdx === undefined || startDepth === undefined) {
      const heading = findHeadingElement(child, blockRef)
      if (!heading) continue
      const headingDepth = headingRank(heading)
      if (!headingDepth) continue
      startIdx = idx
      startDepth = headingDepth
      continue
    }

    const nextHeading = findHeadingElement(child)
    if (!nextHeading) continue
    const nextDepth = headingRank(nextHeading)
    if (!nextDepth) continue
    if (nextDepth <= startDepth) {
      return { startIdx, endIdx: idx }
    }
  }

  return startIdx === undefined ? undefined : { startIdx }
}
