import type { Element, RootContent } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import { FullSlug, isFullSlug } from './path'

export type TranscludeTarget = {
  inner: Element
  targetSlug: FullSlug
  url: string
  alias: string
  blockRef?: string
  rawMetadata?: string
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

function readHastStringProperty(
  properties: Element['properties'] | undefined,
  ...names: string[]
): string | undefined {
  if (!properties) return undefined

  for (const name of names) {
    const value = properties[name]
    if (typeof value === 'string') return value
  }

  return undefined
}

export function readTranscludeTarget(node: Element): TranscludeTarget | undefined {
  const inner = node.children[0]
  if (inner?.type !== 'element') return undefined

  const targetSlug = readHastStringProperty(inner.properties, 'dataSlug', 'data-slug')
  if (!targetSlug || !isFullSlug(targetSlug)) return undefined

  const blockRef = readHastStringProperty(node.properties, 'dataBlock', 'data-block')
  const dataEmbedAlias = readHastStringProperty(
    node.properties,
    'dataEmbedAlias',
    'data-embed-alias',
  )

  return {
    inner,
    targetSlug,
    url: readHastStringProperty(node.properties, 'dataUrl', 'data-url') ?? '',
    alias:
      dataEmbedAlias !== undefined && dataEmbedAlias !== 'undefined'
        ? dataEmbedAlias
        : (blockRef ?? ''),
    blockRef,
    rawMetadata: readHastStringProperty(node.properties, 'dataMetadata', 'data-metadata'),
  }
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
