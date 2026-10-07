import type { JsonCanvas, NodeType } from '../plugins/transformers/jcast/types'
import type { FullSlug } from './path'
import { canvasEdgeGeometry, canvasEdgePath } from './canvas-edge'
import { isRecord } from './type-guards'

// Canvas and base pages need their runtime to render; a link preview draws this ~2 KB summary instead.
export const DOCUMENT_PREVIEW_KIND = 'document-preview-v1'

export type DocumentPreviewType = 'canvas' | 'base'

export interface CanvasPreviewNode {
  kind: NodeType
  x: number
  y: number
  w: number
  h: number
  /** JSON Canvas preset `1`–`6` or a hex colour. */
  color: string | null
  label: string | null
}

export interface CanvasPreviewEdge {
  d: string
  color: string | null
}

export interface CanvasPreview {
  kind: typeof DOCUMENT_PREVIEW_KIND
  type: 'canvas'
  slug: string
  title: string
  box: { x: number; y: number; w: number; h: number }
  nodes: CanvasPreviewNode[]
  edges: CanvasPreviewEdge[]
  counts: { cards: number; groups: number; connections: number }
}

export type BasePreviewSample =
  | { type: 'cards'; items: { title: string; image: string | null }[] }
  | { type: 'table'; columns: string[]; rows: string[][] }
  | { type: 'list'; items: string[] }

export interface BasePreviewView {
  name: string
  type: string
  count: number
}

export interface BasePreview {
  kind: typeof DOCUMENT_PREVIEW_KIND
  type: 'base'
  slug: string
  title: string
  description: string | null
  views: BasePreviewView[]
  /** Leading rows of the first view that renders without a runtime (maps need one). */
  sample: (BasePreviewSample & { view: string }) | null
}

export type DocumentPreview = CanvasPreview | BasePreview

export const documentPreviewSlug = (slug: string): FullSlug => `static/preview/${slug}` as FullSlug

export const documentPreviewUrl = (slug: string, reference: string): URL =>
  new URL(`/${documentPreviewSlug(slug)}.json`, reference)

export const isDocumentPreviewType = (value: unknown): value is DocumentPreviewType =>
  value === 'canvas' || value === 'base'

const CANVAS_KINDS: readonly NodeType[] = ['text', 'file', 'link', 'group']
const CANVAS_COLOR = /^(?:[1-6]|#[0-9a-f]{3,8})$/i
const LABEL_LENGTH = 48
const CANVAS_MARGIN = 40

export const canvasPreviewColor = (value: unknown): string | null =>
  typeof value === 'string' && CANVAS_COLOR.test(value) ? value : null

export function previewLabel(value: string, length = LABEL_LENGTH): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length > length ? `${text.slice(0, length - 1).trimEnd()}…` : text
}

/** First line of a text card, with wikilink, link, and emphasis markup reduced to its words. */
function textCardLabel(text: string): string | null {
  const line = text
    .split('\n')
    .map(entry => entry.trim())
    .find(entry => entry.length > 0)
  if (!line) return null
  const plain = line
    .replace(
      /!?\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]/g,
      (_, target: string, alias?: string) => (alias ?? target.split('/').pop() ?? target).trim(),
    )
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s+|^>\s*|^[-*+]\s+|[*_`~]/g, '')
  return plain ? previewLabel(plain) : null
}

const fileCardLabel = (file: string, displayName?: string): string =>
  previewLabel(displayName ?? file.split('/').pop()?.replace(/\.md$/, '') ?? file)

export function buildCanvasPreview(input: {
  slug: string
  title: string
  canvas: JsonCanvas
  displayNames?: ReadonlyMap<string, string>
}): CanvasPreview {
  const sources = input.canvas.nodes.filter(
    node =>
      CANVAS_KINDS.includes(node.type) &&
      [node.x, node.y, node.width, node.height].every(Number.isFinite),
  )
  // Groups paint first so cards sit on top of their containers.
  const ordered = [
    ...sources.filter(node => node.type === 'group'),
    ...sources.filter(node => node.type !== 'group'),
  ]
  const byId = new Map(ordered.map(node => [node.id, node]))

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const node of ordered) {
    minX = Math.min(minX, node.x)
    minY = Math.min(minY, node.y)
    maxX = Math.max(maxX, node.x + node.width)
    maxY = Math.max(maxY, node.y + node.height)
  }
  const box =
    ordered.length > 0
      ? {
          x: Math.round(minX - CANVAS_MARGIN),
          y: Math.round(minY - CANVAS_MARGIN),
          w: Math.round(maxX - minX + 2 * CANVAS_MARGIN),
          h: Math.round(maxY - minY + 2 * CANVAS_MARGIN),
        }
      : { x: 0, y: 0, w: 0, h: 0 }

  const nodes = ordered.map(
    (node): CanvasPreviewNode => ({
      kind: node.type,
      x: Math.round(node.x),
      y: Math.round(node.y),
      w: Math.round(node.width),
      h: Math.round(node.height),
      color: canvasPreviewColor(node.color),
      label:
        node.type === 'group'
          ? node.label
            ? previewLabel(node.label)
            : null
          : node.type === 'file' && node.file
            ? fileCardLabel(node.file, input.displayNames?.get(node.id))
            : node.type === 'text' && node.text
              ? textCardLabel(node.text)
              : node.type === 'link' && node.url
                ? previewLabel(node.url.replace(/^https?:\/\/(?:www\.)?/, ''))
                : null,
    }),
  )

  const edges = input.canvas.edges.flatMap((edge): CanvasPreviewEdge[] => {
    const from = byId.get(edge.fromNode)
    const to = byId.get(edge.toNode)
    if (!from || !to) return []
    const geometry = canvasEdgeGeometry(from, to, edge.fromSide, edge.toSide)
    return [{ d: canvasEdgePath(geometry, Math.round), color: canvasPreviewColor(edge.color) }]
  })

  const groups = nodes.filter(node => node.kind === 'group').length
  return {
    kind: DOCUMENT_PREVIEW_KIND,
    type: 'canvas',
    slug: input.slug,
    title: input.title,
    box,
    nodes,
    edges,
    counts: { cards: nodes.length - groups, groups, connections: edges.length },
  }
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const isNullableString = (value: unknown): value is string | null =>
  value === null || typeof value === 'string'

const isStringList = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(item => typeof item === 'string')

function isCanvasPreviewNode(value: unknown): value is CanvasPreviewNode {
  return (
    isRecord(value) &&
    CANVAS_KINDS.includes(value.kind as NodeType) &&
    [value.x, value.y, value.w, value.h].every(isFiniteNumber) &&
    (value.color === null || canvasPreviewColor(value.color) !== null) &&
    isNullableString(value.label)
  )
}

function isCanvasPreviewEdge(value: unknown): value is CanvasPreviewEdge {
  return (
    isRecord(value) &&
    typeof value.d === 'string' &&
    /^[MLC0-9 .,-]+$/.test(value.d) &&
    (value.color === null || canvasPreviewColor(value.color) !== null)
  )
}

function isBasePreviewSample(value: unknown): value is BasePreviewSample & { view: string } {
  if (!isRecord(value) || typeof value.view !== 'string') return false
  switch (value.type) {
    case 'cards':
      return (
        Array.isArray(value.items) &&
        value.items.every(
          item => isRecord(item) && typeof item.title === 'string' && isNullableString(item.image),
        )
      )
    case 'table':
      return (
        isStringList(value.columns) &&
        Array.isArray(value.rows) &&
        value.rows.every(row => isStringList(row))
      )
    case 'list':
      return isStringList(value.items)
    default:
      return false
  }
}

export function readDocumentPreview(
  value: unknown,
  type: DocumentPreviewType,
  slug: string,
): DocumentPreview | null {
  if (
    !isRecord(value) ||
    value.kind !== DOCUMENT_PREVIEW_KIND ||
    value.type !== type ||
    value.slug !== slug ||
    typeof value.title !== 'string'
  )
    return null

  if (type === 'canvas') {
    const { box, counts } = value
    const valid =
      isRecord(box) &&
      [box.x, box.y, box.w, box.h].every(isFiniteNumber) &&
      isRecord(counts) &&
      [counts.cards, counts.groups, counts.connections].every(isFiniteNumber) &&
      Array.isArray(value.nodes) &&
      value.nodes.every(isCanvasPreviewNode) &&
      Array.isArray(value.edges) &&
      value.edges.every(isCanvasPreviewEdge)
    return valid ? (value as unknown as CanvasPreview) : null
  }

  const valid =
    isNullableString(value.description) &&
    Array.isArray(value.views) &&
    value.views.every(
      view =>
        isRecord(view) &&
        typeof view.name === 'string' &&
        typeof view.type === 'string' &&
        isFiniteNumber(view.count),
    ) &&
    (value.sample === null || isBasePreviewSample(value.sample))
  return valid ? (value as unknown as BasePreview) : null
}
