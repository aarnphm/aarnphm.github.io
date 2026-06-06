import { isRecord } from '../../util/type-guards'

export type StructuralAnchor = {
  headingId: string | null
  blockId: string | null
  paragraphIndex: number
  localOffset: number
  contextWords: [string, string]
}

export type PdfAnchorRect = {
  page: number
  left: number
  top: number
  width: number
  height: number
}

export type PdfAnchor = { kind: 'pdf'; src: string; rects: PdfAnchorRect[] }

export type CommentAnchor = StructuralAnchor | PdfAnchor

export type MultiplayerComment = {
  id: string
  pageId: string
  parentId: string | null
  anchorHash: string
  anchorStart: number
  anchorEnd: number
  anchorText: string
  content: string
  author: string
  createdAt: number
  updatedAt: number | null
  deletedAt: number | null
  resolvedAt: number | null
  anchor?: CommentAnchor | null
  orphaned?: boolean | null
  lastRecoveredAt?: number | null
}

export type OperationType = 'new' | 'update' | 'delete' | 'resolve'

export type OperationInput = { opId: string; type: OperationType; comment: MultiplayerComment }

export type OperationRecord = OperationInput & { seq: number }

export type BroadcastMessage =
  | { type: 'init'; comments: MultiplayerComment[]; latestSeq: number }
  | { type: 'delta'; ops: OperationRecord[]; latestSeq: number }
  | { type: 'op'; op: OperationRecord }
  | { type: 'ack'; opId: string; seq: number }
  | { type: 'error'; message: string }

export { isRecord }

export function isOperationType(value: unknown): value is OperationType {
  return value === 'new' || value === 'update' || value === 'delete' || value === 'resolve'
}

export function parseStructuralAnchor(value: unknown): StructuralAnchor | null {
  if (!isRecord(value)) return null
  const headingId = value['headingId']
  const blockId = value['blockId']
  const paragraphIndex = value['paragraphIndex']
  const localOffset = value['localOffset']
  const contextWords = value['contextWords']

  if (headingId !== null && typeof headingId !== 'string') return null
  if (blockId !== null && typeof blockId !== 'string') return null
  if (typeof paragraphIndex !== 'number') return null
  if (typeof localOffset !== 'number') return null
  if (!Array.isArray(contextWords) || contextWords.length !== 2) return null
  if (typeof contextWords[0] !== 'string' || typeof contextWords[1] !== 'string') return null

  return {
    headingId: headingId ?? null,
    blockId: blockId ?? null,
    paragraphIndex,
    localOffset,
    contextWords: [contextWords[0], contextWords[1]],
  }
}

function parsePdfAnchorRect(value: unknown): PdfAnchorRect | null {
  if (!isRecord(value)) return null
  const page = value['page']
  const left = value['left']
  const top = value['top']
  const width = value['width']
  const height = value['height']

  if (typeof page !== 'number' || !Number.isInteger(page) || page < 1) return null
  if (typeof left !== 'number' || !Number.isFinite(left)) return null
  if (typeof top !== 'number' || !Number.isFinite(top)) return null
  if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0) return null
  if (typeof height !== 'number' || !Number.isFinite(height) || height <= 0) return null

  return { page, left, top, width, height }
}

export function parseCommentAnchor(value: unknown): CommentAnchor | null {
  if (!isRecord(value)) return null
  if (value['kind'] === 'pdf') {
    const src = value['src']
    const rects = value['rects']
    if (typeof src !== 'string') return null
    if (!Array.isArray(rects) || rects.length === 0) return null

    const parsedRects: PdfAnchorRect[] = []
    for (const rect of rects) {
      const parsed = parsePdfAnchorRect(rect)
      if (!parsed) return null
      parsedRects.push(parsed)
    }

    return { kind: 'pdf', src, rects: parsedRects }
  }

  return parseStructuralAnchor(value)
}

export function isStructuralAnchor(
  value: CommentAnchor | null | undefined,
): value is StructuralAnchor {
  return value !== null && value !== undefined && !('kind' in value)
}

export function parseComment(value: unknown): MultiplayerComment | null {
  if (!isRecord(value)) return null
  const id = value['id']
  const pageId = value['pageId']
  const parentId = value['parentId']
  const anchorHash = value['anchorHash']
  const anchorStart = value['anchorStart']
  const anchorEnd = value['anchorEnd']
  const anchorText = value['anchorText']
  const content = value['content']
  const author = value['author']
  const createdAt = value['createdAt']
  const updatedAt = value['updatedAt']
  const deletedAt = value['deletedAt']
  const resolvedAt = value['resolvedAt']
  const anchorRaw = value['anchor']
  const orphaned = value['orphaned']
  const lastRecoveredAt = value['lastRecoveredAt']

  if (typeof id !== 'string') return null
  if (typeof pageId !== 'string') return null
  if (parentId !== null && typeof parentId !== 'string') return null
  if (typeof anchorHash !== 'string') return null
  if (typeof anchorStart !== 'number') return null
  if (typeof anchorEnd !== 'number') return null
  if (typeof anchorText !== 'string') return null
  if (typeof content !== 'string') return null
  if (typeof author !== 'string') return null
  if (typeof createdAt !== 'number') return null
  if (updatedAt !== null && typeof updatedAt !== 'number') return null
  if (deletedAt !== null && typeof deletedAt !== 'number') return null
  if (resolvedAt !== null && resolvedAt !== undefined && typeof resolvedAt !== 'number') return null
  if (orphaned !== null && orphaned !== undefined && typeof orphaned !== 'boolean') return null
  if (
    lastRecoveredAt !== null &&
    lastRecoveredAt !== undefined &&
    typeof lastRecoveredAt !== 'number'
  )
    return null

  const anchor = anchorRaw ? parseCommentAnchor(anchorRaw) : null

  return {
    id,
    pageId,
    parentId,
    anchorHash,
    anchorStart,
    anchorEnd,
    anchorText,
    content,
    author,
    createdAt,
    updatedAt,
    deletedAt,
    resolvedAt: resolvedAt ?? null,
    anchor,
    orphaned: orphaned ?? null,
    lastRecoveredAt: lastRecoveredAt ?? null,
  }
}

export function parsePendingOps(raw: string): OperationInput[] {
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    return []
  }
  if (!Array.isArray(data)) return []
  const ops: OperationInput[] = []
  for (const item of data) {
    if (!isRecord(item)) continue
    const opId = item['opId']
    const type = item['type']
    const comment = parseComment(item['comment'])
    if (typeof opId !== 'string') continue
    if (!isOperationType(type)) continue
    if (!comment) continue
    ops.push({ opId, type, comment })
  }
  return ops
}
