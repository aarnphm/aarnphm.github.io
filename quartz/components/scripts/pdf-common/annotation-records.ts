import {
  normalizePdfSource,
  type PdfAnchor,
  type PdfAnchorRect,
  type PdfAnnotationState,
  type PdfComment,
  type PdfOperationRecord,
  type PdfOperationType,
  type PdfOptions,
  type PdfState,
} from './viewer-model'

const PDF_COMMENT_ROOM_ENABLED_KEY = 'garden:comments-room-enabled:v1'

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function readCommentRoomEnabled(storage: Storage = localStorage): boolean {
  return storage.getItem(PDF_COMMENT_ROOM_ENABLED_KEY) === 'true'
}

function pdfCommentPageId(src: string): string {
  return `pdf:${normalizePdfSource(src)}`
}

export function getPdfAuthor(): string {
  const login = localStorage.getItem('comment-author-github-login')
  const stored = localStorage.getItem('comment-author')
  if (login && stored !== login) {
    localStorage.setItem('comment-author', login)
    return login
  }
  if (stored) return stored
  if (login) {
    localStorage.setItem('comment-author', login)
    return login
  }
  const author = `anon-${Math.random().toString(36).slice(2, 8)}`
  localStorage.setItem('comment-author', author)
  return author
}

export async function hashPdfText(text: string): Promise<string> {
  const data = new TextEncoder().encode(text)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(hashBuffer))
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('')
}

function readNullableString(value: unknown): string | null | undefined {
  if (value === null || value === undefined) return null
  return typeof value === 'string' ? value : undefined
}

function readNullableNumber(value: unknown): number | null | undefined {
  if (value === null || value === undefined) return null
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function readNullableBoolean(value: unknown): boolean | null | undefined {
  if (value === null || value === undefined) return null
  return typeof value === 'boolean' ? value : undefined
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

function parsePdfAnchor(value: unknown): PdfAnchor | null {
  if (!isRecord(value)) return null
  if (value['kind'] !== 'pdf') return null

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

export function parsePdfComment(value: unknown): PdfComment | null {
  if (!isRecord(value)) return null

  const id = value['id']
  const pageId = value['pageId']
  const parentId = readNullableString(value['parentId'])
  const anchorHash = value['anchorHash']
  const anchorStart = value['anchorStart']
  const anchorEnd = value['anchorEnd']
  const anchorText = value['anchorText']
  const content = value['content']
  const author = value['author']
  const createdAt = value['createdAt']
  const updatedAt = readNullableNumber(value['updatedAt'])
  const deletedAt = readNullableNumber(value['deletedAt'])
  const resolvedAt = readNullableNumber(value['resolvedAt'])
  const orphaned = readNullableBoolean(value['orphaned'])
  const lastRecoveredAt = readNullableNumber(value['lastRecoveredAt'])
  const anchor = parsePdfAnchor(value['anchor'])

  if (typeof id !== 'string') return null
  if (typeof pageId !== 'string') return null
  if (parentId === undefined) return null
  if (typeof anchorHash !== 'string') return null
  if (typeof anchorStart !== 'number' || !Number.isFinite(anchorStart)) return null
  if (typeof anchorEnd !== 'number' || !Number.isFinite(anchorEnd)) return null
  if (typeof anchorText !== 'string') return null
  if (typeof content !== 'string') return null
  if (typeof author !== 'string') return null
  if (typeof createdAt !== 'number' || !Number.isFinite(createdAt)) return null
  if (updatedAt === undefined) return null
  if (deletedAt === undefined) return null
  if (resolvedAt === undefined) return null
  if (orphaned === undefined) return null
  if (lastRecoveredAt === undefined) return null
  if (!anchor) return null

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
    resolvedAt,
    anchor,
    orphaned,
    lastRecoveredAt,
  }
}

export function isParsedPdfComment(value: PdfComment | null): value is PdfComment {
  return value !== null
}

function parsePdfOperationType(value: unknown): PdfOperationType | null {
  if (value === 'new' || value === 'update' || value === 'delete' || value === 'resolve') {
    return value
  }
  return null
}

export function parsePdfOperationRecord(value: unknown): PdfOperationRecord | null {
  if (!isRecord(value)) return null
  const seq = value['seq']
  const opId = value['opId']
  const type = parsePdfOperationType(value['type'])
  const comment = parsePdfComment(value['comment'])

  if (typeof seq !== 'number' || !Number.isFinite(seq)) return null
  if (typeof opId !== 'string') return null
  if (!type || !comment) return null

  return { seq, opId, type, comment }
}

export function upsertPdfComment(comments: PdfComment[], comment: PdfComment): PdfComment[] {
  const idx = comments.findIndex(item => item.id === comment.id)
  if (idx === -1) return [...comments, comment]
  const next = comments.slice()
  next[idx] = comment
  return next
}

export function mergePendingPdfComments(state: PdfState, comments: PdfComment[]): PdfComment[] {
  let next = comments
  for (const op of state.annotations.pendingOps.values()) {
    next = upsertPdfComment(next, op.comment)
  }
  return next
}

export function createPdfAnnotationState(options: PdfOptions): PdfAnnotationState {
  const src = normalizePdfSource(options.src)
  return {
    pageId: pdfCommentPageId(options.src),
    src,
    comments: [],
    pendingOps: new Map(),
    lastSeq: 0,
    hasSnapshot: false,
    reconnectEnabled: false,
    reconnectTimer: 0,
    websocket: null,
    composer: null,
    activeCommentId: null,
  }
}

export function filterPdfComments(state: PdfState, comments: PdfComment[]): PdfComment[] {
  return comments.filter(comment => {
    if (comment.pageId !== state.annotations.pageId) return false
    if (!comment.anchor || comment.anchor.kind !== 'pdf') return false
    return comment.anchor.src === state.annotations.src
  })
}

export function visiblePdfTopLevelComments(state: PdfState): PdfComment[] {
  return state.annotations.comments.filter(comment => {
    if (comment.parentId) return false
    if (comment.deletedAt || comment.resolvedAt) return false
    if (!comment.anchor || comment.anchor.kind !== 'pdf') return false
    return comment.anchor.src === state.annotations.src && comment.anchor.rects.length > 0
  })
}

export function pdfCommentReplies(state: PdfState, commentId: string): PdfComment[] {
  return state.annotations.comments.filter(comment => {
    return comment.parentId === commentId && !comment.deletedAt && !comment.resolvedAt
  })
}
