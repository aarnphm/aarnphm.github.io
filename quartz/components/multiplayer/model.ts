export type StructuralAnchor = {
  headingId: string | null
  blockId: string | null
  paragraphIndex: number
  localOffset: number
  contextWords: [string, string]
}

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
  anchor?: StructuralAnchor | null
  orphaned?: boolean | null
  lastRecoveredAt?: number | null
}

export type OperationType = "new" | "update" | "delete"

export type OperationInput = {
  opId: string
  type: OperationType
  comment: MultiplayerComment
}

export type OperationRecord = OperationInput & {
  seq: number
}

export type BroadcastMessage =
  | {
      type: "init"
      comments: MultiplayerComment[]
      latestSeq: number
    }
  | {
      type: "delta"
      ops: OperationRecord[]
      latestSeq: number
    }
  | {
      type: "op"
      op: OperationRecord
    }
  | {
      type: "ack"
      opId: string
      seq: number
    }
  | {
      type: "error"
      message: string
    }

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function isOperationType(value: unknown): value is OperationType {
  return value === "new" || value === "update" || value === "delete"
}

export function parseStructuralAnchor(value: unknown): StructuralAnchor | null {
  if (!isRecord(value)) return null
  const headingId = value["headingId"]
  const blockId = value["blockId"]
  const paragraphIndex = value["paragraphIndex"]
  const localOffset = value["localOffset"]
  const contextWords = value["contextWords"]

  if (headingId !== null && typeof headingId !== "string") return null
  if (blockId !== null && typeof blockId !== "string") return null
  if (typeof paragraphIndex !== "number") return null
  if (typeof localOffset !== "number") return null
  if (!Array.isArray(contextWords) || contextWords.length !== 2) return null
  if (typeof contextWords[0] !== "string" || typeof contextWords[1] !== "string") return null

  return {
    headingId: headingId ?? null,
    blockId: blockId ?? null,
    paragraphIndex,
    localOffset,
    contextWords: [contextWords[0], contextWords[1]],
  }
}

export function parseComment(value: unknown): MultiplayerComment | null {
  if (!isRecord(value)) return null
  const id = value["id"]
  const pageId = value["pageId"]
  const parentId = value["parentId"]
  const anchorHash = value["anchorHash"]
  const anchorStart = value["anchorStart"]
  const anchorEnd = value["anchorEnd"]
  const anchorText = value["anchorText"]
  const content = value["content"]
  const author = value["author"]
  const createdAt = value["createdAt"]
  const updatedAt = value["updatedAt"]
  const deletedAt = value["deletedAt"]
  const anchorRaw = value["anchor"]
  const orphaned = value["orphaned"]
  const lastRecoveredAt = value["lastRecoveredAt"]

  if (typeof id !== "string") return null
  if (typeof pageId !== "string") return null
  if (parentId !== null && typeof parentId !== "string") return null
  if (typeof anchorHash !== "string") return null
  if (typeof anchorStart !== "number") return null
  if (typeof anchorEnd !== "number") return null
  if (typeof anchorText !== "string") return null
  if (typeof content !== "string") return null
  if (typeof author !== "string") return null
  if (typeof createdAt !== "number") return null
  if (updatedAt !== null && typeof updatedAt !== "number") return null
  if (deletedAt !== null && typeof deletedAt !== "number") return null
  if (orphaned !== null && orphaned !== undefined && typeof orphaned !== "boolean") return null
  if (
    lastRecoveredAt !== null &&
    lastRecoveredAt !== undefined &&
    typeof lastRecoveredAt !== "number"
  )
    return null

  const anchor = anchorRaw ? parseStructuralAnchor(anchorRaw) : null

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
    const opId = item["opId"]
    const type = item["type"]
    const comment = parseComment(item["comment"])
    if (typeof opId !== "string") continue
    if (!isOperationType(type)) continue
    if (!comment) continue
    ops.push({ opId, type, comment })
  }
  return ops
}
