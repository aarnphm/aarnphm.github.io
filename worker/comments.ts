import { drizzle } from "drizzle-orm/d1"
import { eq, and, isNull } from "drizzle-orm"
import { comments } from "./schema"
import { DurableObject } from "cloudflare:workers"

type DbComment = typeof comments.$inferSelect

type Anchor = {
  headingId: string | null
  blockId: string | null
  paragraphIndex: number
  localOffset: number
  contextWords: [string, string]
}

type Comment = Omit<DbComment, "anchor"> & {
  anchor: Anchor | null
}

type OperationType = "new" | "update" | "delete"

type OperationInput = {
  opId: string
  type: OperationType
  comment: Comment
}

type OperationRecord = {
  seq: number
  opId: string
  type: OperationType
  comment: Comment
}

type RateLimit = {
  count: number
  windowStart: number
}

type Session = {
  pageId: string
  ip: string
}

const MAX_OP_LOG = 1000

export class MultiplayerComments extends DurableObject<Env> {
  private sessions: Map<WebSocket, Session>
  private rateLimits: Map<string, RateLimit>
  private sql: DurableObjectStorage["sql"]

  constructor(ctx: DurableObjectState, env: any) {
    super(ctx, env)
    this.sessions = new Map()
    this.rateLimits = new Map()
    this.sql = ctx.storage.sql
    const tableInfo = this.sql.exec("PRAGMA table_info(comment_ops)").toArray() as Array<{
      name: string
      pk: number
    }>
    const hasLegacyPk = tableInfo.some((row) => row.name === "seq" && row.pk === 1)
    if (hasLegacyPk) {
      this.sql.exec("DROP INDEX IF EXISTS idx_comment_ops_page_seq")
      this.sql.exec("DROP INDEX IF EXISTS idx_comment_ops_page_seq_unique")
      this.sql.exec("DROP TABLE IF EXISTS comment_ops")
    }
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS comment_ops (
        pageId TEXT NOT NULL,
        seq INTEGER NOT NULL,
        opId TEXT NOT NULL UNIQUE,
        opType TEXT NOT NULL,
        commentId TEXT NOT NULL,
        commentJson TEXT NOT NULL,
        createdAt INTEGER NOT NULL
      )
    `)
    this.sql.exec(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_comment_ops_page_seq_unique
      ON comment_ops (pageId, seq)
    `)
    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_comment_ops_page_seq
      ON comment_ops (pageId, seq)
    `)
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS page_seq (
        pageId TEXT PRIMARY KEY,
        seq INTEGER NOT NULL
      )
    `)
  }

  private isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value)
  }

  private parseAnchor(value: unknown): Anchor | null {
    if (value === null || value === undefined) return null
    let raw: unknown = value
    if (typeof raw === "string") {
      try {
        raw = JSON.parse(raw)
      } catch {
        return null
      }
    }
    if (!this.isRecord(raw)) return null
    const headingId = raw["headingId"]
    const blockId = raw["blockId"]
    const paragraphIndex = raw["paragraphIndex"]
    const localOffset = raw["localOffset"]
    const contextWords = raw["contextWords"]

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

  private serializeAnchor(anchor: Anchor | null): string | null {
    if (!anchor) return null
    try {
      return JSON.stringify(anchor)
    } catch {
      return null
    }
  }

  private normalizeComment(comment: DbComment | Comment): Comment {
    return {
      ...comment,
      anchor: this.parseAnchor(comment.anchor),
    }
  }

  private checkRateLimit(ip: string): boolean {
    const now = Date.now()
    const windowMs = 60000
    const maxOps = 20

    const limit = this.rateLimits.get(ip)
    if (!limit || now - limit.windowStart > windowMs) {
      this.rateLimits.set(ip, { count: 1, windowStart: now })
      return true
    }

    if (limit.count >= maxOps) {
      return false
    }

    limit.count++
    return true
  }

  private getSeqRange(pageId: string): { minSeq: number | null; maxSeq: number | null } {
    const rows = this.sql.exec(
      "SELECT MIN(seq) AS minSeq, MAX(seq) AS maxSeq FROM comment_ops WHERE pageId = ?",
      pageId,
    )
    const row = rows.toArray()[0] as { minSeq: number | null; maxSeq: number | null } | undefined
    return {
      minSeq: row?.minSeq ?? null,
      maxSeq: row?.maxSeq ?? null,
    }
  }

  private readOpById(opId: string): OperationRecord | null {
    const rows = this.sql.exec(
      "SELECT seq, opId, opType, commentJson FROM comment_ops WHERE opId = ?",
      opId,
    )
    const row = rows.toArray()[0] as
      | { seq: number; opId: string; opType: string; commentJson: string }
      | undefined
    if (!row) return null
    return {
      seq: row.seq,
      opId: row.opId,
      type: row.opType as OperationType,
      comment: this.normalizeComment(JSON.parse(row.commentJson) as Comment),
    }
  }

  private readOpsSince(pageId: string, since: number): OperationRecord[] {
    const rows = this.sql.exec(
      "SELECT seq, opId, opType, commentJson FROM comment_ops WHERE pageId = ? AND seq > ? ORDER BY seq",
      pageId,
      since,
    )
    return rows.toArray().map((row) => {
      const typedRow = row as { seq: number; opId: string; opType: string; commentJson: string }
      return {
        seq: typedRow.seq,
        opId: typedRow.opId,
        type: typedRow.opType as OperationType,
        comment: this.normalizeComment(JSON.parse(typedRow.commentJson) as Comment),
      }
    })
  }

  private nextSeq(pageId: string): number {
    const rows = this.sql.exec("SELECT seq FROM page_seq WHERE pageId = ?", pageId)
    const row = rows.toArray()[0] as { seq: number } | undefined
    const next = (row?.seq ?? 0) + 1
    if (row) {
      this.sql.exec("UPDATE page_seq SET seq = ? WHERE pageId = ?", next, pageId)
    } else {
      this.sql.exec("INSERT INTO page_seq (pageId, seq) VALUES (?, ?)", pageId, next)
    }
    return next
  }

  private storeOp(op: OperationInput, comment: Comment, seq: number): boolean {
    try {
      this.sql.exec(
        "INSERT INTO comment_ops (seq, pageId, opId, opType, commentId, commentJson, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?)",
        seq,
        comment.pageId,
        op.opId,
        op.type,
        comment.id,
        JSON.stringify(comment),
        Date.now(),
      )
      return true
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      if (message.includes("SQLITE_CONSTRAINT") && message.includes("comment_ops.opId")) {
        return false
      }
      throw err
    }
  }

  private trimOps(pageId: string, seq: number) {
    const minAllowed = seq - MAX_OP_LOG
    if (minAllowed > 0) {
      this.sql.exec("DELETE FROM comment_ops WHERE pageId = ? AND seq <= ?", pageId, minAllowed)
    }
  }

  private broadcastOp(op: OperationRecord) {
    const message = JSON.stringify({ type: "op", op })
    for (const [ws, session] of this.sessions) {
      if (session.pageId === op.comment.pageId) {
        ws.send(message)
      }
    }
  }

  private async persistNewComment(comment: Comment): Promise<Comment | null> {
    const db = drizzle(this.env.COMMENTS_ROOM)
    const anchorJson = this.serializeAnchor(this.parseAnchor(comment.anchor))

    await db
      .insert(comments)
      .values({
        id: comment.id,
        pageId: comment.pageId,
        parentId: comment.parentId,
        anchorHash: comment.anchorHash,
        anchorStart: comment.anchorStart,
        anchorEnd: comment.anchorEnd,
        anchorText: comment.anchorText,
        content: comment.content,
        author: comment.author,
        createdAt: comment.createdAt,
        updatedAt: null,
        deletedAt: null,
        anchor: anchorJson,
        orphaned: comment.orphaned ?? null,
        lastRecoveredAt: comment.lastRecoveredAt ?? null,
      })
      .onConflictDoNothing()

    const saved = await db.select().from(comments).where(eq(comments.id, comment.id)).get()
    return saved ? this.normalizeComment(saved) : null
  }

  private async persistUpdateComment(comment: Comment): Promise<Comment | null> {
    const db = drizzle(this.env.COMMENTS_ROOM)
    const now = Date.now()
    const anchorJson = this.serializeAnchor(this.parseAnchor(comment.anchor))

    await db
      .update(comments)
      .set({
        content: comment.content,
        anchorStart: comment.anchorStart,
        anchorEnd: comment.anchorEnd,
        anchor: anchorJson,
        orphaned: comment.orphaned ?? null,
        lastRecoveredAt: comment.lastRecoveredAt ?? null,
        updatedAt: now,
      })
      .where(eq(comments.id, comment.id))

    const updated = await db.select().from(comments).where(eq(comments.id, comment.id)).get()
    return updated ? this.normalizeComment(updated) : null
  }

  private async persistDeleteComment(commentId: string): Promise<Comment | null> {
    const db = drizzle(this.env.COMMENTS_ROOM)
    const now = Date.now()

    const comment = await db.select().from(comments).where(eq(comments.id, commentId)).get()
    if (!comment) return null

    await db.update(comments).set({ deletedAt: now }).where(eq(comments.id, commentId))
    return this.normalizeComment({ ...comment, deletedAt: now })
  }

  private async applyOperation(
    op: OperationInput,
    pageId: string,
  ): Promise<OperationRecord | null> {
    const existing = this.readOpById(op.opId)
    if (existing) return existing

    if (op.comment.pageId !== pageId) {
      return null
    }

    let saved: Comment | null = null

    if (op.type === "new") {
      saved = await this.persistNewComment(op.comment)
    } else if (op.type === "update") {
      saved = await this.persistUpdateComment(op.comment)
    } else if (op.type === "delete") {
      saved = await this.persistDeleteComment(op.comment.id)
    }

    if (!saved) return null

    const seq = this.nextSeq(pageId)
    const stored = this.storeOp(op, saved, seq)
    if (!stored) {
      const existing = this.readOpById(op.opId)
      return existing
    }
    this.trimOps(pageId, seq)

    const record: OperationRecord = {
      seq,
      opId: op.opId,
      type: op.type,
      comment: saved,
    }

    this.broadcastOp(record)
    return record
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url)

    if (url.pathname === "/comments/websocket") {
      const upgradeHeader = request.headers.get("Upgrade")
      if (upgradeHeader !== "websocket") {
        return new Response("expected websocket", { status: 400 })
      }

      const pageId = url.searchParams.get("pageId")
      if (!pageId) {
        return new Response("pageId required", { status: 400 })
      }

      const sinceParam = url.searchParams.get("since")
      const since = sinceParam ? Number.parseInt(sinceParam, 10) : null
      const sinceSeq = Number.isFinite(since) ? since : null

      const ip = request.headers.get("CF-Connecting-IP") || "unknown"

      const pair = new WebSocketPair()
      const [client, server] = Object.values(pair)

      this.ctx.acceptWebSocket(server)
      this.sessions.set(server, { pageId, ip })

      const { minSeq, maxSeq } = this.getSeqRange(pageId)
      const latestSeq = maxSeq ?? 0

      if (sinceSeq !== null && minSeq !== null && sinceSeq >= minSeq - 1) {
        const ops = this.readOpsSince(pageId, sinceSeq)
        server.send(
          JSON.stringify({
            type: "delta",
            ops,
            latestSeq,
          }),
        )
      } else {
        const db = drizzle(this.env.COMMENTS_ROOM)
        const existing = await db
          .select()
          .from(comments)
          .where(and(eq(comments.pageId, pageId), isNull(comments.deletedAt)))
          .orderBy(comments.createdAt)

        server.send(
          JSON.stringify({
            type: "init",
            comments: existing.map((comment) => this.normalizeComment(comment)),
            latestSeq,
          }),
        )
      }

      return new Response(null, {
        status: 101,
        webSocket: client,
      })
    }

    if (url.pathname === "/comments/export") {
      const pageId = url.searchParams.get("pageId")
      if (!pageId) {
        return new Response("pageId required", { status: 400 })
      }

      const db = drizzle(this.env.COMMENTS_ROOM)
      const allComments = await db
        .select()
        .from(comments)
        .where(eq(comments.pageId, pageId))
        .orderBy(comments.createdAt)

      const encoder = new TextEncoder()
      const stream = new ReadableStream({
        start(controller) {
          for (const comment of allComments) {
            const line = JSON.stringify(this.normalizeComment(comment)) + "\n"
            controller.enqueue(encoder.encode(line))
          }
          controller.close()
        },
      })

      return new Response(stream, {
        headers: {
          "Content-Type": "application/x-ndjson",
          "Content-Disposition": `attachment; filename="comments-${pageId.replace(/\//g, "-")}.jsonl"`,
        },
      })
    }

    if (request.method === "POST" && url.pathname === "/comments/add") {
      const body = (await request.json()) as { opId?: string; comment?: Comment } | Comment
      const comment = "comment" in body && body.comment ? body.comment : (body as Comment)
      const opId = "opId" in body && body.opId ? body.opId : crypto.randomUUID()
      const op: OperationInput = { opId, type: "new", comment }
      const record = await this.applyOperation(op, comment.pageId)
      if (!record) return new Response("failed to save comment", { status: 500 })
      return Response.json({ op: record }, { status: 201 })
    }

    if (request.method === "DELETE" && url.pathname === "/comments/delete") {
      const commentId = url.searchParams.get("id")
      if (!commentId) {
        return new Response("comment id required", { status: 400 })
      }
      const pageId = url.searchParams.get("pageId") ?? ""
      const opId = url.searchParams.get("opId") ?? crypto.randomUUID()
      const comment = { id: commentId, pageId } as Comment
      const op: OperationInput = { opId, type: "delete", comment }
      const record = await this.applyOperation(op, pageId)
      if (!record) return new Response("failed to delete comment", { status: 500 })
      return Response.json({ op: record }, { status: 200 })
    }

    if (request.method === "PATCH" && url.pathname === "/comments/modify") {
      const body = (await request.json()) as { opId?: string; comment?: Comment } | Comment
      const comment = "comment" in body && body.comment ? body.comment : (body as Comment)
      const opId = "opId" in body && body.opId ? body.opId : crypto.randomUUID()
      const op: OperationInput = { opId, type: "update", comment }
      const record = await this.applyOperation(op, comment.pageId)
      if (!record) return new Response("failed to update comment", { status: 500 })
      return Response.json({ op: record }, { status: 200 })
    }

    return new Response("not found", { status: 404 })
  }

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer) {
    try {
      const session = this.sessions.get(ws)
      if (!session) return

      if (!this.checkRateLimit(session.ip)) {
        ws.send(JSON.stringify({ type: "error", message: "rate limit exceeded" }))
        return
      }

      const data = JSON.parse(message as string) as { type?: string; op?: OperationInput }
      if (data.type !== "op" || !data.op) return

      const record = await this.applyOperation(data.op, session.pageId)
      if (!record) {
        ws.send(JSON.stringify({ type: "error", message: "operation failed" }))
        return
      }

      ws.send(JSON.stringify({ type: "ack", opId: record.opId, seq: record.seq }))
    } catch (err) {
      console.error("websocket message error:", err)
    }
  }

  webSocketClose(ws: WebSocket) {
    this.sessions.delete(ws)
  }

  webSocketError(ws: WebSocket, error: unknown) {
    console.error("websocket error:", error)
    this.sessions.delete(ws)
  }
}

const commentGithubAuthorPrefix = "comment-auth:github:"

export function normalizeReturnTo(request: Request, raw: string | null): string {
  if (!raw) return "/"
  let target: URL
  try {
    target = new URL(raw, request.url)
  } catch {
    return "/"
  }
  const origin = new URL(request.url).origin
  if (target.origin !== origin) return "/"
  return `${target.pathname}${target.search}${target.hash}`
}

export function normalizeAuthor(raw: string | null): string | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (!trimmed) return null
  if (trimmed.length > 128) return null
  return trimmed
}

function safeJsonForHtml(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c")
}

export async function getGithubCommentAuthor(
  kv: KVNamespace,
  login: string,
): Promise<string | null> {
  const raw = await kv.get(`${commentGithubAuthorPrefix}${login}`)
  if (!raw) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== "object") return null
  const value = parsed as { author?: unknown }
  if (typeof value.author !== "string") return null
  const normalized = normalizeAuthor(value.author)
  return normalized
}

export async function setGithubCommentAuthor(
  kv: KVNamespace,
  login: string,
  author: string,
): Promise<void> {
  await kv.put(
    `${commentGithubAuthorPrefix}${login}`,
    JSON.stringify({ author, updatedAt: Date.now() }),
  )
}

export function renderCommentAuthResponse(
  author: string,
  returnTo: string,
  login: string | null,
): Response {
  const payload = safeJsonForHtml({ author, returnTo, login })
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comment login</title>
</head>
<body>
<script>
const payload = ${payload};
try {
  const author = payload.login || payload.author;
  localStorage.setItem("comment-author", author);
  localStorage.setItem("comment-author-source", "github");
  if (payload.login) {
    localStorage.setItem("comment-author-github-login", payload.login);
  }
} catch {}
window.location.assign(payload.returnTo);
</script>
</body>
</html>`
  return new Response(html, {
    headers: { "Content-Type": "text/html; charset=utf-8" },
  })
}
