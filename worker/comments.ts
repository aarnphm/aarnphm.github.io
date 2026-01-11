import { drizzle } from "drizzle-orm/d1"
import { eq, and, isNull } from "drizzle-orm"
import { comments } from "./schema"
import { DurableObject } from "cloudflare:workers"

type Comment = typeof comments.$inferSelect

type RateLimit = {
  count: number
  windowStart: number
}

export class MultiplayerComments extends DurableObject<Env> {
  private sessions: Map<WebSocket, { pageId: string; ip: string }>
  private rateLimits: Map<string, RateLimit>

  constructor(ctx: DurableObjectState, env: any) {
    super(ctx, env)
    this.sessions = new Map()
    this.rateLimits = new Map()
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

      const ip = request.headers.get("CF-Connecting-IP") || "unknown"

      const pair = new WebSocketPair()
      const [client, server] = Object.values(pair)

      this.ctx.acceptWebSocket(server)
      this.sessions.set(server, { pageId, ip })

      const db = drizzle(this.env.COMMENTS_ROOM)
      const existing = await db
        .select()
        .from(comments)
        .where(and(eq(comments.pageId, pageId), isNull(comments.deletedAt)))
        .orderBy(comments.createdAt)

      server.send(
        JSON.stringify({
          type: "init",
          comments: existing,
        }),
      )

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
            const line = JSON.stringify(comment) + "\n"
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
      const comment = (await request.json()) as Comment
      return this.handleNewComment(comment)
    }

    if (request.method === "DELETE" && url.pathname === "/comments/delete") {
      const commentId = url.searchParams.get("id")
      if (!commentId) {
        return new Response("comment id required", { status: 400 })
      }
      await this.handleDeleteComment(commentId)
      return new Response(null, { status: 204 })
    }

    if (request.method === "PATCH" && url.pathname === "/comments/modify") {
      const comment = (await request.json()) as Comment
      await this.handleUpdateComment(comment)
      return new Response(null, { status: 204 })
    }

    return new Response("not found", { status: 404 })
  }

  private async handleNewComment(comment: Comment): Promise<Response> {
    const db = drizzle(this.env.COMMENTS_ROOM)

    // Check if comment already exists (idempotency)
    const existing = await db.select().from(comments).where(eq(comments.id, comment.id)).get()
    if (existing) {
      return Response.json(existing, { status: 200 })
    }

    await db.insert(comments).values({
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
    })

    const saved = await db.select().from(comments).where(eq(comments.id, comment.id)).get()

    for (const [ws, session] of this.sessions) {
      if (session.pageId === comment.pageId) {
        ws.send(
          JSON.stringify({
            type: "new",
            comment: saved,
          }),
        )
      }
    }

    return Response.json(saved, { status: 201 })
  }

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer) {
    try {
      const session = this.sessions.get(ws)
      if (!session) return

      if (!this.checkRateLimit(session.ip)) {
        ws.send(JSON.stringify({ type: "error", message: "rate limit exceeded" }))
        return
      }

      const data = JSON.parse(message as string)

      if (data.type === "new") {
        await this.handleNewComment(data.comment)
      }

      if (data.type === "update") {
        await this.handleUpdateComment(data.comment)
      }

      if (data.type === "delete") {
        await this.handleDeleteComment(data.commentId)
      }
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

  private async handleUpdateComment(comment: Comment) {
    const db = drizzle(this.env.COMMENTS_ROOM)
    const now = Date.now()

    await db
      .update(comments)
      .set({ content: comment.content, updatedAt: now })
      .where(eq(comments.id, comment.id))

    const updated = await db.select().from(comments).where(eq(comments.id, comment.id)).get()

    for (const [ws, session] of this.sessions) {
      if (session.pageId === comment.pageId) {
        ws.send(
          JSON.stringify({
            type: "update",
            comment: updated,
          }),
        )
      }
    }
  }

  private async handleDeleteComment(commentId: string) {
    const db = drizzle(this.env.COMMENTS_ROOM)
    const now = Date.now()

    const comment = await db.select().from(comments).where(eq(comments.id, commentId)).get()

    if (!comment) return

    await db.update(comments).set({ deletedAt: now }).where(eq(comments.id, commentId))

    for (const [ws, session] of this.sessions) {
      if (session.pageId === comment.pageId) {
        ws.send(
          JSON.stringify({
            type: "delete",
            comment: { ...comment, deletedAt: now },
          }),
        )
      }
    }
  }
}
