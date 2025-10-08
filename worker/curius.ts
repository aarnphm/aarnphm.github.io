import type { Following, Link, User, Trail } from "../quartz/components/types"

const HEADERS: RequestInit = { headers: { "Content-Type": "application/json" } }

interface ApiResponse {
  user?: User
  links?: Link[]
  following?: Following[]
  trails?: Trail[]
  hasMore?: boolean
  page?: number
}

async function queryUsers(): Promise<ApiResponse> {
  try {
    const r = await fetch("https://curius.app/api/users/aaron-pham", HEADERS)
    if (!r.ok) throw new Error("Network error")
    const d: any = await r.json()
    return { user: d.user }
  } catch {
    return { user: {} as User }
  }
}

async function queryLinks(page: number = 0): Promise<ApiResponse> {
  try {
    const r = await fetch(`https://curius.app/api/users/3584/links?page=${page}`, HEADERS)
    if (!r.ok) throw new Error("Network error")
    const d: any = await r.json()
    return {
      links: d.userSaved || [],
      hasMore: d.hasMore ?? false,
      page,
    }
  } catch {
    return { links: [], hasMore: false, page }
  }
}

async function querySearchLinks(): Promise<ApiResponse> {
  try {
    const r = await fetch("https://curius.app/api/users/3584/searchLinks", HEADERS)
    if (!r.ok) throw new Error("Network error")
    const d: any = await r.json()
    return { links: d.links || [] }
  } catch {
    return { links: [] }
  }
}

async function queryTrails(page: number = 0, alias?: string): Promise<ApiResponse> {
  const hasAlias = typeof alias === "string" && alias.length > 0
  try {
    if (hasAlias) {
      const r = await fetch(`https://curius.app/api/links?page=${page}&trailHash=${alias}`, HEADERS)
      if (!r.ok) throw new Error("Network error")
      const d: any = await r.json()
      return {
        links: d.userSaved || [],
        hasMore: d.hasMore ?? false,
        page,
      }
    }

    const r = await fetch("https://curius.app/api/trails/3584", HEADERS)
    if (!r.ok) throw new Error("Network error")
    const d: any = await r.json()
    return { trails: d.trails || [] }
  } catch {
    return hasAlias ? { links: [], hasMore: false, page } : { trails: [] }
  }
}

async function queryFollowing(): Promise<ApiResponse> {
  try {
    const r = await fetch("https://curius.app/api/users/3584/followingLinks", HEADERS)
    if (!r.ok) throw new Error("Network error")
    const d: { users: Following[] } = await r.json()
    return { following: d.users.filter((u) => u.user.userLink !== "aaron-pham") }
  } catch {
    return { following: [] }
  }
}

export default async function handleCurius(request: Request): Promise<Response> {
  if (request.method !== "GET")
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    })
  const url = new URL(request.url)
  const query = url.searchParams.get("query")
  const page = parseInt(url.searchParams.get("page") || "0")
  const alias = url.searchParams.get("name") || undefined
  let resp: Partial<ApiResponse> = {}
  try {
    switch (query) {
      case "user":
        resp = await queryUsers()
        break
      case "links":
        resp = await queryLinks(page)
        break
      case "searchLinks":
        resp = await querySearchLinks()
        break
      case "following":
        resp = await queryFollowing()
        break
      case "trails":
        resp = await queryTrails(page, alias)
        break
      default:
        const [r1, r2, r3, r4] = await Promise.all([
          queryUsers(),
          queryLinks(0),
          queryFollowing(),
          queryTrails(),
        ])
        resp = {
          user: r1.user,
          links: r2.links,
          following: r3.following,
          hasMore: r2.hasMore,
          trails: r4.trails,
        }
        break
    }
    return new Response(JSON.stringify(resp), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  } catch (e: any) {
    return new Response(JSON.stringify({ error: e?.message || "unknown" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }
}
