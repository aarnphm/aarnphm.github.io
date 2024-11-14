import type { VercelRequest, VercelResponse } from "@vercel/node"

const Headers: RequestInit = { headers: { "Content-Type": "application/json" } }

interface Following {
  link: any
  user: any
}

interface Response {
  user?: any
  links?: any[]
  following?: Following[]
}

async function queryUsers() {
  try {
    const response = await fetch("https://curius.app/api/users/aaron-pham", Headers)
    if (!response.ok) {
      throw new Error("Network response was not ok")
    }
    const data = await response.json()
    return { user: data.user }
  } catch (err) {
    console.error(err)
    return { user: {} }
  }
}

async function queryLinks() {
  try {
    const response = await fetch("https://curius.app/api/users/3584/links?page=0", Headers)
    if (!response.ok) {
      throw new Error("Network response was not ok")
    }
    const data = await response.json()
    return { links: data.userSaved || [] }
  } catch (err) {
    console.error(err)
    return { links: [] }
  }
}

async function queryFollowing() {
  try {
    const response = await fetch("https://curius.app/api/users/3584/followingLinks", Headers)
    if (!response.ok) {
      throw new Error("Network response was not ok")
    }
    const data = await response.json()
    return { following: data.users.filter((user) => user.user.userLink !== "aaron-pham") }
  } catch (err) {
    console.error(err)
    return { following: [] }
  }
}

export default async function handler(req: VercelRequest, resp: VercelResponse) {
  const { query } = req.query

  let response: Response = {}
  try {
    switch (query) {
      case "user":
        response = await queryUsers()
        break
      case "links":
        response = await queryLinks()
        break
      case "following":
        response = await queryFollowing()
        break
      default:
        response = await Promise.all([queryUsers(), queryLinks(), queryFollowing()]).then(
          ([r1, r2, r3]) => ({ user: r1.user, links: r2.links, following: r3.following }),
        )
        break
    }
    resp.status(200).json({ ...response })
    return
  } catch (err) {
    console.error(err)
    resp.status(500).json({ error: err })
  }
}
