import { drizzle } from "drizzle-orm/d1"
import { desc } from "drizzle-orm"
import { githubUsers } from "./schema"
import type { GithubUser } from "../quartz/types/mentions"

export async function handleMentions(env: Env): Promise<Response> {
  const db = drizzle(env.COMMENTS_ROOM)

  const users: GithubUser[] = await db
    .select({
      login: githubUsers.login,
      displayName: githubUsers.displayName,
      avatarUrl: githubUsers.avatarUrl,
      lastSeenAt: githubUsers.lastSeenAt,
    })
    .from(githubUsers)
    .orderBy(desc(githubUsers.lastSeenAt))
    .limit(100)

  return Response.json(users)
}
