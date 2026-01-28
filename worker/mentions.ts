import { desc } from "drizzle-orm"
import { drizzle } from "drizzle-orm/d1"
import type { GithubUser } from "../quartz/types/mentions"
import { githubUsers } from "./schema"

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
