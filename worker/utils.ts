import { Octokit } from "octokit"

export type Props = {
  login: string
  name: string
  email: string
  accessToken: string
}

export function getUpstreamAuthorizeUrl({
  upstream_url,
  client_id,
  scope,
  redirect_uri,
  state,
}: {
  upstream_url: string
  client_id: string
  scope: string
  redirect_uri: string
  state?: string
}): string {
  const upstream = new URL(upstream_url)
  upstream.searchParams.set("client_id", client_id)
  upstream.searchParams.set("redirect_uri", redirect_uri)
  upstream.searchParams.set("scope", scope)
  if (state) upstream.searchParams.set("state", state)
  upstream.searchParams.set("response_type", "code")
  return upstream.href
}

export async function fetchUpstreamAuthToken({
  client_id,
  client_secret,
  code,
  redirect_uri,
  upstream_url,
}: {
  code: string | null | undefined
  upstream_url: string
  client_secret: string
  redirect_uri: string
  client_id: string
}): Promise<[string, null] | [null, Response]> {
  if (!code) return [null, new Response("Missing code", { status: 400 })]
  const resp = await fetch(upstream_url, {
    body: new URLSearchParams({ client_id, client_secret, code, redirect_uri }).toString(),
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    method: "POST",
  })
  if (!resp.ok) return [null, new Response("Failed to fetch access token", { status: 500 })]
  const form = await resp.formData()
  const accessToken = form.get("access_token") as string | null
  if (!accessToken) return [null, new Response("Missing access token", { status: 400 })]
  return [accessToken, null]
}

export async function getUserFromGitHub(accessToken: string): Promise<{
  login: string
  name: string
  email: string
}> {
  const octokit = new Octokit({ auth: accessToken })
  const { data } = await octokit.rest.users.getAuthenticated()
  return { login: data.login, name: data.name ?? data.login, email: (data as any).email ?? "" }
}
