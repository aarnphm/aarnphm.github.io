import http from 'node:http'

interface OAuthCallbackOptions {
  provider: string
  redirectUri: string
}

interface OAuthCallbackGrant {
  code: string
  scope: string
}

export function isLoopbackRedirect(uri: string): boolean {
  const url = URL.parse(uri)
  return (
    url !== null &&
    url.protocol === 'http:' &&
    ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) &&
    !url.username &&
    !url.password
  )
}

export function listenForOAuthCallback({ provider, redirectUri }: OAuthCallbackOptions) {
  if (!isLoopbackRedirect(redirectUri))
    throw new Error('OAuth callback requires an HTTP loopback redirect')
  const redirect = new URL(redirectUri)
  const host = redirect.hostname === '[::1]' ? '::1' : '127.0.0.1'
  const server = http.createServer()
  const grant = new Promise<OAuthCallbackGrant>((resolve, reject) => {
    server.on('error', reject)
    server.on('request', (request, response) => {
      response.setHeader('Content-Type', 'text/plain; charset=utf-8')
      response.setHeader('X-Content-Type-Options', 'nosniff')
      response.setHeader('Cache-Control', 'no-store')
      const url = URL.parse(request.url ?? '/', redirect)
      if (!url) {
        response.statusCode = 400
        response.end('Malformed OAuth callback request.')
        return
      }
      if (url.pathname !== redirect.pathname || request.method !== 'GET') {
        response.statusCode = request.method === 'GET' ? 404 : 405
        response.end('Unsupported OAuth callback request.')
        return
      }

      const code = url.searchParams.get('code')
      const error = url.searchParams.get('error')
      if (error || !code) {
        response.statusCode = 400
        response.end(`${provider} authorization failed. Return to the terminal for details.`)
        server.close()
        reject(new Error(error || 'no authorization code in redirect'))
        return
      }

      response.end(`${provider} authorized. Close this tab and return to the terminal.`)
      server.close()
      resolve({ code, scope: url.searchParams.get('scope') ?? '' })
    })
    server.listen({ port: redirect.port ? Number(redirect.port) : 80, host })
  })
  return { server, grant }
}
