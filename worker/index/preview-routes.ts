import { greaterWrongPostUrl, lessWrongTargetFromSearchParams } from '../../quartz/util/lesswrong'
import { readGreaterWrongPreviewHtml } from '../../quartz/util/lesswrong-preview'
import { sepEntryUrl, sepTargetFromSearchParams } from '../../quartz/util/sep'
import { readSepPreviewHtml } from '../../quartz/util/sep-preview'

const PREVIEW_USER_AGENT = 'Mozilla/5.0 (compatible; AarnphmGarden/1.0; +https://aarnphm.xyz)'

export async function handleLessWrong(request: Request): Promise<Response> {
  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', Allow: 'GET' },
    })
  }

  const url = new URL(request.url)
  const target = lessWrongTargetFromSearchParams(url.searchParams)
  if (!target) {
    return new Response(JSON.stringify({ error: 'LessWrong postId is required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const response = await fetch(greaterWrongPostUrl(target), {
    headers: { Accept: 'text/html', 'User-Agent': PREVIEW_USER_AGENT },
    cf: { cacheTtl: 300, cacheEverything: true },
  })

  if (!response.ok) {
    return new Response(JSON.stringify({ error: 'LessWrong preview unavailable' }), {
      status: response.status === 404 ? 404 : 502,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const preview = readGreaterWrongPreviewHtml(await response.text(), target)
  if (!preview) {
    return new Response(JSON.stringify({ error: 'LessWrong preview unavailable' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  return new Response(JSON.stringify(preview), { headers: { 'Content-Type': 'application/json' } })
}

export async function handleSep(request: Request): Promise<Response> {
  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', Allow: 'GET' },
    })
  }

  const url = new URL(request.url)
  const target = sepTargetFromSearchParams(url.searchParams)
  if (!target) {
    return new Response(JSON.stringify({ error: 'SEP entry is required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const response = await fetch(sepEntryUrl(target), {
    headers: { Accept: 'text/html', 'User-Agent': PREVIEW_USER_AGENT },
    cf: { cacheTtl: 86400, cacheEverything: true },
  })

  if (!response.ok) {
    return new Response(JSON.stringify({ error: 'SEP preview unavailable' }), {
      status: response.status === 404 ? 404 : 502,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const preview = readSepPreviewHtml(await response.text(), target)
  if (!preview) {
    return new Response(JSON.stringify({ error: 'SEP preview unavailable' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  return new Response(JSON.stringify(preview), { headers: { 'Content-Type': 'application/json' } })
}
