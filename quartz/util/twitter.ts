export function parseTwitterPostUrl(href: string): string | null {
  let url: URL
  try {
    url = new URL(href)
  } catch {
    return null
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    !/^(?:(?:www|mobile|m)\.)?(?:twitter|x)\.com$/.test(url.hostname)
  )
    return null
  const post = url.pathname.match(/^\/(?:[\w]+\/status|i\/web\/status)\/(\d{1,20})(?:\/|$)/)
  if (!post) return null
  return `https://twitter.com/i/status/${post[1]}`
}
