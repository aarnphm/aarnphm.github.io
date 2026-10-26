import type { BuildCtx } from '../../../util/ctx'
import type { ContentDetails, ContentIndexMap } from './link-index'
import { GlobalConfiguration } from '../../../cfg'
import { FilePath, FullSlug, joinSegments, SimpleSlug, simplifySlug } from '../../../util/path'
import { write } from '../helpers'

export function generateSiteMap(cfg: GlobalConfiguration, idx: ContentIndexMap): string {
  const base = cfg.baseUrl ?? ''
  const createURLEntry = (slug: SimpleSlug, content: ContentDetails): string => {
    let modifiedDate = content.date
    if (!modifiedDate && content.fileData?.frontmatter?.modified) {
      modifiedDate = new Date(content.fileData.frontmatter.modified)
    }
    const lastModified =
      modifiedDate && Number.isFinite(modifiedDate.getTime())
        ? `<lastmod>${modifiedDate.toISOString()}</lastmod>`
        : ''
    return `<url>
    <loc>https://${joinSegments(base, encodeURI(slug))}</loc>
    ${lastModified}
  </url>`
  }

  const urls = Array.from(idx)
    .filter(([, content]) => {
      const frontmatter = content.fileData?.frontmatter
      return frontmatter?.noindex !== true && frontmatter?.protected !== true
    })
    .sort(([slugA], [slugB]) => slugA.localeCompare(slugB))
    .map(([slug, content]) => createURLEntry(simplifySlug(slug), content))
    .join('')
  return `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">${urls}</urlset>`
}

export async function* writeSecurityFiles(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
): AsyncGenerator<FilePath> {
  const baseDomain = cfg.baseUrl ?? 'aarnphm.xyz'
  const securityPolicyEntry =
    linkIndex.get('security-policy' as FullSlug) ??
    Array.from(linkIndex.values()).find(details => {
      const normalizedFile = details.fileName.replace(/\\/g, '/')
      return (
        normalizedFile === ('content/security policy' as FilePath) ||
        normalizedFile.endsWith('/security policy') ||
        details.slug === 'security-policy'
      )
    })

  const fallbackSlug = securityPolicyEntry
    ? simplifySlug(securityPolicyEntry.slug as FullSlug)
    : ('security-policy' as SimpleSlug)
  const policyPermalink = securityPolicyEntry?.fileData?.frontmatter?.permalinks?.[0]
  const policyHref = policyPermalink
    ? `https://${joinSegments(baseDomain, policyPermalink.replace(/^\/+/, ''))}`
    : `https://${joinSegments(baseDomain, fallbackSlug)}`

  const modifiedSource =
    securityPolicyEntry?.fileData?.frontmatter?.modified ?? securityPolicyEntry?.date?.toISOString()
  const lastModifiedDate = modifiedSource ? new Date(modifiedSource) : new Date()
  const safeLastModified = Number.isNaN(lastModifiedDate.getTime()) ? new Date() : lastModifiedDate
  const expiresDate = new Date(safeLastModified.getTime() + 1000 * 60 * 60 * 24 * 180)

  const securityTxt = `Contact: mailto:security@aarnphm.xyz
Encryption: https://${joinSegments(baseDomain, 'pgp-key.txt')}
Policy: ${policyHref}
Canonical: https://${joinSegments(baseDomain, '.well-known', 'security.txt')}
Preferred-Languages: en
Last-Modified: ${safeLastModified.toISOString()}
Expires: ${expiresDate.toISOString()}
`

  yield write({
    ctx,
    content: securityTxt,
    slug: joinSegments('.well-known', 'security') as FullSlug,
    ext: '.txt',
  })
  yield write({ ctx, content: securityTxt, slug: 'security' as FullSlug, ext: '.txt' })
}

export async function writeSiteMap(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  linkIndex: ContentIndexMap,
): Promise<FilePath> {
  return write({
    ctx,
    content: generateSiteMap(cfg, linkIndex),
    slug: 'sitemap' as FullSlug,
    ext: '.xml',
  })
}

export function robotsTxt(baseUrl: string): string {
  return `# As a condition of accessing this website, you agree to abide by the following
# content signals:

# (a)  If a content-signal = yes, you may collect content for the corresponding
#      use.
# (b)  If a content-signal = no, you may not collect content for the
#      corresponding use.
# (c)  If the website operator does not include a content signal for a
#      corresponding use, the website operator neither grants nor restricts
#      permission via content signal with respect to the corresponding use.

# The content signals and their meanings are:

# search:   building a search index and providing search results (e.g., returning
#           hyperlinks and short excerpts from your website's contents). Search does not
#           include providing AI-generated search summaries.
# ai-input: inputting content into one or more AI models (e.g., retrieval
#           augmented generation, grounding, or other real-time taking of content for
#           generative AI search answers).
# ai-train: training or fine-tuning AI models.

# ANY RESTRICTIONS EXPRESSED VIA CONTENT SIGNALS ARE EXPRESS RESERVATIONS OF
# RIGHTS UNDER ARTICLE 4 OF THE EUROPEAN UNION DIRECTIVE 2019/790 ON COPYRIGHT
# AND RELATED RIGHTS IN THE DIGITAL SINGLE MARKET.

User-Agent: *
Content-signal: search=yes,ai-train=yes,ai-input=yes
Allow: /

User-agent: PerplexityBot
Disallow: /

User-agent: Perplexity-User
Disallow: /

Sitemap: https://${joinSegments(baseUrl, 'sitemap.xml')}
`
}
