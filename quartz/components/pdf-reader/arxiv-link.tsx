import { arenaArxivPdfUrl } from '../../util/arena-embed'
import { EXTERNAL_LINK_ICON_PATH } from '../../util/external-link-icon'

export function ArxivLink({ slug }: { slug: string }) {
  const parts = slug.split('/')
  const filename = parts.at(-1) ?? ''
  const href =
    arenaArxivPdfUrl(`https://arxiv.org/pdf/${filename}`) ??
    arenaArxivPdfUrl(`https://arxiv.org/pdf/${parts.slice(-2).join('/')}`)
  if (!href) return null

  return (
    <a
      class="pdf-reader-download"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Open PDF on arXiv (new tab)"
      title="Open PDF on arXiv"
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 15 15"
        fill="currentColor"
        aria-hidden="true"
        focusable="false"
      >
        <path d={EXTERNAL_LINK_ICON_PATH} />
      </svg>
    </a>
  )
}
