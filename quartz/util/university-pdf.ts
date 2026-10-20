import { slugifyFilePath, type FilePath } from './path'

/** A hosted PDF under `thoughts/university/<term>/<course>/`, as a manifest slug. */
const pdfRe = /^thoughts\/university\/([^/]+)\/([^/]+)\/(.+)\.pdf$/i
/** A term folder or a course folder under `thoughts/university`. */
const homeRe = /^thoughts\/university\/([^/]+)(?:\/([^/]+))?$/
const pageRe = /^thoughts\/university\/[^/]+\/[^/]+\/pdf\/.+$/

export interface UniversityPdf {
  term: string
  course: string
  /** Path of the PDF inside the course folder, without the extension. */
  rel: string
  /** Slug of the reader page that wraps this PDF. */
  pageSlug: string
}

/** The term and course of a manifest PDF slug, else null. */
export function universityPdfOf(pdfSlug: string): UniversityPdf | null {
  const match = pdfRe.exec(pdfSlug)
  if (!match) return null
  const [, term, course, rel] = match
  return { term, course, rel, pageSlug: `thoughts/university/${term}/${course}/pdf/${rel}` }
}

/**
 * Reader page of a PDF that a link names by its URL path. The pages sit under `<course>/pdf/` so
 * they never collide with a note, and the PDF file keeps its own path.
 */
export function universityPdfPagePath(pathname: string): string | null {
  let decoded: string
  try {
    decoded = decodeURIComponent(pathname)
  } catch {
    return null
  }
  const pdf = universityPdfOf(slugifyFilePath(decoded.replace(/^\/+/, '') as FilePath))
  return pdf ? `/${pdf.pageSlug}/` : null
}

export function isUniversityPdfPage(slug: string): boolean {
  return pageRe.test(slug)
}

/**
 * Index and course folders that open as a course card: the archive root, a term folder and a
 * course folder. A path ending in `/index` names its folder.
 */
export function universityHomePath(pathname: string): string | null {
  const path = pathname
    .replace(/^\/+/, '')
    .replace(/\/+$/, '')
    .replace(/\/index$/, '')
  if (path === 'thoughts/university') return '/thoughts/university/'
  const match = homeRe.exec(path)
  if (!match) return null
  return `/thoughts/university/${match[1]}/${match[2] ? `${match[2]}/` : ''}`
}

/** What the strip above a university PDF reader needs, resolved by the emitter. */
export interface UniversityPdfPage {
  term: string
  course: string
  title: string
  bytes: number
  /** Slugs of the term index and the course home when those notes exist. */
  termHome?: string
  home?: string
  homeTitle?: string
  /** Neighbours among the course's PDFs, in path order. */
  prev?: { title: string; path: string }
  next?: { title: string; path: string }
  position: [number, number]
  citedBy: { from: string; title: string }[]
  /** Path of the PDF file itself. */
  rawPath: string
}
