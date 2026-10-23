import { isCourseMaterialPath } from '../../util/image-search-scope'
import { searchScopeIncludes } from '../../util/search-scope'

export function searchableImage(slug: string): boolean {
  return (
    !isCourseMaterialPath(slug) &&
    searchScopeIncludes('images', { slug, fileName: slug, filePath: slug })
  )
}

export function imageSimilarityButton(
  slug: string,
  onSelect: (slug: string) => void,
  similarity?: number,
): HTMLButtonElement {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'image-similar-button'
  const scored = similarity !== undefined && Number.isFinite(similarity)
  button.textContent = scored ? `${(similarity * 100).toFixed(1)}%` : 'similar images'
  button.title = scored
    ? `Cosine similarity: ${similarity.toFixed(3)}. Find similar images`
    : 'Find similar images'
  button.setAttribute('aria-label', `Find images similar to ${slug.split('/').at(-1)}`)
  button.addEventListener('click', event => {
    event.stopPropagation()
    onSelect(slug)
  })
  return button
}

export function imageSearchRow(
  anchor: HTMLAnchorElement,
  slug: string,
  onSelect: (slug: string) => void,
  similarity?: number,
): HTMLElement {
  const row = document.createElement('div')
  row.className = 'image-search-row'
  const thumbnail = document.createElement('img')
  thumbnail.className = 'image-search-thumbnail'
  thumbnail.src = anchor.href
  thumbnail.alt = ''
  thumbnail.loading = 'lazy'
  thumbnail.decoding = 'async'
  anchor.prepend(thumbnail)
  if (similarity !== undefined && Number.isFinite(similarity)) {
    const score = document.createElement('span')
    score.className = 'result-likelihood'
    score.title = 'cosine similarity'
    score.textContent = similarity.toFixed(3)
    anchor.append(score)
  }
  row.append(anchor, imageSimilarityButton(slug, onSelect, similarity))
  return row
}

export function imageSearchPreview(
  slug: string,
  url: string,
  onSelect: (slug: string) => void,
): HTMLElement {
  const preview = document.createElement('div')
  preview.className = 'preview-inner image-search-preview'
  const image = document.createElement('img')
  image.src = url
  image.alt = slug.split('/').at(-1) ?? slug
  const open = document.createElement('a')
  open.href = url
  open.textContent = 'open image'
  const actions = document.createElement('div')
  actions.append(open, imageSimilarityButton(slug, onSelect))
  preview.append(image, actions)
  return preview
}

export function imageSimilarityStatus(message: string): HTMLParagraphElement {
  const status = document.createElement('p')
  status.className = 'image-search-status'
  status.setAttribute('role', 'status')
  status.textContent = message
  return status
}
