import { tokenizeTerm } from '../../util/search-text'

export function fitSearchPath(element: HTMLElement, term: string): void {
  const path = element.title
  if (!path || element.clientWidth === 0) return

  const terms = tokenizeTerm(term).map(value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const pattern = terms.length ? new RegExp(`(${terms.join('|')})`, 'gi') : undefined
  const render = (value: string) => {
    const parts = pattern ? value.split(pattern) : [value]
    element.replaceChildren(
      ...parts.map((part, index) => {
        if (index % 2 === 0) return document.createTextNode(part)
        const match = document.createElement('span')
        match.className = 'highlight'
        match.textContent = part
        return match
      }),
    )
  }
  render(path)
  if (element.scrollWidth <= element.clientWidth) return

  const characters = Array.from(path)
  const filenameLength = Array.from(path.slice(path.lastIndexOf('/') + 1)).length
  const shorten = (keep: number) => {
    // Keep the filename whole when it fits, then spend the remaining space on its folders.
    const tail = Math.min(keep - 1, Math.max(Math.ceil(keep / 2), filenameLength))
    return `${characters.slice(0, keep - tail).join('')}…${characters.slice(-tail).join('')}`
  }

  let low = 2
  let high = characters.length - 1
  let fitted = '…'
  while (low <= high) {
    const keep = Math.floor((low + high) / 2)
    const candidate = shorten(keep)
    render(candidate)
    if (element.scrollWidth <= element.clientWidth) {
      fitted = candidate
      low = keep + 1
    } else {
      high = keep - 1
    }
  }
  render(fitted)
}
