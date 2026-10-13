import { classPrefixSelector } from '../../util/page-styles'

const loads = new Map<string, Promise<void>>()
// A page from an emitter without page resources declares no registry.
const styles = (typeof pageStyles === 'undefined' ? [] : pageStyles).map(style => ({
  href: style.href,
  selector: classPrefixSelector(style.classPrefixes),
}))

const linked = (href: string) =>
  [...document.head.querySelectorAll<HTMLLinkElement>('link[rel="stylesheet"]')].some(
    link => link.href === href,
  )

function load(href: string): Promise<void> {
  let loading = loads.get(href)
  if (loading) return loading
  if (linked(href)) {
    loading = Promise.resolve()
  } else {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = href
    link.dataset.persist = 'true'
    loading = new Promise(resolve => {
      link.addEventListener('load', () => resolve(), { once: true })
      link.addEventListener('error', () => resolve(), { once: true })
    })
    document.head.append(link)
  }
  loads.set(href, loading)
  return loading
}

export const pageStylesPending = () => styles.some(({ href }) => !loads.has(href))

/** Links each page-scoped sheet that markup under `root` uses; resolves once they apply. */
export function loadPageStyles(root: ParentNode): Promise<void> {
  const pending = styles.filter(
    ({ href, selector }) =>
      !loads.has(href) &&
      ((root instanceof Element && root.matches(selector)) || root.querySelector(selector)),
  )
  return Promise.all(pending.map(({ href }) => load(href))).then(() => {})
}
