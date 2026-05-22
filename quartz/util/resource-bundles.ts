import { CSSResource, JSResource } from './resources'

export type CssBundlePart =
  | { type: 'bundle'; content: string; index: number }
  | { type: 'resource'; resource: CSSResource }

export type JsBundlePart =
  | { type: 'bundle'; scripts: string[]; loadTime: JSResource['loadTime']; index: number }
  | { type: 'resource'; resource: JSResource }

export const staticCssBundleSlug = (index: number) => `static/resource-style-${index}`

export const staticCssBundlePath = (index: number) => `${staticCssBundleSlug(index)}.css`

export const staticJsBundleSlug = (loadTime: JSResource['loadTime'], index: number) =>
  `static/resource-${loadTime === 'beforeDOMReady' ? 'before' : 'after'}-${index}`

export const staticJsBundlePath = (loadTime: JSResource['loadTime'], index: number) =>
  `${staticJsBundleSlug(loadTime, index)}.js`

export function splitCssBundles(resources: CSSResource[], leadingInline: string[] = []) {
  const parts: CssBundlePart[] = []
  let pending = leadingInline.filter(content => content.length > 0)
  let index = 0
  const flush = () => {
    if (pending.length === 0) return
    parts.push({ type: 'bundle', content: pending.join('\n'), index })
    pending = []
    index += 1
  }

  for (const resource of resources) {
    if (resource.inline ?? false) {
      pending.push(resource.content)
    } else {
      flush()
      parts.push({ type: 'resource', resource })
    }
  }
  flush()

  return parts
}

export function splitJsBundles(
  resources: JSResource[],
  loadTime: JSResource['loadTime'],
  leadingInline: string[] = [],
) {
  const parts: JsBundlePart[] = []
  let pending = leadingInline.filter(script => script.length > 0)
  let index = 0
  const flush = () => {
    if (pending.length === 0) return
    parts.push({ type: 'bundle', scripts: pending, loadTime, index })
    pending = []
    index += 1
  }

  for (const resource of resources) {
    if (resource.loadTime !== loadTime) continue
    if (resource.contentType === 'inline') {
      pending.push(resource.script)
    } else {
      flush()
      parts.push({ type: 'resource', resource })
    }
  }
  flush()

  return parts
}
