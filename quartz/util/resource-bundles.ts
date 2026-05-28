import { CSSResource, JSResource } from './resources'

export type CssBundlePart =
  | { type: 'bundle'; content: string }
  | { type: 'resource'; resource: CSSResource }

export type JsBundlePart =
  | { type: 'bundle'; scripts: string[]; loadTime: JSResource['loadTime'] }
  | { type: 'resource'; resource: JSResource }

export const componentCssResourceKeyPrefix = 'component-css:'

export const componentCssBundleKey = `${componentCssResourceKeyPrefix}bundle`

export const staticCssBundleSlug = 'static/resource-style'

export const staticCssBundleKey = (content: string) => `resource-css:${content}`

export const staticJsBundleSlug = (loadTime: JSResource['loadTime']) =>
  `static/resource-${loadTime === 'beforeDOMReady' ? 'before' : 'after'}`

export const staticJsBundleKey = (loadTime: JSResource['loadTime'], scripts: string[]) =>
  `resource-js:${loadTime}:${JSON.stringify(scripts)}`

export function splitCssBundles(resources: CSSResource[], leadingInline: string[] = []) {
  const parts: CssBundlePart[] = []
  for (const content of leadingInline) {
    if (content.length > 0) parts.push({ type: 'bundle', content })
  }

  for (const resource of resources) {
    if (resource.inline ?? false) {
      if (resource.content.length > 0) parts.push({ type: 'bundle', content: resource.content })
    } else {
      parts.push({ type: 'resource', resource })
    }
  }

  return parts
}

export function splitJsBundles(
  resources: JSResource[],
  loadTime: JSResource['loadTime'],
  leadingInline: string[] = [],
) {
  const parts: JsBundlePart[] = []
  for (const script of leadingInline) {
    if (script.length > 0) parts.push({ type: 'bundle', scripts: [script], loadTime })
  }

  for (const resource of resources) {
    if (resource.loadTime !== loadTime) continue
    if (resource.contentType === 'inline') {
      if (resource.script.length > 0)
        parts.push({ type: 'bundle', scripts: [resource.script], loadTime })
    } else {
      parts.push({ type: 'resource', resource })
    }
  }

  return parts
}
