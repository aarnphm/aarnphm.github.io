import isAbsoluteUrl from 'is-absolute-url'
import { FullSlug, TransformOptions, transformLink } from './path'

function isRootRelativeUrl(value: string): boolean {
  return value.startsWith('/')
}

function isExplicitRelativeUrl(value: string): boolean {
  return value.startsWith('./') || value.startsWith('../')
}

function pathExtension(value: string): string {
  const [pathPart] = value.split(/[?#]/, 1)
  const filename = (pathPart ?? '').split('/').pop() ?? ''
  const dot = filename.lastIndexOf('.')
  return dot >= 0 ? filename.slice(dot).toLowerCase() : ''
}

export function transformResourceUrl(
  fileSlug: FullSlug,
  value: string,
  transformOptions: TransformOptions,
) {
  if (isAbsoluteUrl(value, { httpOnly: false }) || isRootRelativeUrl(value)) {
    return value
  }

  if (pathExtension(value) === '.pdf') {
    if (isExplicitRelativeUrl(value)) return value
    return `/${value}`
  }

  return transformLink(fileSlug, value, transformOptions)
}
