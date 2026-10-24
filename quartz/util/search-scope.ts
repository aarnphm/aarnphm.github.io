import { CODE_FILE_EXTENSIONS, languageFromExt } from './code-languages'
import { isCourseMaterialPath } from './image-search-scope'

export type CodeSearchScope = 'code' | `code:${string}`
export type SearchScope = 'notes' | 'images' | 'pdf' | 'all' | CodeSearchScope
export type SearchQuery = { scope: SearchScope; term: string; explicit: boolean }

type SearchSource = { slug: string; fileName: string; filePath: string }

const imageExtensions = new Set([
  'avif',
  'bmp',
  'gif',
  'heic',
  'heif',
  'ico',
  'jpeg',
  'jpg',
  'jxl',
  'png',
  'svg',
  'tif',
  'tiff',
  'webp',
])
const noteExtensions = new Set(['md', 'canvas', 'base', 'bases', 'ipynb'])

function canonicalCodeLanguage(language: string): string {
  if (language === 'ts' || language === 'tsx') return 'typescript'
  if (language === 'js' || language === 'jsx') return 'javascript'
  return language
}

const codeScopes = new Map<string, CodeSearchScope>()
for (const extension of CODE_FILE_EXTENSIONS) {
  const language = languageFromExt(extension)
  if (!language) continue
  const scope: CodeSearchScope = `code:${canonicalCodeLanguage(language)}`
  codeScopes.set(extension.slice(1), scope)
  codeScopes.set(canonicalCodeLanguage(language), scope)
}
for (const [alias, language] of [
  ['python3', 'python'],
  ['c++', 'cpp'],
  ['golang', 'go'],
  ['shell', 'bash'],
  ['objc', 'objective-c'],
  ['lean4', 'lean'],
  ['webassembly', 'wasm'],
] satisfies [string, string][]) {
  codeScopes.set(alias, `code:${language}`)
}

export function isCodeSearchScope(scope: SearchScope): scope is CodeSearchScope {
  return scope === 'code' || scope.startsWith('code:')
}

function queryScope(value: string): SearchScope | undefined {
  const normalized = value.toLowerCase()
  switch (normalized) {
    case 'notes':
    case 'images':
    case 'pdf':
    case 'all':
    case 'code':
      return normalized
    default:
      return codeScopes.get(normalized.replace(/^\./, ''))
  }
}

export function parseSearchQuery(raw: string): SearchQuery {
  let scope: SearchScope = 'notes'
  let explicit = false
  const term = raw
    .replace(/(^|\s)filter:\s*(\S+)/gi, (match: string, before: string, value: string) => {
      const selected = queryScope(value)
      if (!selected) return match
      scope = selected
      explicit = true
      return before
    })
    .trim()
    .replace(/\s+/g, ' ')
  return { scope, term, explicit }
}

export function searchScopeIncludes(scope: SearchScope, source: SearchSource): boolean {
  if (scope === 'all') return true
  const path = (source.filePath || source.fileName).replace(/\\/g, '/')
  if (/^[a-z][a-z\d+.-]*:\/\//i.test(path)) return false
  const extension = path.split('/').at(-1)?.split('.').at(-1)?.toLowerCase() ?? ''
  if (scope === 'images') {
    return (
      imageExtensions.has(extension) &&
      ![source.slug, source.filePath, source.fileName].some(isCourseMaterialPath)
    )
  }
  if (scope === 'pdf') return extension === 'pdf'
  if (isCodeSearchScope(scope)) {
    if (scope === 'code' && extension === 'json') return false
    if (!CODE_FILE_EXTENSIONS.has(`.${extension}`)) return false
    const language = languageFromExt(extension)
    return (
      scope === 'code' ||
      (language !== undefined && scope === `code:${canonicalCodeLanguage(language)}`)
    )
  }
  if (source.slug.startsWith('arena/') && /(?:^|\/)are\.na\.md$/i.test(path)) return false
  return noteExtensions.has(extension)
}
