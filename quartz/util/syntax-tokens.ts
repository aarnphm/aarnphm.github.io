import { isSupportedLanguage, type Lang } from '@pierre/highlights'

// Token styles become --syntax-light/--syntax-dark pairs that syntax.scss switches by theme.
export const SYNTAX_TOKEN_OPTIONS = { defaultColor: false, cssVariablePrefix: '--syntax-' } as const

export function syntaxLanguage(label: string): Lang {
  const requested = label.toLowerCase() === 'mojo' ? 'python' : label
  return isSupportedLanguage(requested) ? requested : 'plain'
}

// Preserve Garden's palette overrides for the bundled GitHub theme pair.
const roles: Record<string, string> = {
  '#6f42c1/#b392f0': 'value',
  '#005cc5/#79b8ff': 'function',
  '#032f62/#9ecbff': 'string',
  '#032f62/#dbedff': 'property',
  '#24292e/#e1e4e8': 'normal',
  '#586069/#d1d5da': 'punctuation',
  '#f6f8fa/#2f363d': 'comment',
  '#6a737d/#6a737d': 'comment',
  '#22863a/#85e89d': 'tag',
  '#e36209/#ffab70': 'important',
  '#b31d28/#fdaeb7': 'operator',
  '#d73a49/#f97583': 'keyword',
}

export function syntaxRole(styles: Record<string, string>): string | undefined {
  const light = styles['--syntax-light']?.toLowerCase()
  const dark = styles['--syntax-dark']?.toLowerCase()
  return roles[`${light}/${dark}`]
}
