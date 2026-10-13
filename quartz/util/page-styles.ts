// A page-scoped stylesheet styles only markup whose class tokens start with one of its prefixes.
// The server links it on pages that render such markup; the browser loads it when such markup
// arrives later (SPA navigation, popovers, previews).
export type PageStyle = { href: string; classPrefixes: string[] }

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export function classPrefixPattern(prefixes: readonly string[]): RegExp {
  return new RegExp(`class="(?:[^"]*\\s)?(?:${prefixes.map(escapeRegExp).join('|')})`)
}

export function classPrefixSelector(prefixes: readonly string[]): string {
  return prefixes.map(prefix => `[class^="${prefix}"], [class*=" ${prefix}"]`).join(', ')
}
