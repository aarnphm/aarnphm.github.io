import type { Element } from 'hast'
import { h } from 'hastscript'

/** Keep relative dates unresolved until the page is opened in the reader's timezone. */
export function transcludeCurius(
  node: Element,
  slug: string,
  fragments: readonly string[],
  inner: Element,
  alias: string,
): boolean {
  if (slug !== 'curius' || fragments.length !== 1) return false
  const date = fragments[0]
  node.children = [
    h(
      'section.curius-date-view',
      {
        'data-curius-date': date,
        'data-curius-label': alias && !alias.startsWith('#') ? alias : undefined,
        'aria-busy': 'true',
      },
      [
        h(
          'a.internal',
          { href: inner.properties.href, 'data-no-popover': true },
          alias || `curius · ${date}`,
        ),
        h('p', { role: 'status' }, 'loading saved links…'),
        h('progress', { 'aria-label': 'loading saved links' }),
      ],
    ),
  ]
  return true
}
