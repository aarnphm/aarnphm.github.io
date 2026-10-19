import { type Element } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { h } from 'hastscript'
import katex from 'katex'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'

/** Protect TeX from Pele's markup parser and typeset it once, on the server. */
export function diagramMath(source: string): Element | undefined {
  const labels: Element[] = []
  let prefix = '\uE000'
  while (source.includes(prefix)) prefix += '\uE000'
  const prepared = source.replace(/(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$/g, (_, tex: string) => {
    const token = `${prefix}${labels.length}\uE001`
    const html = renderFigureLabel(tex, math =>
      katex.renderToString(math, {
        ...katexOptions,
        macros: { ...customMacros },
        displayMode: false,
        output: 'htmlAndMathml',
        strict: false,
        throwOnError: false,
        trust: false,
      }),
    )
    const root = fromHtml(html, { fragment: true })
    labels.push(
      h(
        'span.pele-math-label',
        { 'data-token': token },
        root.children.filter(n => n.type !== 'doctype'),
      ),
    )
    return token
  })
  if (labels.length === 0) return
  return h(
    'div.pele-math-cache',
    { hidden: true, 'data-source': prepared, 'data-prefix': prefix },
    labels,
  )
}
