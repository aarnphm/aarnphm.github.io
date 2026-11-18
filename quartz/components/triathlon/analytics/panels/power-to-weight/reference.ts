import type { Analytics } from '../../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../../runtime/context'
import { el } from '../../../runtime/dom'
import { powerToWeightDurationLabel } from './model'

export const buildReferenceTable = (
  data: Analytics,
  context: TriathlonContext,
): HTMLElement | null => {
  const trend = data.powerCurve.powerToWeight
  if (trend.references.length === 0 || trend.ageGroup == null) return null
  const wrap = el('div', 'tri-power-weight-reference')
  const table = el('table', 'tri-power-weight-reference-table')
  const caption = el('caption')
  caption.append(
    el('span', undefined, `circa Zwift (${trend.ageGroup.replace('–', '-')}) — `),
    el('a', undefined, 'video', { href: trend.source.url, rel: 'noreferrer' }),
  )
  const head = el('thead')
  const header = el('tr')
  for (const label of [
    context.formatter.text('duration'),
    'P10',
    context.formatter.text('average'),
    'P90',
  ])
    header.appendChild(el('th', undefined, label, { scope: 'col' }))
  head.appendChild(header)
  const body = el('tbody')
  for (const reference of trend.references) {
    const row = el('tr')
    row.append(
      el('th', undefined, powerToWeightDurationLabel(reference.durationS), { scope: 'row' }),
      el('td', undefined, reference.p10.toFixed(2)),
      el('td', undefined, reference.average.toFixed(2)),
      el('td', undefined, reference.p90.toFixed(2)),
    )
    body.appendChild(row)
  }
  table.append(caption, head, body)
  wrap.appendChild(table)
  return wrap
}
