import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'

type EnvironmentEvidenceSource = 'pelotan' | 'mywindsock' | 'garden-estimate' | 'noaa-loofs'

interface EnvironmentEvidenceValue {
  text: string
  source: EnvironmentEvidenceSource
  formulaId?: string
  formulaVersion?: number
}

export interface EnvironmentEvidenceRow {
  label: string
  labelMath?: string
  detail?: string
  value: EnvironmentEvidenceValue | null
}

export interface EnvironmentEvidenceGroup {
  label: string
  rows: readonly EnvironmentEvidenceRow[]
}

export const providerEnvironmentValue = (
  text: string,
  source: Exclude<EnvironmentEvidenceSource, 'garden-estimate' | 'noaa-loofs'>,
): EnvironmentEvidenceValue => ({ text, source })

export const gardenEnvironmentValue = (
  text: string,
  model: { formulaId: string; formulaVersion: number },
): EnvironmentEvidenceValue => ({
  text,
  source: 'garden-estimate',
  formulaId: model.formulaId,
  formulaVersion: model.formulaVersion,
})

export const surfaceCurrentEnvironmentValue = (
  text: string,
  model: { formulaId: string; formulaVersion: number },
): EnvironmentEvidenceValue => ({
  text,
  source: 'noaa-loofs',
  formulaId: model.formulaId,
  formulaVersion: model.formulaVersion,
})

export const environmentTable = <N>(
  f: TriNodeFactory<N>,
  groups: readonly EnvironmentEvidenceGroup[],
): N => {
  const table = f.el('table', 'tri-environment-table tri-environment-summary-table')
  for (const group of groups) {
    const tbody = f.el('tbody')
    const heading = f.el('tr', 'tri-environment-table-group')
    f.add(
      heading,
      f.el('th', undefined, triText(f.presentation.locale, group.label), {
        colspan: '2',
        scope: 'rowgroup',
        'data-i18n': group.label,
      }),
    )
    f.add(tbody, heading)
    for (const row of group.rows) {
      const rendered = f.el('tr')
      const label = triText(f.presentation.locale, row.label)
      const labelAttrs: Record<string, string> = {
        scope: 'row',
        ...(row.labelMath
          ? { 'aria-label': label, 'data-i18n-aria-label': row.label }
          : { 'data-i18n': row.label }),
        ...(row.detail
          ? {
              'data-gloss': '',
              'data-gloss-def': triText(f.presentation.locale, row.detail),
              tabindex: '0',
            }
          : {}),
      }
      const labelCell = f.el(
        'th',
        'tri-environment-row-label',
        row.labelMath ? undefined : label,
        labelAttrs,
      )
      if (row.labelMath) f.add(labelCell, f.math('tri-environment-row-math', row.labelMath))
      if (row.value == null) {
        f.add(
          rendered,
          labelCell,
          f.el('td', 'tri-environment-unavailable', '—', { 'aria-label': 'unavailable' }),
        )
        f.add(tbody, rendered)
        continue
      }
      const sourceLabel =
        row.value.source === 'garden-estimate'
          ? triText(f.presentation.locale, 'Garden estimate')
          : row.value.source === 'pelotan'
            ? 'Pelotan'
            : row.value.source === 'mywindsock'
              ? 'MyWindsock'
              : 'NOAA LOOFS'
      const sourceAttrs: Record<string, string> =
        row.value.source === 'garden-estimate'
          ? {
              'data-analysis-source': 'garden-estimate',
              'data-analysis-formula': row.value.formulaId ?? 'garden-estimate',
              'data-analysis-formula-version': String(row.value.formulaVersion ?? 1),
              'data-gloss': '',
              'data-gloss-def': triText(f.presentation.locale, 'estimate'),
              tabindex: '0',
              'aria-label': `${row.value.text}, ${triText(f.presentation.locale, 'estimate')}`,
            }
          : row.value.source === 'noaa-loofs'
            ? {
                'data-analysis-source': 'provider-modeled',
                'data-analysis-provider': 'noaa-loofs',
                'data-analysis-formula': row.value.formulaId ?? 'garden-surface-current-v1',
                'data-analysis-formula-version': String(row.value.formulaVersion ?? 1),
                'data-gloss': '',
                'data-gloss-def': triText(f.presentation.locale, 'Surface current speed detail'),
                tabindex: '0',
                'aria-label': `${row.value.text}, ${triText(f.presentation.locale, 'provider-modeled')} ${sourceLabel}. ${triText(f.presentation.locale, 'Surface current speed detail')}`,
              }
            : {
                'data-analysis-source': 'provider-native',
                'data-analysis-provider': row.value.source,
                'aria-label': `${row.value.text}, provider-native ${sourceLabel} report`,
              }
      f.add(
        rendered,
        labelCell,
        f.el(
          'td',
          row.value.source === 'garden-estimate' ? 'tri-environment-estimate' : undefined,
          row.value.text,
          sourceAttrs,
        ),
      )
      f.add(tbody, rendered)
    }
    f.add(table, tbody)
  }
  return table
}

export const environmentLink = <N>(f: TriNodeFactory<N>, text: string, href: string): N =>
  f.el('a', undefined, text, { href, target: '_blank', rel: 'noreferrer' })

export const environmentLogoLink = <N>(
  f: TriNodeFactory<N>,
  cls: string,
  src: string,
  alt: string,
  href: string,
  title: string,
): N => {
  const link = f.el('a', `tri-environment-provider-logo ${cls}`, undefined, {
    href,
    target: '_blank',
    rel: 'noreferrer',
    title,
    'aria-label': title,
  })
  f.add(link, f.el('img', undefined, undefined, { src, alt, loading: 'lazy', decoding: 'async' }))
  return link
}
