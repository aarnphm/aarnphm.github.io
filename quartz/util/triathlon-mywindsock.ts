import type { StravaActivityDetail } from '../plugins/stores/strava'
import type { TriNodeFactory } from './triathlon-card'

export const buildMyWindsockGraphs = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const reference = d.analyses.native.myWindsockArchive
  if (!reference) return null
  const section = f.el('section', 'tri-mywindsock', undefined, {
    'data-mywindsock-id': String(d.id),
    'data-mywindsock-path': reference.path,
    'data-mywindsock-captured-at': reference.capturedAt,
    'data-mywindsock-state': 'pending',
    'aria-label': 'myWindsock graphs',
  })
  const provenance = f.el('details', 'tri-mywindsock-provenance tri-mywindsock-json', undefined, {
    'data-mywindsock-provenance': '',
  })
  const captured = f.el('p', 'tri-mywindsock-source', 'Captured ')
  f.add(captured, f.el('time', undefined, reference.capturedAt, { datetime: reference.capturedAt }))
  f.add(provenance, f.el('summary', undefined, 'Provider analysis'), captured)
  f.add(
    section,
    f.el('span', 'tri-ana-block-title', 'myWindsock graphs'),
    provenance,
    f.el('div', 'tri-mywindsock-content', undefined, { 'data-mywindsock-content': '' }),
  )
  return section
}
