import type { StravaActivityDetail } from '../plugins/stores/strava'
import type { TriNodeFactory } from './triathlon-card'
import { triText } from './triathlon-i18n'

export const myWindsockLogoLink = <N>(
  f: Pick<TriNodeFactory<N>, 'el' | 'add'>,
  activityId: number,
  cls: string,
): N => {
  const link = f.el('a', cls, undefined, {
    href: `https://mywindsock.com/activity/${activityId}/`,
    target: '_blank',
    rel: 'noreferrer',
    title: 'Powered by myWindsock',
    'aria-label': 'myWindsock activity analysis',
  })
  f.add(
    link,
    f.el('img', undefined, undefined, {
      src: '/static/triathlon/mywindsock.svg',
      alt: 'myWindsock',
      width: '600',
      height: '265',
      loading: 'lazy',
      decoding: 'async',
    }),
  )
  return link
}

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
    'data-mywindsock-elapsed-s': String(d.elapsedTimeS),
    'data-mywindsock-state': 'pending',
    'aria-label': triText(f.presentation.locale, 'wind graphs'),
    'data-i18n-aria-label': 'wind graphs',
  })
  const head = f.el('div', 'tri-mywindsock-head')
  f.add(head, f.el('div', 'tri-mywindsock-picker', undefined, { 'data-mywindsock-picker': '' }))
  const credit = f.el('div', 'tri-environment-attribution tri-mywindsock-credit')
  f.add(
    credit,
    myWindsockLogoLink(f, d.id, 'tri-environment-provider-logo tri-environment-mywindsock-logo'),
  )
  f.add(
    section,
    head,
    f.el('div', 'tri-mywindsock-content', undefined, { 'data-mywindsock-content': '' }),
    credit,
  )
  return section
}
