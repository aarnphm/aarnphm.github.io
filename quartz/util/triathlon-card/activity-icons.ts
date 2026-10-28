import type { TriNodeFactory } from './construction-contracts'
import {
  SPORT_ICON,
  type ActivityKind,
  type StravaActivityDetail,
} from '../../plugins/stores/strava'
import { triText } from '../triathlon-i18n'

const BATTERY = [
  'M23 10V14',
  'M1 16V8C1 6.89543 1.89543 6 3 6H18C19.1046 6 20 6.89543 20 8V16C20 17.1046 19.1046 18 18 18H3C1.89543 18 1 17.1046 1 16Z',
  'M10.1667 9L8.5 12H12.5L10.8333 15',
]

export const buildIcon = <N>(f: TriNodeFactory<N>, sport: ActivityKind): N => {
  const icon = f.svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of SPORT_ICON[sport]) f.add(icon, f.svg('path', { d }))
  return icon
}

export const buildActivityIcon = <N>(f: TriNodeFactory<N>, activity: StravaActivityDetail): N => {
  const icon = buildIcon(f, activity.sport)
  const stravaId = activity.sources
    ? activity.sources.find(source => source.provider === 'strava')?.activityId
    : activity.sauna
      ? null
      : String(activity.id)
  if (!stravaId || !/^[1-9]\d*$/.test(stravaId)) return icon
  const link = f.el('a', 'tri-act-icon-link', undefined, {
    href: `https://www.strava.com/activities/${stravaId}`,
    target: '_blank',
    rel: 'noopener noreferrer',
    'aria-label': `${activity.name || triText(f.presentation.locale, activity.sport)} · Strava`,
    'data-site-cursor-action': '',
  })
  f.add(link, icon)
  return link
}

export const buildBattery = <N>(f: TriNodeFactory<N>): N => {
  const icon = f.svg('svg', { class: 'tri-ico tri-battery', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of BATTERY) f.add(icon, f.svg('path', { d }))
  return icon
}

export const buildRestStatus = <N>(f: TriNodeFactory<N>): N => {
  const rest = f.el('div', 'tri-pop-rest')
  f.add(
    rest,
    buildBattery(f),
    f.el('span', 'tri-pop-rest-label', triText(f.presentation.locale, 'rest'), {
      'data-i18n': 'rest',
    }),
  )
  return rest
}

export const LAYERS_ICON = [
  'M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z',
  'm22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65',
  'm22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65',
]

export const buildLayers = <N>(f: TriNodeFactory<N>): N => {
  const icon = f.svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of LAYERS_ICON) f.add(icon, f.svg('path', { d }))
  return icon
}
