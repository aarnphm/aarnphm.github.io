import {
  parseActivityCardList,
  type ActivityCardData,
  type ActivityCardStatKey,
} from '../../util/press-kit-activity'
import { localizeActivity } from './press-kit-i18n'
import { activityCardFromElement } from './press-kit-metrics'

export interface BlockDisplay {
  title: boolean
  date: boolean
  metrics: boolean
}

export const isBlockDisplayKey = (key: string): key is keyof BlockDisplay =>
  key === 'title' || key === 'date' || key === 'metrics'

export interface TrainingCardRow {
  key?: ActivityCardStatKey
  label: string
  values: string[]
}

export interface TrainingCardGraphic {
  svg: string
}

export interface TrainingCardBlock {
  id: string
  kind: 'activity' | 'comparison' | 'calculator' | 'memo' | 'entry'
  title: string
  url: string
  activities: ActivityCardData[]
  rows: TrainingCardRow[]
  graphic?: TrainingCardGraphic
  text?: string
  date?: string
  reading?: string
  display?: BlockDisplay
}

export interface DayCardData {
  blocks: TrainingCardBlock[]
}

export function cardBlockDisplay(
  block: TrainingCardBlock,
  show: { title: boolean; date: boolean; metrics?: boolean },
): BlockDisplay {
  return block.display ?? { title: show.title, date: show.date, metrics: show.metrics !== false }
}

export function blockDisplayKeys(block: TrainingCardBlock): (keyof BlockDisplay)[] {
  const keys: (keyof BlockDisplay)[] = ['title']
  if (block.date || block.activities.some(activity => activity.date)) keys.push('date')
  if (
    block.reading ||
    block.rows.length ||
    block.activities.some(activity => activity.stats.length)
  )
    keys.push('metrics')
  return keys
}

const activityId = (activity: ActivityCardData): string =>
  activity.url.split('#tri-activity-')[1] ?? activity.url

// A standalone SVG cannot inherit the page stylesheet. Freeze the visible graph's paint.
function graphicOf(source: SVGSVGElement | null, prefix: string): TrainingCardGraphic | undefined {
  if (!source) return undefined
  const box = source.viewBox.baseVal
  if (box.width <= 0 || box.height <= 0) return undefined
  const copy = source.cloneNode(true)
  if (!(copy instanceof SVGSVGElement)) return undefined
  const originals = [source, ...source.querySelectorAll('*')]
  const copies = [copy, ...copy.querySelectorAll('*')]
  originals.forEach((node, index) => {
    const target = copies[index]
    const style = getComputedStyle(node)
    for (const name of ['fill', 'stroke', 'stroke-width', 'opacity', 'font-size', 'font-weight'])
      target.setAttribute(name, style.getPropertyValue(name))
    if (style.display === 'none') target.remove()
    for (const attribute of Array.from(target.attributes)) {
      if (/^on/i.test(attribute.name) || attribute.name.endsWith('href'))
        target.removeAttribute(attribute.name)
    }
    target.removeAttribute('style')
    target.removeAttribute('class')
  })
  copy
    .querySelectorAll('script, foreignObject, image, use, a, [data-compare-cursor]')
    .forEach(node => node.remove())
  for (const node of copy.querySelectorAll('[id]')) {
    const id = node.id
    node.id = `${prefix}-${id}`
    for (const reference of copy.querySelectorAll('[clip-path], [fill], [stroke]')) {
      for (const name of ['clip-path', 'fill', 'stroke']) {
        const value = reference.getAttribute(name)
        if (value?.includes(`url(#${id})`))
          reference.setAttribute(name, value.replace(`url(#${id})`, `url(#${node.id})`))
      }
    }
  }
  copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  return { svg: copy.outerHTML }
}

export function dayCardData(entry: HTMLElement): DayCardData | undefined {
  const body = entry.querySelector('.stream-entry-body')
  if (!body || !body.querySelector('.tri-day-embed, .tri-compare-embed, .tri-calc-embed'))
    return undefined
  const excluded = new Set(
    Array.from(body.querySelectorAll<HTMLElement>('[data-triathlon-filter]')).flatMap(node =>
      (node.dataset.triathlonFilter ?? '').split(/[&,\s]+/).filter(Boolean),
    ),
  )
  const blocks: TrainingCardBlock[] = []
  const seen = new Set<string>()
  const sources = new Set<string>()
  const add = (block: Omit<TrainingCardBlock, 'id'>): void => {
    if (sources.has(block.url)) return
    sources.add(block.url)
    blocks.push({ ...block, id: `block-${blocks.length}` })
  }
  for (const embed of body.querySelectorAll<HTMLElement>(
    '.tri-day-embed, .tri-compare-embed, .tri-calc-embed, figure.audio-player',
  )) {
    if (embed.matches('.tri-day-embed')) {
      for (const button of embed.querySelectorAll<HTMLElement>('[data-press-kit-activity]')) {
        const activity = activityCardFromElement(button)
        if (!activity || excluded.has(activityId(activity)) || seen.has(activity.url)) continue
        seen.add(activity.url)
        add({
          kind: 'activity',
          title: activity.title,
          url: activity.url,
          activities: [activity],
          rows: [],
        })
      }
    } else if (embed.matches('.tri-compare-embed')) {
      const comparison = embed.querySelector<HTMLElement>('[data-press-kit-activities]')
      const all = parseActivityCardList(comparison?.dataset.pressKitActivities).map(activity =>
        localizeActivity(activity),
      )
      const activities = all.filter(activity => !excluded.has(activityId(activity)))
      if (activities.length < 2) continue
      const keys: Record<string, ActivityCardStatKey> = {
        distance: 'distance',
        'moving-time': 'time',
        speed: 'pace',
        'elevation-gain': 'elevation',
        'average-hr': 'heartRate',
        'average-power': 'power',
      }
      const rows = Array.from(
        embed.querySelectorAll<HTMLElement>('.tri-compare-stats-row[data-stat]'),
      )
        .filter(row => row.dataset.stat && keys[row.dataset.stat])
        .map(row => ({
          key: keys[row.dataset.stat ?? ''],
          label:
            row.querySelector('.tri-compare-stats-label-text')?.getAttribute('aria-label') ??
            row.querySelector('.tri-compare-stats-label-text')?.textContent?.trim() ??
            '',
          values: Array.from(row.querySelectorAll('td')).map(
            cell =>
              cell.querySelector('.tri-compare-stats-value')?.getAttribute('aria-label') ??
              cell.querySelector('.tri-compare-stats-value')?.textContent?.trim() ??
              '',
          ),
        }))
      // A graph containing an excluded recording must never leak it into the export.
      const graphic =
        activities.length === all.length
          ? graphicOf(
              embed.querySelector<SVGSVGElement>('svg.tri-compare-graph'),
              `pk-${entry.id}-${blocks.length}`,
            )
          : undefined
      add({
        kind: 'comparison',
        title: `comparaison · ${[...new Set(activities.map(activity => activity.date))].join(' / ')}`,
        url: `/triathlon#${embed.dataset.compareAnchor}`,
        activities,
        rows: activities.length === all.length ? rows : [],
        graphic,
      })
    } else if (embed.matches('.tri-calc-embed')) {
      add({
        kind: 'calculator',
        title: `${embed.querySelector('.tri-calc-card-dist')?.textContent?.trim() ?? 'triathlon'} · estimation`,
        url: `/triathlon#${embed.dataset.calcHash}`,
        activities: [],
        rows: Array.from(embed.querySelectorAll('.tri-calc-card-row')).map(row => ({
          label: row.querySelector('th')?.textContent?.trim() ?? '',
          values: Array.from(row.querySelectorAll('td'))
            .map(cell => cell.textContent?.trim() ?? '')
            .filter(Boolean),
        })),
      })
    } else {
      const audio = embed.querySelector<HTMLAudioElement>('audio')
      const url = audio?.getAttribute('src') ?? audio?.querySelector('source')?.getAttribute('src')
      if (!url) continue
      add({
        kind: 'memo',
        title: 'mémo audio',
        url: new URL(url, location.href).href,
        activities: [],
        rows: [],
      })
    }
  }
  return blocks.length ? { blocks } : undefined
}
