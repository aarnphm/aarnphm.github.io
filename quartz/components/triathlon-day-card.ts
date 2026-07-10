import type { Element } from 'hast'
import { h, s } from 'hastscript'
import type { QuartzPluginData } from '../plugins/vfile'
import {
  buildDayCard,
  type DayCardExtras,
  type DayCardPayload,
  type DetailCtx,
  type TriNodeFactory,
} from '../util/triathlon-card'

export const triathlonCardFactory: TriNodeFactory<Element> = {
  el: (tag, cls, text, attrs) =>
    h(tag, { ...(cls ? { class: cls } : {}), ...attrs }, text === undefined ? [] : [text]),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => {
    parent.children.push(...children)
  },
}

export const triathlonDayExtras = (page: QuartzPluginData, date: string): DayCardExtras => {
  const extras: DayCardExtras = {}
  const location = page.frontmatter?.['location']
  if (typeof location === 'string' && location !== '') extras.location = location
  const day = page.tracking?.days.find(entry => entry.date === date)
  const event = day?.event ?? (day?.race ? 'race' : null)
  if (event) extras.event = event
  return extras
}

export const triathlonDayProps = (extras: DayCardExtras, date: string): Record<string, string> => {
  const props: Record<string, string> = { 'data-triathlon-date': date }
  if (extras.location) props['data-triathlon-loc'] = extras.location
  if (extras.event) props['data-triathlon-event'] = extras.event
  if (extras.sport) props['data-triathlon-sport'] = extras.sport
  if (extras.expanded) props['data-triathlon-expanded'] = '1'
  if (extras.dateHref) props['data-triathlon-date-href'] = extras.dateHref
  return props
}

export const triathlonDayCard = (
  date: string,
  payload: DayCardPayload | null,
  extras: DayCardExtras,
  ctx: DetailCtx,
): Element => buildDayCard(triathlonCardFactory, date, payload, extras, undefined, ctx)
