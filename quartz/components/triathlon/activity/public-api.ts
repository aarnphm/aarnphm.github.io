import type { TriathlonContext } from '../runtime/context'
import { buildDayCard } from './embeds'

export type DayCardFacade = NonNullable<typeof window.quartzTriathlon>['dayCard']

export const createDayCardFacade =
  (context: TriathlonContext): DayCardFacade =>
  async (date, detailPath, extras) => {
    const result = await context.resources.detail.load(detailPath)
    if (result.status !== 'ready' || context.signal.aborted) return null
    const view = buildDayCard(context.presentation, date, result.value, extras ?? {})
    const cleanup = view.mount()
    context.signal.addEventListener('abort', cleanup, { once: true })
    return view.element
  }
