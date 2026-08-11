export interface TriathlonEvents {
  presentation: { previous: TriathlonPresentation; current: TriathlonPresentation }
  command: { name: string; value?: string }
  analyticsWeek: { source: 'load' | 'effort'; index: number }
}

export type TriathlonEventName = keyof TriathlonEvents
export type TriathlonEventListener<K extends TriathlonEventName> = (
  detail: TriathlonEvents[K],
) => void

export interface TriathlonEventBus {
  dispatch<K extends TriathlonEventName>(name: K, detail: TriathlonEvents[K]): void
  subscribe<K extends TriathlonEventName>(name: K, listener: TriathlonEventListener<K>): () => void
}

import type { TriathlonPresentation } from '../../../util/triathlon-presentation'

export const createTriathlonEventBus = (): TriathlonEventBus => {
  const target = new EventTarget()
  return {
    dispatch: (name, detail) => target.dispatchEvent(new CustomEvent(name, { detail })),
    subscribe: (name, listener) => {
      const receive = (event: Event): void => {
        if (event instanceof CustomEvent) listener(event.detail)
      }
      target.addEventListener(name, receive)
      return () => target.removeEventListener(name, receive)
    },
  }
}
