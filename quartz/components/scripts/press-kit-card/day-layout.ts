import type { FormatKey } from './spec'

export type DayLayout = 'horizontal' | 'vertical'

export const DAY_LAYOUTS: [DayLayout, string][] = [
  ['horizontal', 'côte à côte'],
  ['vertical', 'empilé'],
]

export const isDayLayout = (value: unknown): value is DayLayout =>
  value === 'horizontal' || value === 'vertical'

// The most blocks a format stacks while each band keeps the grid's type size (measured on a
// three-activity day: a fourth band on portrait drops the type to about 0.6 of the grid's).
// Paysage is a 1.91:1 strip, so three bands would drop the type to about 12px; it keeps the grid.
export const STACK_LIMIT: Record<FormatKey, number> = {
  story: 4,
  portrait: 3,
  square: 3,
  landscape: 0,
}

/** The choice applies to three or more blocks; two or fewer always sit side by side. */
export const offersDayLayout = (blocks: number): boolean => blocks > 2

export const stacks = (layout: DayLayout | undefined, format: FormatKey, blocks: number): boolean =>
  layout === 'vertical' && offersDayLayout(blocks) && blocks <= STACK_LIMIT[format]
