export const TRI_ACTIVITY_LAYOUTS = ['one', 'two', 'all'] as const

export type TriActivityLayout = (typeof TRI_ACTIVITY_LAYOUTS)[number]

// The stylesheet pages two activities when an embed has no layout attribute, so server output
// already matches the default before any script runs.
export const DEFAULT_TRI_ACTIVITY_LAYOUT: TriActivityLayout = 'two'

export const TRI_ACTIVITY_LAYOUT_KEY = 'tri-activity-layout'

export const TRI_ACTIVITY_LAYOUT_EVENT = 'tri:activity-layout'

export const isTriActivityLayout = (value: unknown): value is TriActivityLayout =>
  typeof value === 'string' && (TRI_ACTIVITY_LAYOUTS as readonly string[]).includes(value)

export const readTriActivityLayout = (): TriActivityLayout => {
  try {
    const stored = localStorage.getItem(TRI_ACTIVITY_LAYOUT_KEY)
    return isTriActivityLayout(stored) ? stored : DEFAULT_TRI_ACTIVITY_LAYOUT
  } catch {
    return DEFAULT_TRI_ACTIVITY_LAYOUT
  }
}

export const setTriActivityLayout = (layout: TriActivityLayout): void => {
  try {
    localStorage.setItem(TRI_ACTIVITY_LAYOUT_KEY, layout)
  } catch {}
  window.dispatchEvent(new CustomEvent(TRI_ACTIVITY_LAYOUT_EVENT, { detail: { layout } }))
}
