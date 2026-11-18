import type { LatentKind } from '../../../util/latent/types'
import type { Runtime } from './runtime'
import { viewKinds } from '../../../util/latent/filter'

// the kinds a reader chose to show, for the next visit; per browser, like the theme
const KEY = 'latent:view'

/** The kinds this reader last chose to show, or notes and pdfs. Storage may be off. */
export function loadView(present: LatentKind[]): LatentKind[] {
  let saved: string | null
  try {
    saved = localStorage.getItem(KEY)
  } catch {
    saved = null
  }
  return viewKinds(saved ? saved.split(',') : null, present)
}

export function saveView(kinds: LatentKind[]): void {
  try {
    localStorage.setItem(KEY, kinds.join(','))
  } catch {
    // private mode or storage off: the pick lasts for this page
  }
}

/** The document is of a shown kind (an embed, with no view, shows every kind). */
export function shownDoc(rt: Pick<Runtime, 'docs' | 'view'>, d: number): boolean {
  const doc = rt.docs[d]
  return !!doc && (!rt.view || rt.view.includes(doc.s))
}

/** The nearest document of a shown kind before (`step` -1) or after (+1) `d` in time, or -1. */
export function stepShown(rt: Pick<Runtime, 'docs' | 'view'>, d: number, step: 1 | -1): number {
  for (let i = d + step; i >= 0 && i < rt.docs.length; i += step) if (shownDoc(rt, i)) return i
  return -1
}
