import type { LeanCatalogEntry } from './catalog-entry'
import { LEAN_KEYWORDS } from './catalog-keywords'
import { LEAN_MATHLIB } from './catalog-mathlib'
import { LEAN_TACTICS } from './catalog-tactics'

export const LEAN_STATIC_CATALOG: readonly LeanCatalogEntry[] = [
  ...LEAN_TACTICS,
  ...LEAN_KEYWORDS,
  ...LEAN_MATHLIB,
]
