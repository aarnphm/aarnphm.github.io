export type LeanCatalogCategory = 'tactic' | 'keyword' | 'mathlib' | 'local'

export type LeanCatalogEntry = {
  readonly label: string
  readonly category: LeanCatalogCategory
  readonly type: string
  readonly detail: string
  readonly documentation: string
}
