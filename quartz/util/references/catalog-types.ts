import type { FullSlug } from '../path'

export type ReferenceFileKind =
  | 'note'
  | 'base'
  | 'canvas'
  | 'flashcard'
  | 'notebook'
  | 'image'
  | 'audio'
  | 'video'
  | 'pdf'
  | 'code'
  | 'unknown'

export interface ReferenceHeading {
  readonly text: string
  readonly anchor: string
  readonly depth?: number
  readonly path?: readonly string[]
}

export interface ReferenceBaseView {
  readonly name: string
  readonly slug?: FullSlug
}

export interface ReferenceCatalogInput {
  readonly path: string
  readonly slug?: FullSlug
  readonly aliases?: readonly string[]
  readonly headings?: readonly ReferenceHeading[]
  readonly blocks?: readonly string[]
  readonly baseViews?: readonly ReferenceBaseView[]
  /** a `layout: latent` page: its fragments name a view of the embedding space */
  readonly latent?: boolean
}

export interface ReferenceCatalogEntry {
  readonly path: string
  readonly slug: FullSlug
  readonly kind: ReferenceFileKind
  readonly aliases: readonly string[]
  readonly headings?: readonly ReferenceHeading[]
  readonly blocks?: readonly string[]
  readonly baseViews?: readonly ReferenceBaseView[]
  readonly latent?: boolean
}

export interface ReferenceResolutionOptions {
  readonly implicitAlias?: boolean
  readonly decodeUri?: boolean
  readonly aliases?: boolean
}

interface LookupResult {
  readonly lookupKeys: readonly string[]
}

export type ReferenceResolution =
  | (LookupResult & {
      readonly kind: 'found'
      readonly entry: ReferenceCatalogEntry
      readonly implicitLabel?: string
    })
  | (LookupResult & { readonly kind: 'missing'; readonly target: string })
  | (LookupResult & {
      readonly kind: 'ambiguous'
      readonly target: string
      readonly candidates: readonly ReferenceCatalogEntry[]
    })
  | (LookupResult & { readonly kind: 'external'; readonly target: string })
  | (LookupResult & {
      readonly kind: 'unsupported'
      readonly target: string
      readonly reason: string
    })

export interface ReferenceCatalog {
  readonly entries: readonly ReferenceCatalogEntry[]
  resolve(target: string, origin: string, options?: ReferenceResolutionOptions): ReferenceResolution
}

export interface ReferenceTargetDiagnostic {
  readonly code:
    | 'invalid-selector'
    | 'invalid-settings'
    | 'missing-selector'
    | 'unsupported-selector'
  readonly message: string
}

export type ReferenceSelector =
  | { readonly kind: 'whole' }
  | { readonly kind: 'heading'; readonly segments: readonly string[]; readonly anchor: string }
  | { readonly kind: 'block'; readonly id: string; readonly anchor: string }
  | { readonly kind: 'pdf'; readonly raw: string; readonly page?: string; readonly height?: string }
  | { readonly kind: 'base'; readonly view: string; readonly slug?: FullSlug }
  | { readonly kind: 'notebook'; readonly cells: readonly string[] }
  | { readonly kind: 'note-stack'; readonly raw: string; readonly slugs: readonly string[] }
  | { readonly kind: 'provider'; readonly raw: string }
  | { readonly kind: 'triathlon'; readonly raw: string }
  /** a view of a `layout: latent` page; the renderer decodes `view` (util/latent/embed) */
  | {
      readonly kind: 'latent'
      readonly raw: string
      readonly focus?: number
      readonly view?: string
      readonly error?: string
    }
  | { readonly kind: 'resource'; readonly raw: string }

export type ReferenceSettings =
  | { readonly kind: 'css'; readonly raw: string }
  | { readonly kind: 'object'; readonly value: Readonly<Record<string, unknown>> }

export interface ReferenceInterpretation {
  readonly selector: ReferenceSelector
  readonly settings?: ReferenceSettings
  readonly diagnostics: readonly ReferenceTargetDiagnostic[]
}
