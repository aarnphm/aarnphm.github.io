export type ReferenceContext = 'markdown' | 'literal' | 'editor'

export interface ReferenceSpan {
  readonly start: number
  readonly end: number
}

export interface ReferencePart {
  readonly raw: string
  readonly span: ReferenceSpan
}

export interface ReferenceSyntax {
  readonly version: 1
  readonly context: ReferenceContext
  readonly source: 'authored' | 'generated'
  readonly table: boolean
  readonly raw: string
  readonly span: ReferenceSpan
  readonly embed: boolean
  readonly target: ReferencePart
  readonly fragments: readonly ReferencePart[]
  readonly settings?: ReferencePart
  readonly label?: ReferencePart
  readonly labelSeparator?: ReferencePart
}

export interface ReferenceDiagnostic {
  readonly code: 'invalid-reference' | 'invalid-edit' | 'unsupported-label-block'
  readonly message: string
  readonly span?: ReferenceSpan
}

export type ReferenceField = 'target' | 'fragment' | 'settings' | 'label' | 'close'

export type ReferenceParseResult =
  | { readonly kind: 'complete'; readonly reference: ReferenceSyntax }
  | {
      readonly kind: 'incomplete'
      readonly raw: string
      readonly context: ReferenceContext
      readonly span: ReferenceSpan
      readonly field: ReferenceField
    }
  | { readonly kind: 'invalid'; readonly diagnostics: readonly ReferenceDiagnostic[] }

export type ReferenceEditResult = Exclude<ReferenceParseResult, { readonly kind: 'incomplete' }>

export interface ReferenceParts {
  readonly target: string
  readonly subpath?: string
  readonly label?: string
  readonly embed?: boolean
}

export type ReferenceRequest =
  | { readonly kind: 'syntax'; readonly reference: ReferenceSyntax }
  | { readonly kind: 'structured'; readonly parts: ReferenceParts }

export interface ReferenceParseOptions {
  readonly context: ReferenceContext
  readonly offset?: number
  readonly table?: boolean
}

/** Patches contain authored component text; the shared grammar validates the result. */
export interface ReferenceEdit {
  readonly embed?: boolean
  readonly target?: string
  readonly fragments?: readonly string[]
  readonly settings?: string | null
  readonly label?: string | null
}
