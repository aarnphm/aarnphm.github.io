import type {
  ReferenceEdit,
  ReferenceEditResult,
  ReferenceParseOptions,
  ReferenceParseResult,
  ReferenceParts,
  ReferenceRequest,
  ReferenceSyntax,
} from './model'
import { isEscapedAt, readReferenceAt } from './scan'

export type {
  ReferenceSyntax,
  ReferenceParseResult,
  ReferenceRequest,
  ReferenceEdit,
  ReferenceEditResult,
} from './model'

export function parseReference(
  source: string,
  options: ReferenceParseOptions,
): ReferenceParseResult {
  const result = readReferenceAt(source, 0, options)
  if (result.kind === 'complete' && result.reference.raw.length !== source.length) {
    return {
      kind: 'invalid',
      diagnostics: [
        {
          code: 'invalid-reference',
          message: 'Unexpected text after the closing wikilink delimiter.',
        },
      ],
    }
  }
  return result
}

export function scanReferences(source: string, options: ReferenceParseOptions): ReferenceSyntax[] {
  const references: ReferenceSyntax[] = []
  let cursor = 0
  while (cursor < source.length) {
    const opening = source.indexOf('[[', cursor)
    if (opening < 0) break
    if (isEscapedAt(source, opening)) {
      cursor = opening + 2
      continue
    }
    const start =
      source[opening - 1] === '!' && !isEscapedAt(source, opening - 1) ? opening - 1 : opening
    const result = readReferenceAt(source, start, options)
    if (result.kind === 'complete') {
      references.push(result.reference)
      cursor = result.reference.span.end - (options.offset ?? 0)
    } else cursor = opening + 2
  }
  return references
}

export function referenceFromParts(
  parts: ReferenceParts,
): Extract<ReferenceRequest, { kind: 'structured' }> {
  return Object.freeze({ kind: 'structured', parts: Object.freeze({ ...parts }) })
}

export function serializeReference(reference: ReferenceSyntax): string {
  return reference.raw
}

export function editReference(
  reference: ReferenceSyntax,
  patch: ReferenceEdit,
): ReferenceEditResult {
  const source = reference.raw
  const base = reference.span.start
  const close = source.length - 2
  const settingsStart = reference.settings ? reference.settings.span.start - base - 1 : undefined
  const aliasStart = reference.labelSeparator
    ? reference.labelSeparator.span.start - base
    : undefined
  const targetEnd = Math.min(reference.target.span.end - base, aliasStart ?? close)
  const fragmentEnd = settingsStart ?? aliasStart ?? close
  const originalTarget = source.slice(reference.target.span.start - base, targetEnd)
  const fragmentText =
    patch.fragments === undefined
      ? source.slice(targetEnd, fragmentEnd)
      : patch.fragments.map(fragment => `#${fragment}`).join('')
  const settingsText =
    patch.settings === undefined
      ? settingsStart === undefined
        ? ''
        : source.slice(settingsStart, aliasStart ?? close)
      : patch.settings === null
        ? ''
        : `#${patch.settings}`
  const labelText =
    patch.label === undefined
      ? aliasStart === undefined
        ? ''
        : source.slice(aliasStart, close)
      : patch.label === null
        ? ''
        : `|${patch.label}`
  let raw = `${(patch.embed ?? reference.embed) ? '!' : ''}[[${patch.target ?? originalTarget}${fragmentText}${settingsText}${labelText}]]`
  if (reference.table) raw = raw.replace(/((^|[^\\])(\\\\)*)\|/g, '$1\\|')
  const parsed = parseReference(raw, { context: reference.context, table: reference.table })
  if (parsed.kind !== 'complete') {
    return {
      kind: 'invalid',
      diagnostics: [
        {
          code: 'invalid-edit',
          message: 'The edited components do not form a complete single-line wikilink.',
        },
      ],
    }
  }
  const output = parsed.reference
  const inContext = (text: string) =>
    reference.table ? text.replace(/((^|[^\\])(\\\\)*)\|/g, '$1\\|') : text
  const expectedFragments = patch.fragments?.map(inContext)
  if (
    (patch.target !== undefined && output.target.raw !== inContext(patch.target)) ||
    (expectedFragments !== undefined &&
      (output.fragments.length !== expectedFragments.length ||
        output.fragments.some((fragment, index) => fragment.raw !== expectedFragments[index]))) ||
    (patch.settings !== undefined &&
      output.settings?.raw !== (patch.settings === null ? undefined : inContext(patch.settings))) ||
    (patch.label !== undefined &&
      output.label?.raw !== (patch.label === null ? undefined : inContext(patch.label)))
  ) {
    return {
      kind: 'invalid',
      diagnostics: [
        {
          code: 'invalid-edit',
          message:
            'A component contains a delimiter that changes another reference field; escape the delimiter within that component.',
        },
      ],
    }
  }
  return { kind: 'complete', reference: Object.freeze({ ...output, source: 'generated' }) }
}

export type ReferenceCursorResult =
  | { readonly kind: 'inactive' }
  | { readonly kind: 'complete'; readonly reference: ReferenceSyntax }
  | {
      readonly kind: 'incomplete'
      readonly field:
        | 'target'
        | 'heading'
        | 'block'
        | 'settings'
        | 'label'
        | 'close'
        | 'heading-search'
        | 'block-search'
      readonly raw: string
      readonly start: number
    }

export function referenceAtCursor(
  source: string,
  cursor: number,
  options: { protected?: boolean } = {},
): ReferenceCursorResult {
  if (options.protected || cursor < 0 || cursor > source.length) return { kind: 'inactive' }
  const prefix = source.slice(0, cursor)
  const opening = prefix.lastIndexOf('[[')
  if (opening < 0 || isEscapedAt(source, opening)) return { kind: 'inactive' }
  const start =
    source[opening - 1] === '!' && !isEscapedAt(source, opening - 1) ? opening - 1 : opening
  const raw = prefix.slice(start)
  const result = parseReference(raw, { context: 'editor', offset: start })
  if (result.kind === 'complete') return result
  if (result.kind === 'invalid') return { kind: 'inactive' }
  const body = prefix.slice(opening + 2)
  const field = body.startsWith('##')
    ? 'heading-search'
    : body.startsWith('^^')
      ? 'block-search'
      : result.field === 'fragment'
        ? /#\^[^#]*$/.test(body)
          ? 'block'
          : 'heading'
        : result.field
  return { kind: 'incomplete', field, raw, start }
}
