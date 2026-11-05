import type { State } from 'micromark-util-types'
import type {
  ReferenceField,
  ReferenceParseOptions,
  ReferenceParseResult,
  ReferencePart,
} from './model'
import { tokenizeReference } from './lexer'

export function isEscapedAt(text: string, index: number): boolean {
  let count = 0
  for (let cursor = index - 1; cursor >= 0 && text.charCodeAt(cursor) === 92; cursor--) count++
  return count % 2 === 1
}

export function readReferenceAt(
  text: string,
  start: number,
  options: ReferenceParseOptions,
): ReferenceParseResult {
  let cursor = start
  let complete = false
  let failed = false
  let embed = false
  let field: ReferenceField = 'target'
  let openingEnd = start
  let closingStart = start
  let targetStart = -1
  let targetEnd = -1
  let settingsStart = -1
  let settingsEnd = -1
  let aliasStart = -1
  let aliasEnd = -1
  const markers: number[] = []
  const offset = options.offset ?? 0
  const accept: State = () => undefined
  const reject: State = () => {
    failed = true
    return undefined
  }
  let state: State | undefined = tokenizeReference(
    {
      enter(type) {
        switch (type) {
          case 'wikilinkEmbedMarker':
            embed = true
            break
          case 'wikilinkTarget':
            targetStart = cursor
            break
          case 'wikilinkAnchorMarker':
            markers.push(cursor)
            field = 'fragment'
            break
          case 'wikilinkMetadataMarker':
            settingsStart = cursor
            field = 'settings'
            break
          case 'wikilinkAliasMarker':
            aliasStart = cursor
            field = 'label'
            break
          case 'wikilinkCloseMarker':
            closingStart = cursor
            field = 'close'
            break
        }
      },
      exit(type) {
        switch (type) {
          case 'wikilinkOpenMarker':
            openingEnd = cursor
            break
          case 'wikilinkTarget':
            targetEnd = cursor
            break
          case 'wikilinkMetadataCloseMarker':
            settingsEnd = cursor
            break
          case 'wikilinkAliasMarker':
            aliasEnd = cursor
            break
          case 'wikilink':
            complete = true
            break
        }
      },
      consume() {
        cursor++
      },
    },
    accept,
    reject,
  )

  while (state && cursor < text.length && !complete && !failed) {
    const code = text.charCodeAt(cursor)
    // Micromark supplies negative line-ending codes to this same lexer.
    state = state(code === 10 ? -4 : code === 13 ? -5 : code)
  }
  const span = Object.freeze({ start: start + offset, end: cursor + offset })
  if (!complete) {
    if (
      !failed &&
      cursor === text.length &&
      (text.startsWith('[[', start) || text.startsWith('![[', start))
    ) {
      return { kind: 'incomplete', raw: text.slice(start), context: options.context, span, field }
    }
    return {
      kind: 'invalid',
      diagnostics: [
        { code: 'invalid-reference', message: 'Expected a complete single-line wikilink.', span },
      ],
    }
  }

  const part = (from: number, to: number): ReferencePart =>
    Object.freeze({
      raw: text.slice(from, to),
      span: Object.freeze({ start: from + offset, end: to + offset }),
    })
  const separatorStart =
    aliasStart >= 0 ? aliasStart - (isEscapedAt(text, aliasStart) ? 1 : 0) : closingStart
  const fragments: ReferencePart[] = []
  for (let index = 0; index < markers.length; index++) {
    const marker = markers[index]
    if (marker + 1 !== settingsStart)
      fragments.push(part(marker + 1, markers[index + 1] ?? separatorStart))
  }
  return {
    kind: 'complete',
    reference: Object.freeze({
      version: 1,
      context: options.context,
      source: 'authored',
      table: options.table ?? false,
      raw: text.slice(start, cursor),
      span,
      embed,
      target:
        targetStart >= 0
          ? part(targetStart, Math.min(targetEnd, separatorStart))
          : part(openingEnd, openingEnd),
      fragments: Object.freeze(fragments),
      ...(settingsStart >= 0
        ? { settings: part(settingsStart, settingsEnd >= 0 ? settingsEnd : settingsStart + 1) }
        : {}),
      ...(aliasStart >= 0
        ? { label: part(aliasEnd, closingStart), labelSeparator: part(separatorStart, aliasEnd) }
        : {}),
    }),
  }
}
