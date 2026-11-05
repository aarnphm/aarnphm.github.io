import type { ReferenceRequestRecord } from './dependencies'
import type { ReferencePart, ReferenceSyntax } from './model'

export interface ReferenceFields {
  readonly target: string
  readonly fragments: readonly string[]
  readonly settings?: string
  readonly label?: string
}

const decodeEscapes = (raw: string): string =>
  raw.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, '$1')

/** Markdown supplies compiler-decoded fields; literal properties retain character references. */
export function referenceFields(
  reference: ReferenceSyntax,
  decoded?: ReferenceFields,
): ReferenceFields {
  if (!decoded) {
    return {
      target: decodeEscapes(reference.target.raw),
      fragments: reference.fragments.map(part => decodeEscapes(part.raw)).filter(Boolean),
      ...(reference.settings ? { settings: decodeEscapes(reference.settings.raw) } : {}),
      ...(reference.label ? { label: decodeEscapes(reference.label.raw) } : {}),
    }
  }
  const field = (part: ReferencePart | undefined, value: string): string =>
    reference.labelSeparator?.raw === '\\|' &&
    part?.span.end === reference.labelSeparator.span.start &&
    value.endsWith('\\')
      ? value.slice(0, -1)
      : value
  return {
    target: field(reference.target, decoded.target),
    fragments: decoded.fragments
      .map((value, index) =>
        field(
          index === decoded.fragments.length - 1 ? reference.fragments.at(-1) : undefined,
          value,
        ),
      )
      .filter(Boolean),
    ...(decoded.settings !== undefined ? { settings: decoded.settings } : {}),
    ...(reference.label ? { label: decoded.label ?? '' } : {}),
  }
}

export function referenceRequest(
  reference: ReferenceSyntax,
  origin: string,
  fields: ReferenceFields = referenceFields(reference),
): ReferenceRequestRecord {
  return {
    ...fields,
    origin,
    embed: reference.embed,
    decodeUri: false,
    implicitAlias: true,
    ...(reference.source === 'authored' ? { authoredSpan: reference.span } : {}),
  }
}

declare module 'mdast' {
  interface Data {
    referenceFields?: ReferenceFields
  }
}
