import type {
  ReferenceCatalogEntry,
  ReferenceHeading,
  ReferenceInterpretation,
  ReferenceSelector,
  ReferenceSettings,
  ReferenceTargetDiagnostic,
} from './catalog-types'
import { parseNoteStack } from '../note-stack'
import { slugAnchor } from '../path'
import { isRecord } from '../type-guards'
import { labelToText, parseReferenceLabel } from './label'

type ReferenceTargetInfo = Pick<
  ReferenceCatalogEntry,
  'path' | 'kind' | 'headings' | 'blocks' | 'baseViews'
>

export interface ReferenceInterpretationInput {
  readonly fragments?: readonly string[]
  readonly settings?: string
}

export interface PdfFragment {
  readonly page?: string
  readonly height?: string
  readonly diagnostics: readonly ReferenceTargetDiagnostic[]
}

export function interpretPdfFragment(anchor: string): PdfFragment {
  const diagnostics: ReferenceTargetDiagnostic[] = []
  const parameters = new URLSearchParams(anchor.replace(/^#/, ''))
  const values: { page?: string; height?: string } = {}
  for (const key of new Set(parameters.keys())) {
    const entries = parameters.getAll(key)
    if (key !== 'page' && key !== 'height') {
      diagnostics.push({
        code: 'unsupported-selector',
        message: `Unsupported PDF parameter: ${key}`,
      })
      continue
    }
    const value = entries[0]
    if (
      entries.length !== 1 ||
      !/^\d+$/.test(value) ||
      !Number.isSafeInteger(Number(value)) ||
      Number(value) <= 0
    ) {
      diagnostics.push({
        code: 'invalid-selector',
        message: `PDF ${key} must be one positive integer`,
      })
      continue
    }
    values[key] = String(Number(value))
  }
  return { ...values, diagnostics }
}

const fold = (text: string): string => text.normalize('NFC').trim().toLowerCase()
const headingText = (text: string): string =>
  fold(text.replace(/\$([^$]+)\$/g, '$1').replace(/\s+/g, ' '))
const authoredHeadingText = (text: string): string =>
  headingText(/[*_`\\]/.test(text) ? labelToText(parseReferenceLabel(text)) : text)

function headingPaths(
  headings: readonly ReferenceHeading[],
): readonly { heading: ReferenceHeading; path: readonly string[] }[] {
  const parents: { depth: number; text: string }[] = []
  return headings.map(heading => {
    const depth = heading.depth ?? 1
    while (parents.length > 0 && parents[parents.length - 1].depth >= depth) parents.pop()
    const current = heading.path ?? [...parents.map(parent => parent.text), heading.text]
    parents.push({ depth, text: heading.text })
    return { heading, path: current }
  })
}

function matchingHeading(
  entry: ReferenceTargetInfo,
  segments: readonly string[],
): ReferenceHeading | undefined {
  const target = segments.map(authoredHeadingText)
  const paths = headingPaths(entry.headings ?? [])
  const textMatch = paths.find(candidate => {
    const suffix = candidate.path.slice(-target.length).map(headingText)
    return suffix.length === target.length && suffix.every((part, index) => part === target[index])
  })
  const anchorMatch =
    target.length === 1
      ? paths.find(
          candidate => fold(candidate.heading.anchor.replace(/^#/, '')) === fold(segments[0]),
        )
      : undefined
  return (textMatch ?? anchorMatch)?.heading
}

function headingSelector(
  entry: ReferenceTargetInfo,
  segments: readonly string[],
  diagnostics: ReferenceTargetDiagnostic[],
): ReferenceSelector {
  const matched = matchingHeading(entry, segments)
  if (!matched && entry.headings)
    diagnostics.push({
      code: 'missing-selector',
      message: `Heading not found in ${entry.path}: ${segments.join('#')}`,
    })
  let text = segments[segments.length - 1] ?? ''
  if (text.includes('/') && text.includes(' ')) {
    const tail = text.slice(text.lastIndexOf('/') + 1)
    const space = tail.indexOf(' ')
    if (space >= 0 && tail.slice(space + 1).trim()) text = tail.slice(space + 1).trim()
  }
  const fallback = text
    .replace(/\$([^$]+)\$/g, (_, content: string) => content.trim())
    .replace(/\s+/g, ' ')
    .trim()
  return {
    kind: 'heading',
    segments: Object.freeze([...segments]),
    anchor: `#${matched ? matched.anchor.replace(/^#/, '') : slugAnchor(fallback)}`,
  }
}

function settingsFor(
  entry: ReferenceTargetInfo,
  raw: string | undefined,
  diagnostics: ReferenceTargetDiagnostic[],
): ReferenceSettings | undefined {
  if (raw === undefined) return undefined
  const text = raw.trim().replace(/^\{([\s\S]*)\}$/, '$1')
  if (entry.kind === 'image') return { kind: 'css', raw: text }
  const candidates = [
    `{${text}}`,
    `{${text
      .replace(/(\w+)\s*:/g, '"$1":')
      .replace(/'/g, '"')
      .replace(/,\s*([}\]])/g, '$1')}}`,
  ]
  for (const candidate of candidates) {
    try {
      const value: unknown = JSON.parse(candidate)
      if (isRecord(value)) return { kind: 'object', value: Object.freeze(value) }
    } catch {
      // The existing authoring format permits relaxed object keys and single quotes.
    }
  }
  diagnostics.push({
    code: 'invalid-settings',
    message: `Invalid settings for ${entry.kind}: ${raw}`,
  })
  return undefined
}

export function interpretReference(
  entry: ReferenceTargetInfo,
  input: ReferenceInterpretationInput = {},
): ReferenceInterpretation {
  const diagnostics: ReferenceTargetDiagnostic[] = []
  const fragments = input.fragments ?? []
  const raw = fragments.join('#')
  const stack =
    entry.path.replace(/^\//, '').replace(/\.md$/, '') === 'notes' ? parseNoteStack(raw) : undefined
  let selector: ReferenceSelector = { kind: 'whole' }
  if (raw) {
    if (stack) {
      selector = { kind: 'note-stack', raw, slugs: stack.slugs }
      if (!stack.valid)
        diagnostics.push({ code: 'invalid-selector', message: 'Invalid note trail' })
    } else if (entry.kind === 'pdf') {
      const pdf = interpretPdfFragment(raw)
      diagnostics.push(...pdf.diagnostics)
      selector = {
        kind: 'pdf',
        raw,
        ...(pdf.page ? { page: pdf.page } : {}),
        ...(pdf.height ? { height: pdf.height } : {}),
      }
    } else if (entry.kind === 'base') {
      const view = entry.baseViews?.find(candidate => fold(candidate.name) === fold(raw))
      if (entry.baseViews && !view)
        diagnostics.push({ code: 'missing-selector', message: `Base view not found: ${raw}` })
      selector = {
        kind: 'base',
        view: view?.name ?? raw,
        ...(view?.slug ? { slug: view.slug } : {}),
      }
    } else if (entry.kind === 'notebook') {
      selector =
        !raw.startsWith('^') && matchingHeading(entry, fragments)
          ? headingSelector(entry, fragments, diagnostics)
          : { kind: 'notebook', cells: Object.freeze([...fragments]) }
    } else if (/^(?:triathlon)(?:\/|\.md$)/.test(entry.path)) {
      selector = { kind: 'triathlon', raw }
    } else if (/^(?:curius)(?:\/|\.md$)/.test(entry.path)) {
      selector = { kind: 'provider', raw }
    } else if (['note', 'canvas', 'flashcard'].includes(entry.kind)) {
      if (raw.startsWith('^')) {
        const authored = raw.slice(1)
        const id = entry.blocks?.find(candidate => fold(candidate) === fold(authored)) ?? authored
        if (entry.blocks && !entry.blocks.some(candidate => fold(candidate) === fold(authored)))
          diagnostics.push({ code: 'missing-selector', message: `Block not found: ${authored}` })
        selector = { kind: 'block', id, anchor: `#^${id}` }
      } else selector = headingSelector(entry, fragments, diagnostics)
    } else selector = { kind: 'resource', raw }
  }
  const settings = settingsFor(entry, input.settings, diagnostics)
  return { selector, ...(settings ? { settings } : {}), diagnostics }
}
