import type { Paragraph, PhrasingContent } from 'mdast'

const calloutMapping = {
  note: 'note',
  abstract: 'abstract',
  summary: 'abstract',
  tldr: 'abstract',
  info: 'info',
  todo: 'todo',
  tip: 'tip',
  hint: 'tip',
  important: 'tip',
  success: 'success',
  check: 'success',
  done: 'success',
  question: 'question',
  help: 'question',
  faq: 'question',
  warning: 'warning',
  attention: 'warning',
  caution: 'warning',
  failure: 'failure',
  missing: 'failure',
  fail: 'failure',
  danger: 'danger',
  error: 'danger',
  bug: 'bug',
  example: 'example',
  quote: 'quote',
  cite: 'quote',
  math: 'math',
  proof: 'proof',
  lemma: 'lemma',
  theorem: 'theorem',
  theory: 'theory',
  proposition: 'proposition',
  propos: 'proposition',
  definition: 'definition',
  corollary: 'corollary',
  conjecture: 'conjecture',
  claim: 'claim',
  axiom: 'axiom',
  thm: 'theorem',
  def: 'definition',
}

const mathCalloutTypes = new Set<string>([
  'math',
  'proof',
  'lemma',
  'theorem',
  'theory',
  'proposition',
  'propos',
  'definition',
  'corollary',
  'conjecture',
  'claim',
  'axiom',
])

const mathCalloutTitlePrefixes = {
  proof: 'Proof',
  lemma: 'Lemma',
  theorem: 'Theorem',
  theory: 'Theory',
  proposition: 'Proposition',
  definition: 'Definition',
  corollary: 'Corollary',
  conjecture: 'Conjecture',
  claim: 'Claim',
  axiom: 'Axiom',
}

const proofLineRegex = /^(\s*)(proof)([:.])(\s*)/i

function isCalloutMappingKey(value: string): value is keyof typeof calloutMapping {
  return Object.hasOwn(calloutMapping, value)
}

function isMathCalloutTitlePrefixKey(
  value: string,
): value is keyof typeof mathCalloutTitlePrefixes {
  return Object.hasOwn(mathCalloutTitlePrefixes, value)
}

export function canonicalizeCallout(calloutName: string): string {
  const normalizedCallout = calloutName.toLowerCase()
  return isCalloutMappingKey(normalizedCallout) ? calloutMapping[normalizedCallout] : calloutName
}

export function isMathCallout(calloutType: string): boolean {
  return mathCalloutTypes.has(canonicalizeCallout(calloutType))
}

export function mathCalloutTitlePrefix(calloutType: string, title: string): string | undefined {
  const canonicalType = canonicalizeCallout(calloutType)

  if (!isMathCalloutTitlePrefixKey(canonicalType)) {
    return undefined
  }

  const prefix = mathCalloutTitlePrefixes[canonicalType]
  return title.toLowerCase().includes(prefix.toLowerCase()) ? undefined : prefix
}

function trimMathCalloutTitlePrefix(title: string, prefix: string): string {
  return title.replace(new RegExp(`^${prefix}\\b\\s*`, 'i'), '').trim()
}

function unwrapMathCalloutName(name: string): string {
  const trimmedName = name.trim()
  const wrappedName = trimmedName.match(/^\(([^)]+)\)(.*)$/)
  return wrappedName ? `${wrappedName[1]}${wrappedName[2]}`.trim() : trimmedName
}

function explicitMathCalloutNumberedTitle(
  title: string,
  prefix: string,
): { number: string; name: string } | undefined {
  const titleWithoutPrefix = trimMathCalloutTitlePrefix(title.trim(), prefix)
  const match = titleWithoutPrefix.match(/^(\d+(?:\.\d+)*)(?:\s*[.,:]?\s*(.*))?$/)

  if (!match) {
    return undefined
  }

  return { number: match[1], name: unwrapMathCalloutName(match[2] ?? '') }
}

export function formatMathCalloutTitle(
  calloutType: string,
  titleContent: string,
  fullTitle: string,
  index: number,
): string | undefined {
  const canonicalType = canonicalizeCallout(calloutType)

  if (canonicalType === 'math') {
    const trimmedTitle = titleContent.trim()
    return trimmedTitle.length > 0 ? `${index}. ${trimmedTitle}` : `${index}.`
  }

  if (!isMathCalloutTitlePrefixKey(canonicalType)) {
    return undefined
  }

  const prefix = mathCalloutTitlePrefixes[canonicalType]
  const trimmedTitle = titleContent.trim()
  const trimmedFullTitle = fullTitle.trim()
  const explicitTitle = explicitMathCalloutNumberedTitle(trimmedFullTitle, prefix)

  if (explicitTitle) {
    return explicitTitle.name.length > 0
      ? `${prefix} ${explicitTitle.number}. ${explicitTitle.name}`
      : `${prefix} ${explicitTitle.number}.`
  }

  const numberedPrefix = `${prefix} ${index}.`
  return trimmedTitle.length > 0 ? `${numberedPrefix} ${trimmedTitle}` : numberedPrefix
}

export function isStandaloneProofLine(paragraph: Paragraph): boolean {
  const firstChild = paragraph.children[0]

  if (!firstChild || firstChild.type !== 'text') {
    return false
  }

  const match = firstChild.value.match(proofLineRegex)

  if (!match) {
    return false
  }

  return paragraph.children.length === 1 && firstChild.value.slice(match[0].length).trim() === ''
}

export function italicizeProofLine(paragraph: Paragraph): boolean {
  const firstChild = paragraph.children[0]

  if (!firstChild || firstChild.type !== 'text') {
    return false
  }

  const match = firstChild.value.match(proofLineRegex)

  if (!match) {
    return false
  }

  const [, leading, label, punctuation, spacing] = match
  const rest = firstChild.value.slice(match[0].length)
  const replacement: PhrasingContent[] = []

  if (leading.length > 0) {
    replacement.push({ type: 'text', value: leading })
  }

  replacement.push({ type: 'emphasis', children: [{ type: 'text', value: label }] })
  replacement.push({ type: 'text', value: `${punctuation}${spacing}${rest}` })

  paragraph.children.splice(0, 1, ...replacement)
  return true
}
