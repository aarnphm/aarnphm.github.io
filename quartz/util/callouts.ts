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
  propos: 'propos',
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

const proofLineRegex = /^(\s*)(proof)([:.])(\s*)/i

function isCalloutMappingKey(value: string): value is keyof typeof calloutMapping {
  return Object.hasOwn(calloutMapping, value)
}

export function canonicalizeCallout(calloutName: string): string {
  const normalizedCallout = calloutName.toLowerCase()
  return isCalloutMappingKey(normalizedCallout) ? calloutMapping[normalizedCallout] : calloutName
}

export function isMathCallout(calloutType: string): boolean {
  return mathCalloutTypes.has(canonicalizeCallout(calloutType))
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
