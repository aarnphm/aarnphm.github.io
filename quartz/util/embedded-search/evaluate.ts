import type { QueryExpression, SearchCorpus, SearchDocument, SearchEvaluation } from './types'

interface EvaluationContext {
  readonly document: SearchDocument
  readonly text: string
  readonly caseSensitive: boolean
  readonly propertyValue?: unknown
  readonly tag?: string
}

function scalarValues(value: unknown): string[] {
  if (Array.isArray(value)) return value.flatMap(scalarValues)
  if (value instanceof Date) return [value.toISOString()]
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
    ? [String(value)]
    : []
}

function compareProperty(query: string, value: unknown): boolean | undefined {
  if (query === 'null') return value === null
  const comparison = /^(<=|>=|<|>|=)(.+)$/.exec(query)
  if (!comparison) return undefined
  const rightText = comparison[2].trim()
  return scalarValues(value).some(leftText => {
    const leftNumber = Number(leftText)
    const rightNumber = Number(rightText)
    const numeric =
      leftText.trim().length > 0 &&
      rightText.length > 0 &&
      Number.isFinite(leftNumber) &&
      Number.isFinite(rightNumber)
    const isoDate = /^\d{4}-\d{2}-\d{2}(?:[Tt][0-9:.+-]+[Zz]?)?$/
    const leftDate = isoDate.test(leftText) ? Date.parse(leftText) : NaN
    const rightDate = isoDate.test(rightText) ? Date.parse(rightText) : NaN
    const dates = Number.isFinite(leftDate) && Number.isFinite(rightDate)
    const left = numeric ? leftNumber : dates ? leftDate : leftText.normalize('NFC')
    const right = numeric ? rightNumber : dates ? rightDate : rightText.normalize('NFC')
    switch (comparison[1]) {
      case '<':
        return left < right
      case '<=':
        return left <= right
      case '>':
        return left > right
      case '>=':
        return left >= right
      default:
        return left === right
    }
  })
}

export function evaluateEmbeddedQuery(
  expression: QueryExpression,
  corpus: SearchCorpus,
): SearchEvaluation {
  if (corpus.kind === 'error') return corpus
  let operations = 2000000
  const regexes = new Map<QueryExpression, RegExp>()
  function matches(node: QueryExpression, context: EvaluationContext): boolean {
    if (--operations < 0) throw new Error('query exceeds the evaluation work limit')
    const { document, text, caseSensitive } = context
    switch (node.kind) {
      case 'and':
        return node.children.every(child => matches(child, context))
      case 'or':
        return node.children.some(child => matches(child, context))
      case 'not':
        return !matches(node.child, context)
      case 'term': {
        if (context.tag !== undefined) {
          const value = node.value.replace(/^#/, '')
          const tag = caseSensitive ? context.tag : context.tag.toLowerCase()
          const query = caseSensitive ? value : value.toLowerCase()
          return tag === query || tag.startsWith(`${query}/`)
        }
        if ('propertyValue' in context) {
          const comparison = compareProperty(node.value, context.propertyValue)
          if (comparison !== undefined) return comparison
        }
        return caseSensitive
          ? text.includes(node.value)
          : text.toLowerCase().includes(node.value.toLowerCase())
      }
      case 'regex': {
        if (text.length > 4096)
          throw new Error(
            'regular expression scope exceeds 4,096 characters; narrow it with line:, block:, file: or path:',
          )
        const operators = node.source.replace(/\\.|\[(?:\\.|[^\]])*\]/g, '')
        let width = node.source.length
        for (const repeated of operators.matchAll(/\{(\d+)\}/g))
          width += Number(repeated[1]) - repeated[0].length
        operations -=
          (/[*+?]/.test(operators) ? text.length * text.length : text.length) * Math.max(1, width)
        if (operations < 0) throw new Error('query exceeds the evaluation work limit')
        let regex = regexes.get(node)
        if (!regex) {
          const flags = caseSensitive
            ? node.flags.replaceAll('i', '')
            : node.flags.includes('i')
              ? node.flags
              : `${node.flags}i`
          regex = new RegExp(node.source, flags)
          regexes.set(node, regex)
        }
        return regex.test(text)
      }
      case 'property':
        return Object.entries(document.properties).some(
          ([key, value]) =>
            matches(node.key, { ...context, text: key }) &&
            (!node.value ||
              matches(node.value, {
                ...context,
                text: scalarValues(value).join('\n'),
                propertyValue: value,
              })),
        )
      case 'scope': {
        if (node.scope === 'match-case' || node.scope === 'ignore-case')
          return matches(node.child, { ...context, caseSensitive: node.scope === 'match-case' })
        const evaluate = (value: string) => matches(node.child, { ...context, text: value })
        switch (node.scope) {
          case 'file':
            return evaluate(document.path.slice(document.path.lastIndexOf('/') + 1))
          case 'path':
            return evaluate(document.path)
          case 'content':
            return evaluate(document.text)
          case 'tag':
            return document.tags.some(tag => matches(node.child, { ...context, text: tag, tag }))
          case 'line':
            return document.text.split(/\r?\n/).some(evaluate)
          case 'block':
            return document.blocks.some(evaluate)
          case 'section':
            return document.sections.some(evaluate)
          case 'task':
            return document.tasks.some(task => evaluate(task.text))
          case 'task-todo':
            return document.tasks.some(task => !task.checked && evaluate(task.text))
          case 'task-done':
            return document.tasks.some(task => task.checked && evaluate(task.text))
        }
      }
    }
  }
  try {
    const documents: SearchDocument[] = []
    for (const document of corpus.documents) {
      if (!matches(expression, { document, text: document.text, caseSensitive: false })) continue
      if (documents.length === 100) return { kind: 'results', documents, truncated: true }
      documents.push(document)
    }
    return { kind: 'results', documents, truncated: false }
  } catch (error) {
    return {
      kind: 'error',
      message: error instanceof Error ? error.message : 'query evaluation failed',
    }
  }
}
