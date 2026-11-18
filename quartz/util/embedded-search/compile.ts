import type { CompiledQuery, QueryExpression, SearchScope } from './types'

interface Token {
  readonly kind: 'word' | 'phrase' | 'regex' | '(' | ')' | '[' | ']' | ':' | '-'
  readonly value: string
  readonly flags?: string
}

const scopes = new Set<string>([
  'file',
  'path',
  'content',
  'tag',
  'match-case',
  'ignore-case',
  'line',
  'block',
  'section',
  'task',
  'task-todo',
  'task-done',
])

function isScope(value: string): value is SearchScope {
  return scopes.has(value)
}

function tokenize(query: string): Token[] {
  if (query.length > 2048) throw new SyntaxError('query exceeds 2,048 characters')
  const tokens: Token[] = []
  let index = 0
  while (index < query.length) {
    const char = query[index]
    if (/\s/.test(char)) {
      index++
      continue
    }
    if (
      char === '(' ||
      char === ')' ||
      char === '[' ||
      char === ']' ||
      char === ':' ||
      char === '-'
    ) {
      tokens.push({ kind: char, value: char })
      index++
    } else if (char === '"') {
      index++
      let value = ''
      while (index < query.length && query[index] !== '"') {
        if (query[index] === '\\') index++
        if (index >= query.length) throw new SyntaxError('unfinished quoted escape')
        value += query[index++]
      }
      if (query[index] !== '"') throw new SyntaxError('unclosed quoted phrase')
      index++
      if (!value) throw new SyntaxError('empty quoted phrase')
      tokens.push({ kind: 'phrase', value })
    } else if (char === '/') {
      index++
      let value = ''
      let inClass = false
      while (index < query.length) {
        const next = query[index++]
        if (next === '\\') {
          if (index >= query.length) throw new SyntaxError('unfinished regular expression escape')
          value += next + query[index++]
        } else if (next === '/' && !inClass) break
        else {
          value += next
          if (next === '[') inClass = true
          if (next === ']') inClass = false
        }
      }
      if (query[index - 1] !== '/' || !value)
        throw new SyntaxError('unclosed or empty regular expression')
      let flags = ''
      while (index < query.length && /[a-z]/i.test(query[index])) flags += query[index++]
      validateRegex(value, flags)
      tokens.push({ kind: 'regex', value, flags })
    } else {
      let value = ''
      while (index < query.length && !/[\s()[\]:]/.test(query[index])) {
        if (query[index] === '\\') index++
        if (index >= query.length) throw new SyntaxError('unfinished escape')
        value += query[index++]
      }
      if (!value) throw new SyntaxError(`unexpected character ${query[index]}`)
      tokens.push({ kind: 'word', value })
    }
    if (tokens.length > 256) throw new SyntaxError('query exceeds 256 tokens')
  }
  return tokens
}

function validateRegex(source: string, flags: string): void {
  if (source.length > 128) throw new SyntaxError('regular expression exceeds 128 characters')
  if (/[^imsu]/.test(flags) || new Set(flags).size !== flags.length)
    throw new SyntaxError('regular expression flags must be unique i, m, s or u')
  if (/\\[1-9]|\(\?/.test(source) || /\)[*+?{]/.test(source))
    throw new SyntaxError(
      'regular expression groups cannot use lookarounds, backreferences or quantifiers',
    )
  const operators = source.replace(/\\.|\[(?:\\.|[^\]])*\]/g, '')
  if ((operators.match(/[*+?]/g)?.length ?? 0) > 1 || /\{\d+,/.test(operators))
    throw new SyntaxError('regular expression permits at most one variable quantifier')
  for (const match of operators.matchAll(/\{(\d+)\}/g))
    if (Number(match[1]) > 1000)
      throw new SyntaxError('regular expression repetition exceeds 1,000')
  new RegExp(source, flags)
}

class QueryParser {
  private index = 0
  private depth = 0
  constructor(private readonly tokens: readonly Token[]) {}
  private peek(): Token | undefined {
    return this.tokens[this.index]
  }
  private take(): Token {
    const token = this.tokens[this.index++]
    if (!token) throw new SyntaxError('expected a search expression')
    return token
  }
  private atStop(stops: readonly string[]): boolean {
    const token = this.peek()
    return token === undefined || stops.includes(token.kind)
  }
  private combine(kind: 'and' | 'or', children: QueryExpression[]): QueryExpression {
    return children.length === 1 ? children[0] : { kind, children }
  }
  parse(stops: readonly string[] = []): QueryExpression {
    const children = [this.and(stops)]
    while (this.peek()?.kind === 'word' && this.peek()?.value === 'OR') {
      this.take()
      children.push(this.and(stops))
    }
    return this.combine('or', children)
  }
  private and(stops: readonly string[]): QueryExpression {
    const children: QueryExpression[] = []
    while (!this.atStop(stops) && !(this.peek()?.kind === 'word' && this.peek()?.value === 'OR')) {
      if (this.peek()?.kind === 'word' && this.peek()?.value === 'AND') {
        if (!children.length) throw new SyntaxError('AND requires a preceding expression')
        this.take()
        if (this.atStop(stops) || this.peek()?.value === 'OR')
          throw new SyntaxError('AND requires a following expression')
      }
      children.push(this.unary(stops))
    }
    if (!children.length) throw new SyntaxError('expected a search expression')
    return this.combine('and', children)
  }
  private unary(stops: readonly string[] = []): QueryExpression {
    if (++this.depth > 32) throw new SyntaxError('query exceeds 32 nesting levels')
    const token = this.take()
    let result: QueryExpression
    if (token.kind === '-') result = { kind: 'not', child: this.unary(stops) }
    else if (token.kind === '(') {
      result = this.parse([')'])
      if (this.take().kind !== ')') throw new SyntaxError('unclosed group')
    } else if (token.kind === '[') {
      const key = this.parse([':', ']'])
      let value: QueryExpression | undefined
      if (this.peek()?.kind === ':') {
        this.take()
        value = this.parse([']'])
      }
      if (this.take().kind !== ']') throw new SyntaxError('unclosed property expression')
      result = { kind: 'property', key, ...(value ? { value } : {}) }
    } else if (token.kind === 'word' && this.peek()?.kind === ':' && !stops.includes(':')) {
      this.take()
      if (!isScope(token.value))
        throw new SyntaxError(`unsupported search operator: ${token.value}`)
      result = { kind: 'scope', scope: token.value, child: this.unary() }
    } else if (token.kind === 'word' || token.kind === 'phrase') {
      if (token.kind === 'word' && (token.value === 'AND' || token.value === 'OR'))
        throw new SyntaxError(`unexpected ${token.value}`)
      result = { kind: 'term', value: token.value }
    } else if (token.kind === 'regex')
      result = { kind: 'regex', source: token.value, flags: token.flags ?? '' }
    else throw new SyntaxError(`unexpected ${token.value}`)
    this.depth--
    return result
  }
  finish(): QueryExpression {
    const expression = this.parse()
    if (this.index !== this.tokens.length) throw new SyntaxError(`unexpected ${this.peek()?.value}`)
    return expression
  }
}

function validateScopes(
  node: QueryExpression,
  context: 'document' | 'text' | 'property' = 'document',
): void {
  if (node.kind === 'and' || node.kind === 'or')
    node.children.forEach(child => validateScopes(child, context))
  else if (node.kind === 'not') validateScopes(node.child, context)
  else if (node.kind === 'scope') {
    if (node.scope === 'match-case' || node.scope === 'ignore-case')
      validateScopes(node.child, context)
    else {
      if (context !== 'document')
        throw new SyntaxError(
          'nested text scopes are unsupported; combine scopes at document level',
        )
      validateScopes(node.child, 'text')
    }
  } else if (node.kind === 'property') {
    if (context !== 'document')
      throw new SyntaxError(
        'nested property scopes are unsupported; combine properties at document level',
      )
    validateScopes(node.key, 'property')
    if (node.value) validateScopes(node.value, 'property')
  }
}

export function compileEmbeddedQuery(query: string): CompiledQuery {
  try {
    const tokens = tokenize(query)
    if (!tokens.length) return { kind: 'error', message: 'query is empty' }
    const expression = new QueryParser(tokens).finish()
    validateScopes(expression)
    return { kind: 'compiled', expression }
  } catch (error) {
    return { kind: 'error', message: error instanceof Error ? error.message : 'invalid query' }
  }
}
