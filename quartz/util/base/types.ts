import { BaseFilter } from "./query"
import { QuartzPluginData } from "../../plugins/vfile"

// comparison operators for filter expressions
export type ComparisonOp = "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains" | "!contains"

// base file structure
export interface BaseFile {
  filters: BaseFilter
  views: BaseView[]
  properties?: Record<string, PropertyConfig>
  summaries?: Record<string, string>
}

export interface PropertyConfig {
  displayName?: string
}

export interface BaseView {
  type: "table" | "list" | "gallery" | "board" | "calendar" | "card" | "cards"
  name: string
  order?: string[] // column IDs for table view
  sort?: BaseSortConfig[]
  columnSize?: Record<string, number>
  groupBy?: string | BaseGroupBy
  limit?: number
  filters?: BaseFilter
  // optional/custom view options (kept flexible for forward compatibility)
  // common ones used in this repo:
  image?: string
  cardSize?: number
  nestedProperties?: boolean
  [key: string]: any
}

export interface BaseSortConfig {
  property: string
  direction: "ASC" | "DESC"
}

export interface BaseGroupBy {
  property: string
  direction: "ASC" | "DESC"
}

// resolved data for rendering
export interface BaseTableData {
  view: BaseView
  rows: QuartzPluginData[]
  columns: string[]
}

// tokenize arithmetic expression
type Token =
  | { type: "number"; value: number }
  | { type: "property"; value: string }
  | { type: "operator"; value: "+" | "-" | "*" | "/" | "%" }
  | { type: "lparen" }
  | { type: "rparen" }

function tokenizeExpression(expr: string): Token[] {
  const tokens: Token[] = []
  let i = 0

  while (i < expr.length) {
    const char = expr[i]

    // skip whitespace
    if (/\s/.test(char)) {
      i++
      continue
    }

    // operators
    if (["+", "-", "*", "/", "%"].includes(char)) {
      tokens.push({ type: "operator", value: char as "+" | "-" | "*" | "/" | "%" })
      i++
      continue
    }

    // parentheses
    if (char === "(") {
      tokens.push({ type: "lparen" })
      i++
      continue
    }

    if (char === ")") {
      tokens.push({ type: "rparen" })
      i++
      continue
    }

    // numbers (including decimals)
    if (/\d/.test(char)) {
      let numStr = ""
      while (i < expr.length && /[\d.]/.test(expr[i])) {
        numStr += expr[i]
        i++
      }
      tokens.push({ type: "number", value: parseFloat(numStr) })
      continue
    }

    // property references (alphanumeric, dots, underscores)
    if (/[a-zA-Z_]/.test(char)) {
      let propStr = ""
      while (i < expr.length && /[a-zA-Z0-9_.]/.test(expr[i])) {
        propStr += expr[i]
        i++
      }
      tokens.push({ type: "property", value: propStr })
      continue
    }

    throw new Error(`Unexpected character in expression: ${char}`)
  }

  return tokens
}

// recursive descent parser for arithmetic expressions
// grammar:
//   expr    -> term (('+' | '-') term)*
//   term    -> factor (('*' | '/' | '%') factor)*
//   factor  -> number | property | '(' expr ')'
class ExpressionParser {
  private tokens: Token[]
  private pos: number

  constructor(tokens: Token[]) {
    this.tokens = tokens
    this.pos = 0
  }

  private current(): Token | undefined {
    return this.tokens[this.pos]
  }

  private advance(): void {
    this.pos++
  }

  parse(): (props: Record<string, any>) => number {
    const tree = this.parseExpr()
    return tree
  }

  private parseExpr(): (props: Record<string, any>) => number {
    let left = this.parseTerm()

    while (this.current()?.type === "operator") {
      const op = this.current() as { type: "operator"; value: "+" | "-" | "*" | "/" | "%" }
      if (op.value !== "+" && op.value !== "-") break

      this.advance()
      const right = this.parseTerm()
      const operator = op.value

      const prevLeft = left
      left = (props: Record<string, any>) => {
        const leftVal = prevLeft(props)
        const rightVal = right(props)
        if (operator === "+") return leftVal + rightVal
        if (operator === "-") return leftVal - rightVal
        throw new Error(`Unexpected operator: ${operator}`)
      }
    }

    return left
  }

  private parseTerm(): (props: Record<string, any>) => number {
    let left = this.parseFactor()

    while (this.current()?.type === "operator") {
      const op = this.current() as { type: "operator"; value: "+" | "-" | "*" | "/" | "%" }
      if (op.value !== "*" && op.value !== "/" && op.value !== "%") break

      this.advance()
      const right = this.parseFactor()
      const operator = op.value

      const prevLeft = left
      left = (props: Record<string, any>) => {
        const leftVal = prevLeft(props)
        const rightVal = right(props)
        if (operator === "*") return leftVal * rightVal
        if (operator === "/") return leftVal / rightVal
        if (operator === "%") return leftVal % rightVal
        throw new Error(`Unexpected operator: ${operator}`)
      }
    }

    return left
  }

  private parseFactor(): (props: Record<string, any>) => number {
    const token = this.current()

    if (!token) {
      throw new Error("Unexpected end of expression")
    }

    if (token.type === "number") {
      this.advance()
      const value = token.value
      return () => value
    }

    if (token.type === "property") {
      this.advance()
      const propName = token.value
      return (props: Record<string, any>) => {
        const value = props[propName]
        if (typeof value !== "number") {
          throw new Error(`Property ${propName} is not a number: ${value}`)
        }
        return value
      }
    }

    if (token.type === "lparen") {
      this.advance()
      const expr = this.parseExpr()
      const closeParen = this.current()
      if (!closeParen || closeParen.type !== "rparen") {
        throw new Error("Expected closing parenthesis")
      }
      this.advance()
      return expr
    }

    throw new Error(`Unexpected token: ${JSON.stringify(token)}`)
  }
}

// compile arithmetic expression to evaluator function
export function compileExpression(expr: string): (props: Record<string, any>) => number {
  const tokens = tokenizeExpression(expr)
  const parser = new ExpressionParser(tokens)
  return parser.parse()
}

// check if string contains arithmetic operators
function hasArithmeticOperators(str: string): boolean {
  return /[+\-*/%()]/.test(str)
}

// parse value from string to proper type
// note: now() and today() return Date objects (not timestamps) for compatibility with date arithmetic
function parseValue(val: string): string | number | boolean | Date {
  const trimmed = val.trim()

  // handle function calls
  const funcMatch = trimmed.match(/^(now|today|date)\(/)
  if (funcMatch) {
    const funcName = funcMatch[1]
    if (funcName === "now") {
      // return Date object representing current moment
      return new Date()
    }
    if (funcName === "today") {
      // return Date object representing today at midnight
      const d = new Date()
      d.setHours(0, 0, 0, 0)
      return d
    }
    if (funcName === "date") {
      // extract argument from date("2023-01-01")
      const argMatch = trimmed.match(/^date\(['"](.+)['"]\)$/)
      if (argMatch) {
        const dateStr = argMatch[1]
        const d = new Date(dateStr)
        if (!isNaN(d.getTime())) {
          return d
        }
      }
    }
  }

  // handle quoted strings
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    const unquoted = trimmed.slice(1, -1)

    // try parsing as ISO date
    if (/^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?)?$/.test(unquoted)) {
      const timestamp = new Date(unquoted).getTime()
      // check for invalid date
      if (!isNaN(timestamp)) {
        return timestamp
      }
    }

    return unquoted
  }
  // handle booleans
  if (trimmed === "true") return true
  if (trimmed === "false") return false
  // handle numbers
  const num = Number(trimmed)
  if (!isNaN(num)) return num

  // handle unquoted ISO dates (for backward compatibility)
  if (/^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?)?$/.test(trimmed)) {
    const timestamp = new Date(trimmed).getTime()
    if (!isNaN(timestamp)) {
      return timestamp
    }
  }

  // fallback to string
  return trimmed
}

// tokenize filter expression for boolean operators
function tokenizeFilter(expr: string): string[] {
  const tokens: string[] = []
  let current = ""
  let parenDepth = 0
  let inQuotes = false
  let quoteChar = ""

  for (let i = 0; i < expr.length; i++) {
    const char = expr[i]
    const next = expr[i + 1]

    // track quote state
    if ((char === '"' || char === "'") && (i === 0 || expr[i - 1] !== "\\")) {
      if (!inQuotes) {
        inQuotes = true
        quoteChar = char
        current += char
      } else if (char === quoteChar) {
        inQuotes = false
        quoteChar = ""
        current += char
      } else {
        current += char
      }
      continue
    }

    // skip processing inside quotes
    if (inQuotes) {
      current += char
      continue
    }

    // track parentheses
    if (char === "(") {
      parenDepth++
      current += char
      continue
    }
    if (char === ")") {
      parenDepth--
      current += char
      continue
    }

    // detect && and || operators at depth 0
    if (parenDepth === 0) {
      if (char === "&" && next === "&") {
        if (current.trim()) tokens.push(current.trim())
        tokens.push("&&")
        current = ""
        i++ // skip next &
        continue
      }
      if (char === "|" && next === "|") {
        if (current.trim()) tokens.push(current.trim())
        tokens.push("||")
        current = ""
        i++ // skip next |
        continue
      }
    }

    current += char
  }

  if (current.trim()) {
    tokens.push(current.trim())
  }

  return tokens
}

// parse filter with inline boolean operators
function parseFilterWithBoolean(expr: string): BaseFilter {
  const tokens = tokenizeFilter(expr)

  // no boolean operators found
  if (tokens.length === 1) {
    return parseFilterAtom(tokens[0])
  }

  // parse with precedence: || has lower precedence than &&
  // first split by ||, then recursively split by &&
  const orGroups: string[][] = [[]]
  let currentGroup = 0

  for (const token of tokens) {
    if (token === "||") {
      currentGroup++
      orGroups[currentGroup] = []
    } else {
      orGroups[currentGroup].push(token)
    }
  }

  // if we have multiple OR groups
  if (orGroups.length > 1) {
    return {
      type: "or",
      conditions: orGroups.map((group) => parseAndGroup(group)),
    }
  }

  // single group, parse as AND
  return parseAndGroup(orGroups[0])
}

// parse a group of tokens connected by &&
function parseAndGroup(tokens: string[]): BaseFilter {
  const andConditions: BaseFilter[] = []

  for (const token of tokens) {
    if (token !== "&&") {
      andConditions.push(parseFilterAtom(token))
    }
  }

  if (andConditions.length === 1) {
    return andConditions[0]
  }

  return {
    type: "and",
    conditions: andConditions,
  }
}

// parse a single atomic filter expression (no boolean operators)
function parseFilterAtom(raw: string): BaseFilter {
  const trimmed = raw.trim()

  // check for negation prefix FIRST (before parentheses)
  const negated = trimmed.startsWith("!")
  const cleanRaw = negated ? trimmed.slice(1).trim() : trimmed

  // handle parentheses grouping (after negation check)
  if (cleanRaw.startsWith("(") && cleanRaw.endsWith(")")) {
    const inner = cleanRaw.slice(1, -1)
    const innerFilter = parseFilterWithBoolean(inner)
    // if negated, wrap in not
    if (negated) {
      return {
        type: "not",
        conditions: [innerFilter],
      }
    }
    return innerFilter
  }

  // try method call: 'property.method("arg1", "arg2")' or '!property.method("arg")'
  const methodMatch = cleanRaw.match(/^([\w.-]+)\.([\w]+)\((.*)\)$/)
  if (methodMatch) {
    const [, property, method, argsStr] = methodMatch

    // check if this is a file.* function (special case)
    if (property === "file") {
      const fullName = `${property}.${method}`
      const args = argsStr
        .split(",")
        .map((s) => s.trim().replace(/^["']|["']$/g, ""))
        .filter(Boolean)

      if (negated) {
        return {
          type: "not",
          conditions: [
            {
              type: "function",
              name: fullName,
              args,
            },
          ],
        }
      }

      return {
        type: "function",
        name: fullName,
        args,
      }
    }

    // regular method call on property
    const args = argsStr
      .split(",")
      .map((s) => s.trim().replace(/^["']|["']$/g, ""))
      .filter(Boolean)
    return {
      type: "method",
      property,
      method,
      args,
      negated,
    }
  }

  // try function call: 'file.hasTag("book", "article")'
  const funcMatch = cleanRaw.match(/^([\w.]+)\((.*)\)$/)
  if (funcMatch) {
    const [, name, argsStr] = funcMatch
    const args = argsStr
      .split(",")
      .map((s) => s.trim().replace(/^["']|["']$/g, ""))
      .filter(Boolean)

    if (negated) {
      return {
        type: "not",
        conditions: [
          {
            type: "function",
            name,
            args,
          },
        ],
      }
    }

    return {
      type: "function",
      name,
      args,
    }
  }

  // try comparison expression: 'property == "value"' or 'status != "finished"'
  // also support arithmetic: 'price * 1.13 > 50' or '(end - start) / 86400000 > 7'
  const comparisonMatch = trimmed.match(/^(.+?)\s*(==|!=|>=|<=|>|<|contains|!contains)\s*(.+)$/)
  if (comparisonMatch) {
    const [, leftStr, operator, rightStr] = comparisonMatch
    const left = leftStr.trim()
    const right = rightStr.trim()

    // check if left side has arithmetic operators
    if (hasArithmeticOperators(left)) {
      // parse right side as value (must be constant)
      const value = parseValue(right)
      return {
        type: "comparison",
        property: left,
        operator: operator.trim() as ComparisonOp,
        value,
        isExpression: true,
      }
    }

    // standard property comparison
    const value = parseValue(right)
    return {
      type: "comparison",
      property: left,
      operator: operator.trim() as ComparisonOp,
      value,
    }
  }

  // handle implicit boolean property reference: 'deleted' or '!deleted'
  // these are shorthand for 'deleted == true' or 'deleted == false'
  if (negated) {
    // !propertyName -> propertyName == false
    if (/^[a-zA-Z_][\w.]*$/.test(cleanRaw)) {
      return {
        type: "comparison",
        property: cleanRaw,
        operator: "==",
        value: false,
      }
    }
  } else {
    // propertyName -> propertyName == true (but only if no operators found)
    if (/^[a-zA-Z_][\w.]*$/.test(trimmed)) {
      return {
        type: "comparison",
        property: trimmed,
        operator: "==",
        value: true,
      }
    }
  }

  throw new Error(`Invalid filter string: ${trimmed}`)
}

// parse raw YAML object into typed BaseFilter
export function parseFilter(raw: any): BaseFilter {
  // handle string-based expressions
  if (typeof raw === "string") {
    // check if expression contains inline boolean operators (&& or ||)
    if (raw.includes("&&") || raw.includes("||")) {
      return parseFilterWithBoolean(raw)
    }

    // fallback to atomic parsing
    return parseFilterAtom(raw)
  }

  if (!raw || typeof raw !== "object") {
    throw new Error("Invalid filter: must be an object or string")
  }

  // check for logical operators
  if ("and" in raw && Array.isArray(raw.and)) {
    return {
      type: "and",
      conditions: raw.and.map(parseFilter),
    }
  }

  if ("or" in raw && Array.isArray(raw.or)) {
    return {
      type: "or",
      conditions: raw.or.map(parseFilter),
    }
  }

  if ("not" in raw && Array.isArray(raw.not)) {
    return {
      type: "not",
      conditions: raw.not.map(parseFilter),
    }
  }

  throw new Error(`Invalid filter structure: ${JSON.stringify(raw)}`)
}

// parse YAML views array
export function parseViews(raw: any): BaseView[] {
  if (!Array.isArray(raw)) {
    throw new Error("Views must be an array")
  }

  return raw.map((v) => {
    if (!v || typeof v !== "object") {
      throw new Error("Each view must be an object")
    }

    if (!v.type || !v.name) {
      throw new Error("Each view must have 'type' and 'name' fields")
    }

    // Preserve any additional properties on the view (e.g., image, cardSize, nestedProperties)
    // while normalizing the known/parsed fields.
    const parsed: BaseView = {
      ...v,
      type: v.type,
      name: v.name,
      order: v.order,
      sort: v.sort,
      columnSize: v.columnSize,
      groupBy: v.groupBy,
      limit: v.limit,
      filters: v.filters ? parseFilter(v.filters) : undefined,
    }

    return parsed
  })
}
