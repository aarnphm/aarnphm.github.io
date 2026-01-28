export type Position = {
  offset: number
  line: number
  column: number
}

export type Span = {
  start: Position
  end: Position
  file?: string
}

export type Program = {
  type: "Program"
  body: Expr | null
  span: Span
}

export type Expr =
  | Literal
  | Identifier
  | UnaryExpr
  | BinaryExpr
  | LogicalExpr
  | CallExpr
  | MemberExpr
  | IndexExpr
  | ListExpr
  | ErrorExpr

export type Literal =
  | { type: "Literal"; kind: "number"; value: number; span: Span }
  | { type: "Literal"; kind: "string"; value: string; span: Span }
  | { type: "Literal"; kind: "boolean"; value: boolean; span: Span }
  | { type: "Literal"; kind: "null"; value: null; span: Span }
  | { type: "Literal"; kind: "date"; value: string; span: Span }
  | { type: "Literal"; kind: "duration"; value: string; span: Span }
  | { type: "Literal"; kind: "regex"; value: string; flags: string; span: Span }

export type Identifier = {
  type: "Identifier"
  name: string
  span: Span
}

export type UnaryExpr = {
  type: "UnaryExpr"
  operator: "!" | "-"
  argument: Expr
  span: Span
}

export type BinaryExpr = {
  type: "BinaryExpr"
  operator: "+" | "-" | "*" | "/" | "%" | "==" | "!=" | ">" | ">=" | "<" | "<="
  left: Expr
  right: Expr
  span: Span
}

export type LogicalExpr = {
  type: "LogicalExpr"
  operator: "&&" | "||"
  left: Expr
  right: Expr
  span: Span
}

export type CallExpr = {
  type: "CallExpr"
  callee: Expr
  args: Expr[]
  span: Span
}

export type MemberExpr = {
  type: "MemberExpr"
  object: Expr
  property: string
  span: Span
}

export type IndexExpr = {
  type: "IndexExpr"
  object: Expr
  index: Expr
  span: Span
}

export type ListExpr = {
  type: "ListExpr"
  elements: Expr[]
  span: Span
}

export type ErrorExpr = {
  type: "ErrorExpr"
  message: string
  span: Span
}

export function spanFrom(start: Span, end: Span): Span {
  return {
    start: start.start,
    end: end.end,
    file: start.file || end.file,
  }
}
