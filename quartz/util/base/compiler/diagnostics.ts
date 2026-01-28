import { Span } from "./ast"

export type BaseExpressionDiagnostic = {
  kind: "lex" | "parse"
  message: string
  span: Span
  context: string
  source: string
}
