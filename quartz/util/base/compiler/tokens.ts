import { Span } from "./ast"

export type Operator =
  | "=="
  | "!="
  | ">="
  | "<="
  | ">"
  | "<"
  | "&&"
  | "||"
  | "+"
  | "-"
  | "*"
  | "/"
  | "%"
  | "!"

export type Punctuation = "." | "," | "(" | ")" | "[" | "]"

export type Token =
  | { type: "number"; value: number; span: Span }
  | { type: "string"; value: string; span: Span }
  | { type: "boolean"; value: boolean; span: Span }
  | { type: "null"; span: Span }
  | { type: "identifier"; value: string; span: Span }
  | { type: "this"; span: Span }
  | { type: "operator"; value: Operator; span: Span }
  | { type: "punctuation"; value: Punctuation; span: Span }
  | { type: "regex"; pattern: string; flags: string; span: Span }
  | { type: "eof"; span: Span }
