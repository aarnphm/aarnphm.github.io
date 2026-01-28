import { Expr } from "./ast"

export type BasesExpressions = {
  filters?: Expr
  viewFilters: Record<string, Expr>
  formulas: Record<string, Expr>
  summaries: Record<string, Expr>
  viewSummaries: Record<string, Record<string, Expr>>
  propertyExpressions: Record<string, Expr>
}
