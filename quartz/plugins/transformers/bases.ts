import yaml from "js-yaml"
import { Root } from "mdast"
import { QuartzTransformerPlugin } from "../../types/plugin"
import {
  parseFilter,
  parseViews,
  parseFormulas,
  BaseFile,
  PropertyConfig,
} from "../../util/base/types"
import {
  parseExpressionSource,
  BaseExpressionDiagnostic,
  BasesExpressions,
} from "../../util/base/compiler"
import { Expr, LogicalExpr, UnaryExpr, spanFrom } from "../../util/base/compiler/ast"

export const ObsidianBases: QuartzTransformerPlugin = () => {
  return {
    name: "ObsidianBases",
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file) => {
            // Detect .base files by extension
            const isBaseFile = file.path?.endsWith(".base")

            if (!isBaseFile) {
              file.data.bases = false
              return
            }

            file.data.bases = true

            // Parse YAML and store config for emitter to use
            const parsed = yaml.load(String(file.value)) as any

            const normalizeProperties = (
              raw: object,
            ): Record<string, PropertyConfig> | undefined => {
              const normalized: Record<string, PropertyConfig> = {}

              for (const [key, value] of Object.entries(raw)) {
                if (!value || typeof value !== "object") {
                  continue
                }

                const propConfig = value as PropertyConfig
                normalized[key] = propConfig

                const withoutPrefix = key.replace(/^(?:note|file)\./, "")
                if (withoutPrefix !== key && !(withoutPrefix in normalized)) {
                  normalized[withoutPrefix] = propConfig
                }

                const segments = withoutPrefix.split(".")
                const lastSegment = segments[segments.length - 1]
                if (lastSegment && !(lastSegment in normalized)) {
                  normalized[lastSegment] = propConfig
                }
              }

              return Object.keys(normalized).length > 0 ? normalized : undefined
            }

            const properties = normalizeProperties(parsed.properties)

            const diagnostics: BaseExpressionDiagnostic[] = []
            const expressions: BasesExpressions = {
              viewFilters: {},
              formulas: {},
              summaries: {},
              viewSummaries: {},
            }
            const isRecord = (value: unknown): value is Record<string, unknown> =>
              typeof value === "object" && value !== null && !Array.isArray(value)
            const builtinSummaries = new Set(
              [
                "average",
                "avg",
                "min",
                "max",
                "sum",
                "range",
                "median",
                "stddev",
                "earliest",
                "latest",
                "checked",
                "unchecked",
                "empty",
                "filled",
                "unique",
              ].map((value) => value.toLowerCase()),
            )
            const summaryKeys = parsed.summaries ? Object.keys(parsed.summaries) : []

            const addExpressionDiagnostics = (source: string, context: string) => {
              const result = parseExpressionSource(source, file.path)
              if (result.diagnostics.length === 0) {
                return
              }
              for (const diagnostic of result.diagnostics) {
                diagnostics.push({
                  kind: diagnostic.kind,
                  message: diagnostic.message,
                  span: diagnostic.span,
                  context,
                  source,
                })
              }
            }

            const parseExpression = (source: string): Expr | null => {
              const result = parseExpressionSource(source, file.path)
              if (!result.program.body) return null
              return result.program.body
            }

            const buildLogical = (
              operator: "&&" | "||",
              expressionsList: Expr[],
            ): Expr | null => {
              if (expressionsList.length === 0) return null
              let current: Expr | null = null
              for (const next of expressionsList) {
                if (!current) {
                  current = next
                  continue
                }
                const span = spanFrom(current.span, next.span)
                const node: LogicalExpr = {
                  type: "LogicalExpr",
                  operator,
                  left: current,
                  right: next,
                  span,
                }
                current = node
              }
              return current
            }

            const negateExpressions = (expressionsList: Expr[]): Expr[] =>
              expressionsList.map((expr) => {
                const node: UnaryExpr = {
                  type: "UnaryExpr",
                  operator: "!",
                  argument: expr,
                  span: spanFrom(expr.span, expr.span),
                }
                return node
              })

            const buildFilterExpr = (raw: unknown, context: string): Expr | null => {
              if (typeof raw === "string") {
                addExpressionDiagnostics(raw, context)
                return parseExpression(raw)
              }
              if (!isRecord(raw)) return null
              const record = raw
              if (Array.isArray(record.and)) {
                const parts = record.and
                  .map((entry, index) => buildFilterExpr(entry, `${context}.and[${index}]`))
                  .filter((entry): entry is Expr => Boolean(entry))
                return buildLogical("&&", parts)
              }
              if (Array.isArray(record.or)) {
                const parts = record.or
                  .map((entry, index) => buildFilterExpr(entry, `${context}.or[${index}]`))
                  .filter((entry): entry is Expr => Boolean(entry))
                return buildLogical("||", parts)
              }
              if (Array.isArray(record.not)) {
                const parts = record.not
                  .map((entry, index) => buildFilterExpr(entry, `${context}.not[${index}]`))
                  .filter((entry): entry is Expr => Boolean(entry))
                return buildLogical("&&", negateExpressions(parts))
              }
              return null
            }

            const walkSummaries = (raw: unknown, context: string) => {
              if (!isRecord(raw)) return
              const columns =
                "columns" in raw && isRecord(raw.columns) ? raw.columns : raw
              for (const [key, value] of Object.entries(columns)) {
                if (typeof value !== "string") continue
                const normalized = value.toLowerCase().trim()
                if (builtinSummaries.has(normalized)) continue
                if (summaryKeys.includes(value)) continue
                addExpressionDiagnostics(value, `${context}.${key}`)
              }
            }

            if (parsed.filters) {
              const expr = buildFilterExpr(parsed.filters, "filters")
              if (expr) {
                expressions.filters = expr
              }
            }

            if (parsed.views && Array.isArray(parsed.views)) {
              parsed.views.forEach((view, index) => {
                if (!isRecord(view)) return
                if (view.filters) {
                  const expr = buildFilterExpr(view.filters, `views[${index}].filters`)
                  if (expr) {
                    expressions.viewFilters[String(index)] = expr
                  }
                }
                if (view.summaries) {
                  walkSummaries(view.summaries, `views[${index}].summaries`)
                }
              })
            }

            if (parsed.formulas && typeof parsed.formulas === "object") {
              for (const [name, expression] of Object.entries(parsed.formulas)) {
                if (typeof expression === "string") {
                  addExpressionDiagnostics(expression, `formulas.${name}`)
                  const expr = parseExpression(expression)
                  if (expr) {
                    expressions.formulas[name] = expr
                  }
                }
              }
            }

            if (parsed.summaries && typeof parsed.summaries === "object") {
              for (const [name, expression] of Object.entries(parsed.summaries)) {
                if (typeof expression === "string") {
                  addExpressionDiagnostics(expression, `summaries.${name}`)
                  const expr = parseExpression(expression)
                  if (expr) {
                    expressions.summaries[name] = expr
                  }
                }
              }
            }

            if (parsed.views && Array.isArray(parsed.views)) {
              parsed.views.forEach((view, index) => {
                if (!isRecord(view) || !view.summaries) return
                const summaries = view.summaries
                if (!isRecord(summaries)) return
                const columns =
                  "columns" in summaries && isRecord(summaries.columns)
                    ? summaries.columns
                    : summaries
                const viewKey = String(index)
                const viewMap: Record<string, Expr> = {}
                for (const [column, summaryValue] of Object.entries(columns)) {
                  if (typeof summaryValue !== "string") continue
                  const normalized = summaryValue.toLowerCase().trim()
                  if (builtinSummaries.has(normalized)) continue
                  const topLevel = expressions.summaries[summaryValue]
                  if (topLevel) {
                    viewMap[column] = topLevel
                    continue
                  }
                  const expr = parseExpression(summaryValue)
                  if (expr) {
                    viewMap[column] = expr
                  }
                }
                if (Object.keys(viewMap).length > 0) {
                  expressions.viewSummaries[viewKey] = viewMap
                }
              })
            }

            const config: BaseFile = {
              filters: parseFilter(parsed.filters ?? { and: [] }),
              views: parseViews(parsed.views),
              properties,
              summaries: parsed.summaries,
              formulas: parseFormulas(parsed.formulas),
            }
            file.data.basesConfig = config
            file.data.basesDiagnostics = diagnostics
            file.data.basesExpressions = expressions

            tree.children = []

            file.data.frontmatter = {
              title: file.path?.replace(".base", "").split("/").pop() || "",
              pageLayout: "default" as const,
              description: `bases renderer of ${file.data.slug}`,
              tags: ["bases"],
            }
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    bases?: boolean
    basesConfig?: BaseFile
    basesDiagnostics?: BaseExpressionDiagnostic[]
    basesExpressions?: BasesExpressions
  }
}
