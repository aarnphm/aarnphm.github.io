import yaml from "js-yaml"
import fs from "node:fs/promises"
import path from "node:path"
import {
  parseExpressionSource,
  compileExpression,
  buildPropertyExpressionSource,
  BUILTIN_SUMMARY_TYPES,
} from "../util/base/compiler"
import { Expr, LogicalExpr, UnaryExpr, spanFrom } from "../util/base/compiler/ast"
import { Diagnostic } from "../util/base/compiler/errors"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

type CollectedExpression = {
  kind: string
  context: string
  source: string
  ast: Expr | null
  ir: unknown
  diagnostics: Diagnostic[]
}

const parseToExpr = (source: string, filePath: string) => {
  const result = parseExpressionSource(source, filePath)
  return { expr: result.program.body ?? null, diagnostics: result.diagnostics }
}

const buildLogical = (operator: "&&" | "||", expressionsList: Expr[]): Expr | null => {
  if (expressionsList.length === 0) return null
  let current: Expr | null = null
  for (const next of expressionsList) {
    if (!current) {
      current = next
      continue
    }
    const span = spanFrom(current.span, next.span)
    const node: LogicalExpr = { type: "LogicalExpr", operator, left: current, right: next, span }
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

const buildFilterExpr = (
  raw: unknown,
  context: string,
  diagnostics: Diagnostic[],
  filePath: string,
): Expr | null => {
  if (typeof raw === "string") {
    const parsed = parseToExpr(raw, filePath)
    diagnostics.push(...parsed.diagnostics)
    return parsed.expr
  }
  if (!isRecord(raw)) return null
  if (Array.isArray(raw.and)) {
    const parts = raw.and
      .map((entry, index) =>
        buildFilterExpr(entry, `${context}.and[${index}]`, diagnostics, filePath),
      )
      .filter((entry): entry is Expr => Boolean(entry))
    return buildLogical("&&", parts)
  }
  if (Array.isArray(raw.or)) {
    const parts = raw.or
      .map((entry, index) =>
        buildFilterExpr(entry, `${context}.or[${index}]`, diagnostics, filePath),
      )
      .filter((entry): entry is Expr => Boolean(entry))
    return buildLogical("||", parts)
  }
  if (Array.isArray(raw.not)) {
    const parts = raw.not
      .map((entry, index) =>
        buildFilterExpr(entry, `${context}.not[${index}]`, diagnostics, filePath),
      )
      .filter((entry): entry is Expr => Boolean(entry))
    return buildLogical("&&", negateExpressions(parts))
  }
  return null
}

const collectPropertyExpressions = (
  views: unknown[],
): Map<string, { source: string; context: string }> => {
  const entries = new Map<string, { source: string; context: string }>()
  const addProperty = (property: string, context: string) => {
    const key = property.trim()
    if (!key || entries.has(key)) return
    const source = buildPropertyExpressionSource(key)
    if (!source) return
    entries.set(key, { source, context })
  }

  views.forEach((view, viewIndex) => {
    if (!isRecord(view)) return
    const viewContext = `views[${viewIndex}]`
    if (Array.isArray(view.order)) {
      view.order.forEach((entry, orderIndex) => {
        if (typeof entry === "string") {
          addProperty(entry, `${viewContext}.order[${orderIndex}]`)
        }
      })
    }

    if (Array.isArray(view.sort)) {
      view.sort.forEach((entry, sortIndex) => {
        if (isRecord(entry) && typeof entry.property === "string") {
          addProperty(entry.property, `${viewContext}.sort[${sortIndex}].property`)
        }
      })
    }

    if (typeof view.groupBy === "string") {
      addProperty(view.groupBy, `${viewContext}.groupBy`)
    } else if (isRecord(view.groupBy) && typeof view.groupBy.property === "string") {
      addProperty(view.groupBy.property, `${viewContext}.groupBy.property`)
    }

    if (view.summaries && isRecord(view.summaries)) {
      const columns =
        "columns" in view.summaries && isRecord(view.summaries.columns)
          ? view.summaries.columns
          : view.summaries
      for (const key of Object.keys(columns)) {
        addProperty(key, `${viewContext}.summaries.${key}`)
      }
    }

    if (typeof view.image === "string") {
      addProperty(view.image, `${viewContext}.image`)
    }

    if (view.type === "map") {
      const coords = typeof view.coordinates === "string" ? view.coordinates : "coordinates"
      addProperty(coords, `${viewContext}.coordinates`)
      if (typeof view.markerIcon === "string") {
        addProperty(view.markerIcon, `${viewContext}.markerIcon`)
      }
      if (typeof view.markerColor === "string") {
        addProperty(view.markerColor, `${viewContext}.markerColor`)
      }
    }
  })

  return entries
}

const main = async () => {
  const inputPath = process.argv[2] ? String(process.argv[2]) : "content/antilibrary.base"
  const filePath = path.resolve(process.cwd(), inputPath)
  const raw = await fs.readFile(filePath, "utf8")
  const parsed = yaml.load(raw)
  const config = isRecord(parsed) ? parsed : {}

  const collected: CollectedExpression[] = []

  if (config.filters !== undefined) {
    const diagnostics: Diagnostic[] = []
    const expr = buildFilterExpr(config.filters, "filters", diagnostics, filePath)
    collected.push({
      kind: "filters",
      context: "filters",
      source: typeof config.filters === "string" ? config.filters : JSON.stringify(config.filters),
      ast: expr,
      ir: expr ? compileExpression(expr) : null,
      diagnostics,
    })
  }

  if (isRecord(config.formulas)) {
    for (const [name, value] of Object.entries(config.formulas)) {
      if (typeof value !== "string") continue
      const parsedExpr = parseToExpr(value, filePath)
      collected.push({
        kind: "formula",
        context: `formulas.${name}`,
        source: value,
        ast: parsedExpr.expr,
        ir: parsedExpr.expr ? compileExpression(parsedExpr.expr) : null,
        diagnostics: parsedExpr.diagnostics,
      })
    }
  }

  const topLevelSummaries = isRecord(config.summaries) ? config.summaries : {}

  if (isRecord(config.summaries)) {
    for (const [name, value] of Object.entries(config.summaries)) {
      if (typeof value !== "string") continue
      const parsedExpr = parseToExpr(value, filePath)
      collected.push({
        kind: "summary",
        context: `summaries.${name}`,
        source: value,
        ast: parsedExpr.expr,
        ir: parsedExpr.expr ? compileExpression(parsedExpr.expr) : null,
        diagnostics: parsedExpr.diagnostics,
      })
    }
  }

  if (Array.isArray(config.views)) {
    config.views.forEach((view, index) => {
      if (!isRecord(view)) return
      if (view.filters !== undefined) {
        const diagnostics: Diagnostic[] = []
        const expr = buildFilterExpr(view.filters, `views[${index}].filters`, diagnostics, filePath)
        collected.push({
          kind: "view.filter",
          context: `views[${index}].filters`,
          source: typeof view.filters === "string" ? view.filters : JSON.stringify(view.filters),
          ast: expr,
          ir: expr ? compileExpression(expr) : null,
          diagnostics,
        })
      }

      if (view.summaries && isRecord(view.summaries)) {
        const columns =
          "columns" in view.summaries && isRecord(view.summaries.columns)
            ? view.summaries.columns
            : view.summaries
        for (const [column, summaryValue] of Object.entries(columns)) {
          if (typeof summaryValue !== "string") continue
          const normalized = summaryValue.toLowerCase().trim()
          const builtins = new Set<string>(BUILTIN_SUMMARY_TYPES)
          if (builtins.has(normalized)) continue
          const summarySource =
            summaryValue in topLevelSummaries && typeof topLevelSummaries[summaryValue] === "string"
              ? String(topLevelSummaries[summaryValue])
              : summaryValue
          const parsedExpr = parseToExpr(summarySource, filePath)
          collected.push({
            kind: "view.summary",
            context: `views[${index}].summaries.${column}`,
            source: summarySource,
            ast: parsedExpr.expr,
            ir: parsedExpr.expr ? compileExpression(parsedExpr.expr) : null,
            diagnostics: parsedExpr.diagnostics,
          })
        }
      }
    })
  }

  const views = Array.isArray(config.views) ? config.views : []
  const propertyExpressions = collectPropertyExpressions(views)
  for (const [_, entry] of propertyExpressions.entries()) {
    const parsedExpr = parseToExpr(entry.source, filePath)
    collected.push({
      kind: "property",
      context: entry.context,
      source: entry.source,
      ast: parsedExpr.expr,
      ir: parsedExpr.expr ? compileExpression(parsedExpr.expr) : null,
      diagnostics: parsedExpr.diagnostics,
    })
  }

  const payload = { file: inputPath, count: collected.length, expressions: collected }

  process.stdout.write(JSON.stringify(payload, null, 2))
}

main()
