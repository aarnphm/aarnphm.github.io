import { Root } from "hast"
import { h } from "hastscript"
import { QuartzPluginData } from "../../plugins/vfile"
import {
  resolveRelative,
  FullSlug,
  joinSegments,
  FilePath,
  slugifyFilePath,
  simplifySlug,
  isAbsoluteURL,
} from "../../util/path"
import { extractWikilinksWithPositions } from "../../util/wikilinks"
import {
  BaseExpressionDiagnostic,
  ProgramIR,
  buildPropertyExpressionSource,
  evaluateExpression,
  evaluateFilterExpression,
  valueToUnknown,
  EvalContext,
  Value,
} from "./compiler"
import { computeViewSummaries } from "./query"
import {
  BaseView,
  BaseGroupBy,
  PropertyConfig,
  ViewSummaryConfig,
  parseViewSummaries,
  BasesConfigFile,
} from "./types"

function getFileBaseName(filePath?: string, slug?: string): string | undefined {
  const source = filePath ?? slug
  if (!source) return undefined
  const fragment = source.split("/").pop() || source
  return fragment.replace(/\.[^/.]+$/, "")
}

function getFileDisplayName(file?: QuartzPluginData): string | undefined {
  if (!file) return undefined
  const title = file.frontmatter?.title
  if (typeof title === "string" && title.length > 0) return title
  const baseFromPath = getFileBaseName(file.filePath as string | undefined)
  if (baseFromPath) return baseFromPath
  const baseFromSlug = getFileBaseName(file.slug)
  if (baseFromSlug) return baseFromSlug.replace(/-/g, " ")
  return undefined
}

function fallbackNameFromSlug(slug: FullSlug): string {
  const base = getFileBaseName(slug) ?? slug
  return base.replace(/-/g, " ")
}

function findFileBySlug(
  allFiles: QuartzPluginData[],
  targetSlug: FullSlug,
): QuartzPluginData | undefined {
  const targetSimple = simplifySlug(targetSlug)
  return allFiles.find(
    (entry) => entry.slug && simplifySlug(entry.slug as FullSlug) === targetSimple,
  )
}

function renderInternalLinkNode(
  targetSlug: FullSlug,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  alias?: string,
  anchor?: string,
) {
  const targetFile = findFileBySlug(allFiles, targetSlug)
  const displayText =
    alias && alias.trim().length > 0
      ? alias.trim()
      : (getFileDisplayName(targetFile) ?? fallbackNameFromSlug(targetSlug))

  const hrefBase = resolveRelative(currentSlug, targetSlug)
  const href = anchor && anchor.length > 0 ? `${hrefBase}${anchor}` : hrefBase
  const dataSlug = anchor && anchor.length > 0 ? `${targetSlug}${anchor}` : targetSlug

  return h("a.internal", { href, "data-slug": dataSlug }, displayText)
}

function splitTargetAndAlias(raw: string): { target: string; alias?: string } {
  let buffer = ""
  let alias: string | undefined
  let escaped = false
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]
    if (escaped) {
      buffer += ch
      escaped = false
      continue
    }
    if (ch === "\\") {
      escaped = true
      continue
    }
    if (ch === "|" && alias === undefined) {
      alias = raw.slice(i + 1)
      break
    }
    buffer += ch
  }

  const target = buffer.replace(/\\\|/g, "|").trim()
  const cleanedAlias = alias?.replace(/\\\|/g, "|").trim()
  return { target, alias: cleanedAlias?.length ? cleanedAlias : undefined }
}

function normalizeTargetSlug(
  target: string,
  currentSlug: FullSlug,
  anchor?: string,
): { slug: FullSlug; anchor?: string } {
  const trimmed = target.trim()
  if (!trimmed) return { slug: currentSlug, anchor }
  const slug = slugifyFilePath(trimmed as FilePath)
  return { slug, anchor }
}

function renderInlineString(
  value: string,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
): (string | ReturnType<typeof h>)[] {
  if (!value.includes("[[")) {
    return [value]
  }

  const nodes: (string | ReturnType<typeof h>)[] = []
  const ranges = extractWikilinksWithPositions(value)
  let lastIndex = 0
  for (const range of ranges) {
    const start = range.start
    if (start > lastIndex) nodes.push(value.slice(lastIndex, start))

    const parsed = range.wikilink
    const raw = value.slice(range.start, range.end)
    if (parsed.embed) {
      nodes.push(raw)
      lastIndex = range.end
      continue
    }

    const { slug, anchor } = normalizeTargetSlug(parsed.target, currentSlug, parsed.anchor)
    nodes.push(renderInternalLinkNode(slug, currentSlug, allFiles, parsed.alias, anchor))
    lastIndex = range.end
  }

  if (lastIndex < value.length) {
    nodes.push(value.slice(lastIndex))
  }

  return nodes
}

function getPropertyDisplayName(
  property: string,
  properties?: Record<string, PropertyConfig>,
): string {
  const candidates: string[] = []

  const addCandidate = (candidate: string | undefined) => {
    if (!candidate) return
    if (!candidates.includes(candidate)) {
      candidates.push(candidate)
    }
  }

  addCandidate(property)

  const withoutPrefix = property.replace(/^(?:note|file)\./, "")
  addCandidate(withoutPrefix)

  if (!property.startsWith("note.")) {
    addCandidate(`note.${property}`)
  }
  if (!property.startsWith("file.")) {
    addCandidate(`file.${property}`)
  }

  addCandidate(withoutPrefix.split(".").pop())

  for (const candidate of candidates) {
    const displayName = properties?.[candidate]?.displayName
    if (displayName && displayName.length > 0) {
      return displayName
    }
  }

  const base = withoutPrefix.length > 0 ? withoutPrefix : property
  return base
    .split(".")
    .pop()!
    .replace(/-/g, " ")
    .replace(/_/g, " ")
    .replace(/([A-Z])/g, " $1")
    .trim()
}

function renderBooleanCheckbox(value: boolean): any {
  return h("input", {
    type: "checkbox",
    checked: value ? true : undefined,
    disabled: true,
    class: "base-checkbox",
  })
}

function buildTableHead(columns: string[], properties?: Record<string, PropertyConfig>): any {
  return h(
    "tr",
    columns.map((col) => h("th", {}, getPropertyDisplayName(col, properties))),
  )
}

type EvalContextFactory = (file: QuartzPluginData) => EvalContext

type PropertyExprGetter = (property: string) => ProgramIR | null

function resolveValueWithFormulas(
  file: QuartzPluginData,
  property: string,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): unknown {
  const expr = getPropertyExpr(property)
  if (!expr) return undefined
  const ctx = getContext(file)
  const cacheKey = property.trim()
  if (cacheKey.length > 0 && ctx.propertyCache?.has(cacheKey)) {
    return valueToUnknown(ctx.propertyCache.get(cacheKey)!)
  }
  ctx.diagnosticContext = `property.${property}`
  ctx.diagnosticSource = buildPropertyExpressionSource(property) ?? property
  const value = evaluateExpression(expr, ctx)
  if (cacheKey.length > 0 && ctx.propertyCache) {
    ctx.propertyCache.set(cacheKey, value)
  }
  return valueToUnknown(value)
}

function buildTableCell(
  file: QuartzPluginData,
  column: string,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): any {
  const fallbackSlugSegment = file.slug?.split("/").pop() || ""
  const fallbackTitle =
    getFileBaseName(file.filePath as string | undefined) ||
    fallbackSlugSegment.replace(/\.[^/.]+$/, "").replace(/-/g, " ")

  if (column === "title" || column === "file.title" || column === "note.title") {
    const titleValue = resolveValueWithFormulas(file, "file.title", getContext, getPropertyExpr)
    const resolvedTitle =
      typeof titleValue === "string" && titleValue.length > 0 ? titleValue : fallbackTitle
    const slug = (file.slug || "") as FullSlug
    const href = resolveRelative(currentSlug, slug)
    return h("td", [h("a.internal", { href, "data-slug": slug }, resolvedTitle)])
  }

  if (column === "file.name") {
    const nameValue = resolveValueWithFormulas(file, "file.name", getContext, getPropertyExpr)
    const resolvedName =
      typeof nameValue === "string" && nameValue.length > 0 ? nameValue : fallbackTitle
    const slug = (file.slug || "") as FullSlug
    const href = resolveRelative(currentSlug, slug)
    return h("td", [h("a.internal", { href, "data-slug": slug }, resolvedName)])
  }

  if (column === "file.links") {
    const links = resolveValueWithFormulas(file, "file.links", getContext, getPropertyExpr)
    const count = Array.isArray(links) ? links.length : 0
    return h("td", {}, String(count))
  }

  if (column === "file.backlinks" || column === "file.inlinks") {
    const backlinks = resolveValueWithFormulas(file, column, getContext, getPropertyExpr)
    if (!Array.isArray(backlinks) || backlinks.length === 0) {
      return h("td", {}, "")
    }
    const nodes: any[] = []
    backlinks.forEach((entry: string) => {
      if (!entry) return
      let raw = entry.trim()
      let alias: string | undefined
      if (raw.startsWith("!")) {
        raw = raw.slice(1)
      }
      if (raw.startsWith("[[") && raw.endsWith("]]")) {
        const inner = raw.slice(2, -2)
        const parsed = splitTargetAndAlias(inner)
        raw = parsed.target
        alias = parsed.alias
      }
      const { slug: targetSlug, anchor } = normalizeTargetSlug(raw, currentSlug)
      if (nodes.length > 0) {
        nodes.push(", ")
      }
      nodes.push(
        renderInternalLinkNode(
          targetSlug,
          currentSlug,
          allFiles,
          alias,
          anchor && anchor.length > 0 ? anchor : undefined,
        ),
      )
    })
    return h("td", {}, nodes)
  }

  const canEvalExpr = Boolean(getPropertyExpr(column))

  if (!canEvalExpr && column.startsWith("note.")) {
    const actualColumn = column.replace("note.", "")
    return buildTableCell(file, actualColumn, currentSlug, allFiles, getContext, getPropertyExpr)
  }

  const value = resolveValueWithFormulas(file, column, getContext, getPropertyExpr)

  if (value === undefined || value === null) {
    return h("td", {}, "")
  }

  if (Array.isArray(value)) {
    const parts: any[] = []
    value.forEach((item, idx) => {
      if (typeof item === "string") {
        parts.push(...renderInlineString(item, currentSlug, allFiles))
      } else {
        parts.push(String(item))
      }
      if (idx < value.length - 1) {
        parts.push(", ")
      }
    })
    return h("td", {}, parts)
  }

  if (value instanceof Date) {
    return h("td", {}, value.toISOString().split("T")[0])
  }

  if (typeof value === "string") {
    const rendered = renderInlineString(value, currentSlug, allFiles)
    return h("td", {}, rendered)
  }

  if (typeof value === "boolean") {
    return h("td", {}, [renderBooleanCheckbox(value)])
  }

  return h("td", {}, String(value))
}

function applySorting(
  files: QuartzPluginData[],
  sortConfig: { property: string; direction: "ASC" | "DESC" }[] = [],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): QuartzPluginData[] {
  if (sortConfig.length === 0) return files

  const normalizeSortValue = (val: any) => {
    if (val instanceof Date) {
      return val.getTime()
    }
    if (Array.isArray(val)) {
      return val.join(", ")
    }
    return val
  }

  return [...files].sort((a, b) => {
    for (const { property, direction } of sortConfig) {
      const aRaw = resolveValueWithFormulas(a, property, getContext, getPropertyExpr)
      const bRaw = resolveValueWithFormulas(b, property, getContext, getPropertyExpr)

      const aVal = normalizeSortValue(aRaw)
      const bVal = normalizeSortValue(bRaw)

      let comparison = 0
      const aMissing = aVal === undefined || aVal === null || aVal === ""
      const bMissing = bVal === undefined || bVal === null || bVal === ""

      if (aMissing && bMissing) {
        comparison = 0
      } else if (aMissing) {
        comparison = 1
      } else if (bMissing) {
        comparison = -1
      } else if (typeof aVal === "string" && typeof bVal === "string") {
        comparison = aVal.localeCompare(bVal)
      } else {
        comparison = aVal > bVal ? 1 : aVal < bVal ? -1 : 0
      }

      if (comparison !== 0) {
        return direction === "ASC" ? comparison : -comparison
      }
    }
    return 0
  })
}

function groupFiles(
  files: QuartzPluginData[],
  groupBy: string | BaseGroupBy,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): Map<string, QuartzPluginData[]> {
  const groups = new Map<string, QuartzPluginData[]>()

  const property = typeof groupBy === "string" ? groupBy : groupBy.property
  const direction = typeof groupBy === "string" ? "ASC" : groupBy.direction

  for (const file of files) {
    const value = resolveValueWithFormulas(file, property, getContext, getPropertyExpr)
    const key = value === undefined || value === null ? "(empty)" : String(value)

    if (!groups.has(key)) {
      groups.set(key, [])
    }
    groups.get(key)!.push(file)
  }

  const sortedGroups = new Map(
    [...groups.entries()].sort(([a], [b]) => {
      if (direction === "ASC") {
        return a.localeCompare(b)
      } else {
        return b.localeCompare(a)
      }
    }),
  )

  return sortedGroups
}

function buildTableSummaryRow(
  columns: string[],
  files: QuartzPluginData[],
  summaryConfig: ViewSummaryConfig | undefined,
  allFiles: QuartzPluginData[],
  summaryExpressions: Record<string, ProgramIR> | undefined,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): any | undefined {
  if (!summaryConfig?.columns || Object.keys(summaryConfig.columns).length === 0) {
    return undefined
  }

  const summaryValues = computeViewSummaries(
    columns,
    files,
    summaryConfig,
    allFiles,
    getContext,
    (file, column) => resolveValueWithFormulas(file, column, getContext, getPropertyExpr),
    summaryExpressions,
  )

  const hasValues = Object.values(summaryValues).some((v) => v !== undefined)
  if (!hasValues) {
    return undefined
  }

  const cells = columns.map((col) => {
    const value = summaryValues[col]
    if (value === undefined) {
      return h("td.base-summary-cell", {}, "")
    }
    return h("td.base-summary-cell", {}, String(value))
  })

  return h("tfoot", [h("tr.base-summary-row", cells)])
}

function buildTable(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, { displayName?: string }>,
  topLevelSummaries?: Record<string, string>,
  summaryExpressions?: Record<string, ProgramIR>,
): any {
  const columns = view.order || []

  const summaryConfig = parseViewSummaries(view.summaries, topLevelSummaries)

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr)
    const allRows: any[] = []

    for (const [groupName, groupFiles] of groups) {
      const groupHeader = h("tr.base-group-header", [
        h("td", { colspan: columns.length }, groupName),
      ])
      allRows.push(groupHeader)

      for (const file of groupFiles) {
        const cells = columns.map((col) =>
          buildTableCell(file, col, currentSlug, allFiles, getContext, getPropertyExpr),
        )
        allRows.push(h("tr", cells))
      }
    }

    const tbody = h("tbody", allRows)
    const thead = h("thead", buildTableHead(columns, properties))
    return h("table.base-table", [thead, tbody])
  }

  const rows = files.map((f) => {
    const cells = columns.map((col) =>
      buildTableCell(f, col, currentSlug, allFiles, getContext, getPropertyExpr),
    )
    return h("tr", cells)
  })

  const tbody = h("tbody", rows)
  const thead = h("thead", buildTableHead(columns, properties))
  const tfoot = buildTableSummaryRow(
    columns,
    files,
    summaryConfig,
    allFiles,
    summaryExpressions,
    getContext,
    getPropertyExpr,
  )
  const tableChildren = tfoot ? [thead, tbody, tfoot] : [thead, tbody]
  return h("table.base-table", tableChildren)
}

function listValueToPlainText(value: any): string | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => listValueToPlainText(item))
      .filter((part): part is string => Boolean(part && part.length > 0))
    if (parts.length === 0) return undefined
    return parts.join(", ")
  }
  if (value instanceof Date) {
    return value.toISOString().split("T")[0]
  }
  if (typeof value === "string") {
    const cleaned = value
      .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2")
      .replace(/\[\[([^\]]+)\]\]/g, "$1")
      .trim()
    return cleaned.length > 0 ? cleaned : undefined
  }
  const stringified = String(value).trim()
  return stringified.length > 0 ? stringified : undefined
}

function hasRenderableValue(value: any): boolean {
  if (value === undefined || value === null) return false
  if (Array.isArray(value)) {
    return value.some((item) => hasRenderableValue(item))
  }
  if (value instanceof Date) return true
  if (typeof value === "string") return value.trim().length > 0
  return true
}

function renderPropertyValueNodes(
  value: any,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
): any[] {
  if (value === undefined || value === null) return []
  if (Array.isArray(value)) {
    const nodes: any[] = []
    value.forEach((item, idx) => {
      nodes.push(...renderPropertyValueNodes(item, currentSlug, allFiles))
      if (idx < value.length - 1) {
        nodes.push(", ")
      }
    })
    return nodes
  }
  if (value instanceof Date) {
    return [value.toISOString().split("T")[0]]
  }
  if (typeof value === "string") {
    return renderInlineString(value, currentSlug, allFiles)
  }
  return [String(value)]
}

function buildList(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): any {
  const nestedProperties = view.nestedProperties === true || view.indentProperties === true
  const order = Array.isArray(view.order) && view.order.length > 0 ? view.order : ["title"]
  const [primaryProp, ...secondaryProps] = order

  const renderListItem = (file: QuartzPluginData) => {
    const slug = (file.slug || "") as FullSlug
    const href = resolveRelative(currentSlug, slug)

    const fallbackTitle =
      getFileDisplayName(file) ?? fallbackNameFromSlug((file.slug || "") as FullSlug)

    const primaryValue = primaryProp
      ? resolveValueWithFormulas(file, primaryProp, getContext, getPropertyExpr)
      : resolveValueWithFormulas(file, "title", getContext, getPropertyExpr)
    const primaryText = listValueToPlainText(primaryValue) ?? fallbackTitle
    const anchor = h("a.internal", { href, "data-slug": slug }, primaryText)

    const seen = new Set<string>()
    if (primaryProp) {
      seen.add(primaryProp)
    }

    if (!nestedProperties) {
      const rawSeparator = typeof view.separator === "string" ? view.separator : ","
      const separator = rawSeparator.endsWith(" ") ? rawSeparator : `${rawSeparator} `
      const inlineNodes: any[] = []

      for (const propertyKey of secondaryProps) {
        if (!propertyKey || seen.has(propertyKey)) continue
        const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
        if (!hasRenderableValue(value)) continue

        const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
        if (renderedValue.length === 0) continue

        inlineNodes.push(separator)
        inlineNodes.push(...renderedValue)
        seen.add(propertyKey)
      }

      return inlineNodes.length > 0 ? h("li", [anchor, ...inlineNodes]) : h("li", [anchor])
    }

    const metadataItems: any[] = []

    for (const propertyKey of secondaryProps) {
      if (!propertyKey || seen.has(propertyKey)) continue
      const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
      if (!hasRenderableValue(value)) continue

      const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (renderedValue.length === 0) continue

      const label = getPropertyDisplayName(propertyKey, properties)
      metadataItems.push(h("li", [h("span.base-list-meta-label", `${label}: `), ...renderedValue]))
      seen.add(propertyKey)
    }

    if (metadataItems.length === 0) {
      return h("li", [anchor])
    }

    return h("li", [anchor, h("ul.base-list-nested", metadataItems)])
  }

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr)
    const groupElements: any[] = []

    for (const [groupName, groupedFiles] of groups) {
      const items = groupedFiles.map((file) => renderListItem(file))
      groupElements.push(
        h("div.base-list-group", [
          h("h3.base-list-group-header", groupName),
          h("ul.base-list", items),
        ]),
      )
    }

    return h("div.base-list-container", groupElements)
  }

  const items = files.map((file) => renderListItem(file))
  return h("ul.base-list", items)
}

function normalizeCalendarDate(value: any): string | undefined {
  if (value instanceof Date) {
    return value.toISOString().split("T")[0]
  }
  if (typeof value === "string") {
    const match = value.match(/^\d{4}-\d{2}-\d{2}/)
    if (match) return match[0]
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().split("T")[0]
    }
    return undefined
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().split("T")[0]
    }
  }
  return undefined
}

function buildCalendar(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): any {
  const nestedProperties = view.nestedProperties === true || view.indentProperties === true
  const order = Array.isArray(view.order) && view.order.length > 0 ? view.order : ["title"]
  const [primaryProp, ...secondaryProps] = order
  const dateField =
    typeof view.date === "string"
      ? view.date
      : typeof view.dateField === "string"
        ? view.dateField
        : typeof view.dateProperty === "string"
          ? view.dateProperty
          : "date"

  const renderListItem = (file: QuartzPluginData) => {
    const slug = (file.slug || "") as FullSlug
    const href = resolveRelative(currentSlug, slug)

    const fallbackTitle =
      getFileDisplayName(file) ?? fallbackNameFromSlug((file.slug || "") as FullSlug)

    const primaryValue = primaryProp
      ? resolveValueWithFormulas(file, primaryProp, getContext, getPropertyExpr)
      : resolveValueWithFormulas(file, "title", getContext, getPropertyExpr)
    const primaryText = listValueToPlainText(primaryValue) ?? fallbackTitle
    const anchor = h("a.internal", { href, "data-slug": slug }, primaryText)

    const seen = new Set<string>()
    if (primaryProp) {
      seen.add(primaryProp)
    }

    if (!nestedProperties) {
      const rawSeparator = typeof view.separator === "string" ? view.separator : ","
      const separator = rawSeparator.endsWith(" ") ? rawSeparator : `${rawSeparator} `
      const inlineNodes: any[] = []

      for (const propertyKey of secondaryProps) {
        if (!propertyKey || seen.has(propertyKey)) continue
        const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
        if (!hasRenderableValue(value)) continue

        const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
        if (renderedValue.length === 0) continue

        inlineNodes.push(separator)
        inlineNodes.push(...renderedValue)
        seen.add(propertyKey)
      }

      return inlineNodes.length > 0 ? h("li", [anchor, ...inlineNodes]) : h("li", [anchor])
    }

    const metadataItems: any[] = []

    for (const propertyKey of secondaryProps) {
      if (!propertyKey || seen.has(propertyKey)) continue
      const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
      if (!hasRenderableValue(value)) continue

      const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (renderedValue.length === 0) continue

      const label = getPropertyDisplayName(propertyKey, properties)
      metadataItems.push(h("li", [h("span.base-list-meta-label", `${label}: `), ...renderedValue]))
      seen.add(propertyKey)
    }

    if (metadataItems.length === 0) {
      return h("li", [anchor])
    }

    return h("li", [anchor, h("ul.base-list-nested", metadataItems)])
  }

  const groups = new Map<string, QuartzPluginData[]>()
  for (const file of files) {
    const dateValue = resolveValueWithFormulas(file, dateField, getContext, getPropertyExpr)
    const dateKey = normalizeCalendarDate(dateValue) ?? "(no date)"
    if (!groups.has(dateKey)) {
      groups.set(dateKey, [])
    }
    groups.get(dateKey)!.push(file)
  }

  const groupElements: any[] = []
  const sorted = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
  for (const [dateKey, groupedFiles] of sorted) {
    const items = groupedFiles.map((file) => renderListItem(file))
    groupElements.push(
      h("div.base-calendar-group", [
        h("h3.base-calendar-group-header", dateKey),
        h("ul.base-list", items),
      ]),
    )
  }

  return h("div.base-calendar-container", groupElements)
}

function buildCards(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): any {
  const imageField = view.image || "image"

  const renderCard = (file: QuartzPluginData) => {
    const title = file.frontmatter?.title || file.slug?.split("/").pop() || ""
    const slug = (file.slug || "") as FullSlug
    const href = resolveRelative(currentSlug, slug)

    let imageUrl: string | undefined
    const imageValue = resolveValueWithFormulas(file, imageField, getContext, getPropertyExpr)
    const toRelativeFromSlug = (target: string): string => {
      if (isAbsoluteURL(target)) return target
      const imgSlug = slugifyFilePath(target as FilePath)
      return resolveRelative(currentSlug, imgSlug)
    }

    if (imageValue) {
      if (typeof imageValue === "string") {
        const wl = imageValue.match(/^\[\[(.+?)\]\]$/)
        if (wl) {
          const inner = wl[1]
          const { target } = splitTargetAndAlias(inner)
          const { slug } = normalizeTargetSlug(target, currentSlug)
          imageUrl = resolveRelative(currentSlug, slug)
        } else {
          imageUrl = toRelativeFromSlug(imageValue)
        }
      } else if (Array.isArray(imageValue) && imageValue.length > 0) {
        const first = imageValue[0]
        if (typeof first === "string") {
          const wl = first.match(/^\[\[(.+?)\]\]$/)
          if (wl) {
            const inner = wl[1]
            const { target } = splitTargetAndAlias(inner)
            const { slug } = normalizeTargetSlug(target, currentSlug)
            imageUrl = resolveRelative(currentSlug, slug)
          } else {
            imageUrl = toRelativeFromSlug(first)
          }
        }
      }
    }

    const metadataItems: any[] = []
    const order = view.order || []
    const metadataFields = order.filter(
      (field) => field !== "title" && field !== imageField && field !== "file.title",
    )

    for (const field of metadataFields) {
      const value = resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
      if (value !== undefined && value !== null && value !== "") {
        const label = field
          .replace("file.", "")
          .replace("note.", "")
          .replace(/-/g, " ")
          .replace(/([A-Z])/g, " $1")
          .trim()

        let displayValue: any
        if (Array.isArray(value)) {
          const parts: any[] = []
          value.forEach((item, idx) => {
            if (typeof item === "string") {
              parts.push(...renderInlineString(item, currentSlug, allFiles))
            } else {
              parts.push(String(item))
            }
            if (idx < value.length - 1) {
              parts.push(", ")
            }
          })
          displayValue = parts
        } else if (value instanceof Date) {
          displayValue = value.toISOString().split("T")[0]
        } else if (typeof value === "string") {
          const rendered = renderInlineString(value, currentSlug, allFiles)
          displayValue = rendered
        } else {
          displayValue = String(value)
        }

        metadataItems.push(
          h("div.base-card-meta-item", [
            h("span.base-card-meta-label", label),
            h("span.base-card-meta-value", displayValue),
          ]),
        )
      }
    }

    const cardChildren = []
    if (imageUrl) {
      cardChildren.push(
        h(
          "a.base-card-image-link",
          {
            href,
            "data-slug": slug,
            style: {
              "background-image": `url(${imageUrl})`,
              "background-size": "cover",
              top: "0px",
              "inset-inline": "0px",
            },
          },
          [],
        ),
      )
    }

    const contentChildren = [
      h("a.base-card-title-link", { href, "data-slug": slug }, [h("h3.base-card-title", title)]),
    ]
    if (metadataItems.length > 0) {
      contentChildren.push(h("div.base-card-meta", metadataItems))
    }

    cardChildren.push(h("div.base-card-content", contentChildren))

    return h("div.base-card", cardChildren)
  }

  const styleParts: string[] = []
  if (typeof view.cardSize === "number" && view.cardSize > 0) {
    styleParts.push(`--base-card-min: ${view.cardSize}px;`)
  }
  if (typeof view.cardAspect === "number" && view.cardAspect > 0) {
    styleParts.push(`--base-card-aspect: ${view.cardAspect};`)
  }
  const varStyle = styleParts.length > 0 ? styleParts.join(" ") : undefined

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr)
    const groupElements: any[] = []

    const groupSizes = view.groupSizes as Record<string, number> | undefined
    const groupAspects = view.groupAspects as Record<string, number> | undefined

    for (const [groupName, groupFiles] of groups) {
      const cards = groupFiles.map((file) => renderCard(file))
      let gridStyle: string | undefined
      const parts: string[] = []
      const size = groupSizes?.[groupName]
      if (typeof size === "number" && size > 0) {
        parts.push(`--base-card-min: ${size}px;`)
      }
      const aspect = groupAspects?.[groupName]
      if (typeof aspect === "number" && aspect > 0) {
        parts.push(`--base-card-aspect: ${aspect};`)
      }
      gridStyle = parts.length > 0 ? parts.join(" ") : undefined

      groupElements.push(
        h("div.base-card-group", [
          h("h3.base-card-group-header", groupName),
          h("div.base-card-grid", gridStyle ? { style: gridStyle } : {}, cards),
        ]),
      )
    }

    return h("div.base-card-container", varStyle ? { style: varStyle } : {}, groupElements)
  }

  const cards = files.map((file) => renderCard(file))
  return h("div.base-card-grid", varStyle ? { style: varStyle } : {}, cards)
}

function buildMap(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): any {
  const resolveMapProperty = (file: QuartzPluginData, prop: string | undefined): unknown => {
    if (!prop) return undefined
    const key = prop.trim()
    if (!key) return undefined
    if (getPropertyExpr(key)) {
      return resolveValueWithFormulas(file, key, getContext, getPropertyExpr)
    }
    if (key.startsWith("note.")) {
      const stripped = key.replace(/^note\./, "")
      if (getPropertyExpr(stripped)) {
        return resolveValueWithFormulas(file, stripped, getContext, getPropertyExpr)
      }
    } else {
      const prefixed = `note.${key}`
      if (getPropertyExpr(prefixed)) {
        return resolveValueWithFormulas(file, prefixed, getContext, getPropertyExpr)
      }
    }
    return undefined
  }

  const coordinatesProp = view.coordinates || "coordinates"

  const markers: any[] = []

  for (const file of files) {
    const coordsValue = resolveMapProperty(file, coordinatesProp)
    if (!coordsValue || !Array.isArray(coordsValue) || coordsValue.length < 2) {
      continue
    }

    const lat = parseFloat(String(coordsValue[0]))
    const lon = parseFloat(String(coordsValue[1]))

    if (isNaN(lat) || isNaN(lon)) {
      continue
    }

    const title = getFileDisplayName(file) ?? fallbackNameFromSlug((file.slug || "") as FullSlug)
    const slug = (file.slug || "") as FullSlug

    const popupFields: Record<string, any> = {}
    const order = view.order || []
    for (const field of order) {
      if (field === "title" || field === "file.title" || field === "note.title") continue
      const value = resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
      if (value !== undefined && value !== null && value !== "") {
        popupFields[field] = value
      }
    }

    const icon = resolveMapProperty(file, view.markerIcon)
    const color = resolveMapProperty(file, view.markerColor)

    markers.push({
      lat,
      lon,
      title,
      slug,
      icon: icon ? String(icon) : undefined,
      color: color ? String(color) : undefined,
      popupFields,
    })
  }

  const config = {
    defaultZoom: view.defaultZoom ?? 12,
    defaultCenter: view.defaultCenter,
    clustering: view.clustering !== false,
  }

  const attrs: Record<string, any> = {
    "data-markers": JSON.stringify(markers),
    "data-config": JSON.stringify(config),
    "data-current-slug": currentSlug,
  }

  if (properties) {
    attrs["data-properties"] = JSON.stringify(properties)
  }

  return h("div.base-map", attrs)
}

function resolveViewSlug(baseSlug: FullSlug, viewName: string, viewIndex: number): FullSlug {
  if (viewIndex === 0) return baseSlug
  const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
  return joinSegments(baseSlug, slugifiedName) as FullSlug
}

function renderDiagnostics(
  diagnostics: BaseExpressionDiagnostic[] | undefined,
  currentSlug: FullSlug,
): any | undefined {
  if (!diagnostics || diagnostics.length === 0) {
    return undefined
  }
  const items = diagnostics.map((diag, index) => {
    const line = diag.span.start.line
    const column = diag.span.start.column
    const location = Number.isFinite(line) && Number.isFinite(column) ? `${line}:${column}` : ""
    const label = location.length > 0 ? `${diag.context} (${location})` : diag.context
    return h("li.base-diagnostics-item", { key: String(index) }, [
      h("div.base-diagnostics-label", label),
      h("div.base-diagnostics-message", diag.message),
      h("code.base-diagnostics-source", diag.source),
    ])
  })
  return h("div.base-diagnostics", [
    h("div.base-diagnostics-title", "bases diagnostics"),
    h("div.base-diagnostics-meta", [
      h("span", "page"),
      h("span.base-diagnostics-page", currentSlug),
    ]),
    h("ul.base-diagnostics-list", items),
  ])
}

export type BaseViewMeta = { name: string; type: BaseView["type"]; slug: FullSlug }

export type RenderedBaseView = { view: BaseView; slug: FullSlug; tree: Root }

export type BaseMetadata = { baseSlug: FullSlug; currentView: string; allViews: BaseViewMeta[] }

export function renderBaseViewsForFile(
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  thisFile?: QuartzPluginData,
): { views: RenderedBaseView[]; allViews: BaseViewMeta[] } {
  const config = baseFileData.basesConfig as BasesConfigFile | undefined
  if (!config || !baseFileData.slug) {
    return { views: [], allViews: [] }
  }

  const baseSlug = baseFileData.slug as FullSlug
  const expressions = baseFileData.basesExpressions
  const formulaExpressions = expressions?.formulas
  const summaryExpressionsByView = expressions?.viewSummaries
  const viewFilterExpressions = expressions?.viewFilters
  const formulaCaches = new Map<string, Map<string, Value>>()
  const propertyCaches = new Map<string, Map<string, Value>>()
  const runtimeDiagnostics: BaseExpressionDiagnostic[] = []
  const runtimeDiagnosticSet = new Set<string>()
  const propertyExpressions = expressions?.propertyExpressions ?? {}

  const fileIndex = new Map<string, QuartzPluginData>()
  for (const entry of allFiles) {
    if (!entry.slug) continue
    fileIndex.set(simplifySlug(entry.slug as FullSlug), entry)
  }

  const backlinkSets = new Map<string, Set<string>>()
  for (const entry of allFiles) {
    if (!entry.slug) continue
    const sourceSlug = simplifySlug(entry.slug as FullSlug)
    const links = Array.isArray(entry.links) ? entry.links.map((link) => String(link)) : []
    for (const link of links) {
      if (!link) continue
      let set = backlinkSets.get(link)
      if (!set) {
        set = new Set()
        backlinkSets.set(link, set)
      }
      set.add(sourceSlug)
    }
  }
  const backlinksIndex = new Map<string, string[]>()
  for (const [key, set] of backlinkSets) {
    backlinksIndex.set(key, [...set])
  }

  const getPropertyExpr: PropertyExprGetter = (property) => {
    const key = property.trim()
    if (!key) return null
    return propertyExpressions[key] ?? null
  }

  const baseThisFile = thisFile ?? baseFileData

  const getEvalContext: EvalContextFactory = (file) => {
    const slug = file.slug ? String(file.slug) : file.filePath ? String(file.filePath) : ""
    let cache = formulaCaches.get(slug)
    if (!cache) {
      cache = new Map()
      formulaCaches.set(slug, cache)
    }
    let propertyCache = propertyCaches.get(slug)
    if (!propertyCache) {
      propertyCache = new Map()
      propertyCaches.set(slug, propertyCache)
    }
    return {
      file,
      thisFile: baseThisFile,
      allFiles,
      fileIndex,
      backlinksIndex,
      formulas: formulaExpressions,
      formulaSources: config.formulas,
      formulaCache: cache,
      formulaStack: new Set(),
      propertyCache,
      diagnostics: runtimeDiagnostics,
      diagnosticSet: runtimeDiagnosticSet,
    }
  }

  const allViews = config.views.map((view, idx) => ({
    name: view.name,
    type: view.type,
    slug: resolveViewSlug(baseSlug, view.name, idx),
  }))

  const baseFilterSource = typeof config.filters === "string" ? config.filters : ""
  const baseMatchedFiles = expressions?.filters
    ? allFiles.filter((file) => {
        const ctx = getEvalContext(file)
        ctx.diagnosticContext = "filters"
        ctx.diagnosticSource = baseFilterSource
        return evaluateFilterExpression(expressions.filters!, ctx)
      })
    : allFiles

  const views: RenderedBaseView[] = []

  for (const [viewIndex, view] of config.views.entries()) {
    const slug = resolveViewSlug(baseSlug, view.name, viewIndex)

    let matchedFiles = baseMatchedFiles
    const viewFilter = viewFilterExpressions ? viewFilterExpressions[String(viewIndex)] : undefined
    if (viewFilter) {
      const viewFilterSource = typeof view.filters === "string" ? view.filters : ""
      matchedFiles = matchedFiles.filter((file) => {
        const ctx = getEvalContext(file)
        ctx.diagnosticContext = `views[${viewIndex}].filters`
        ctx.diagnosticSource = viewFilterSource
        return evaluateFilterExpression(viewFilter, ctx)
      })
    }

    const sortedFiles = applySorting(matchedFiles, view.sort, getEvalContext, getPropertyExpr)
    const limitedFiles = view.limit ? sortedFiles.slice(0, view.limit) : sortedFiles

    const diagnostics = [...(baseFileData.basesDiagnostics ?? []), ...runtimeDiagnostics]
    const diagnosticsNode = renderDiagnostics(
      diagnostics.length > 0 ? diagnostics : undefined,
      slug,
    )
    const wrapView = (node: any) =>
      h("div.base-view", { "data-base-view-type": view.type, "data-base-view-name": view.name }, [
        node,
      ])

    let tree: Root
    if (view.type === "table") {
      const tableNode = wrapView(
        buildTable(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
          config.summaries,
          summaryExpressionsByView ? summaryExpressionsByView[String(viewIndex)] : undefined,
        ),
      )
      tree = {
        type: "root",
        children: diagnosticsNode ? [diagnosticsNode, tableNode] : [tableNode],
      }
    } else if (view.type === "list") {
      const listNode = wrapView(
        buildList(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        ),
      )
      tree = { type: "root", children: diagnosticsNode ? [diagnosticsNode, listNode] : [listNode] }
    } else if (view.type === "card" || view.type === "cards" || view.type === "gallery") {
      const cardsNode = wrapView(
        buildCards(limitedFiles, view, slug, allFiles, getEvalContext, getPropertyExpr),
      )
      tree = {
        type: "root",
        children: diagnosticsNode ? [diagnosticsNode, cardsNode] : [cardsNode],
      }
    } else if (view.type === "board") {
      const boardNode = wrapView(
        buildCards(limitedFiles, view, slug, allFiles, getEvalContext, getPropertyExpr),
      )
      tree = {
        type: "root",
        children: diagnosticsNode ? [diagnosticsNode, boardNode] : [boardNode],
      }
    } else if (view.type === "calendar") {
      const calendarNode = wrapView(
        buildCalendar(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        ),
      )
      tree = {
        type: "root",
        children: diagnosticsNode ? [diagnosticsNode, calendarNode] : [calendarNode],
      }
    } else if (view.type === "map") {
      const mapNode = wrapView(
        buildMap(limitedFiles, view, slug, getEvalContext, getPropertyExpr, config.properties),
      )
      tree = { type: "root", children: diagnosticsNode ? [diagnosticsNode, mapNode] : [mapNode] }
    } else {
      continue
    }

    views.push({ view, slug, tree })
  }

  return { views, allViews }
}
