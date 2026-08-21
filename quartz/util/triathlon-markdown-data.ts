import { isRecord } from './type-guards'

export type MarkdownTableAlignment = 'left' | 'center' | 'right'

export type MarkdownTableRow = Readonly<Record<string, unknown>>

export interface GfmTableOptions {
  alignments?: Readonly<Record<string, MarkdownTableAlignment>>
  plainTextColumns?: readonly string[]
}

export interface TitledSectionsOptions {
  title?: string
  headingDepth?: number
}

const isContainer = (value: unknown): boolean => Array.isArray(value) || isRecord(value)

const escapeMarkdownText = (value: string): string =>
  value
    .replaceAll('\\', '\\\\')
    .replaceAll('&', '&amp;')
    .replace(/([!"#()*<>[\]_`|~])/g, '\\$1')
    .replace(/\r\n|\r|\n/g, '<br>')

export const escapeMarkdownHeading = (value: string): string =>
  escapeMarkdownText(value.replace(/\r\n|\r|\n/g, ' '))

const scalarText = (value: unknown): string => {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'string') return `"${escapeMarkdownText(value)}"`
  if (typeof value === 'bigint') return `${value}n`
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  const serialized = JSON.stringify(value)
  return serialized ?? String(value)
}

const alignmentMarker = (alignment: MarkdownTableAlignment | undefined): string => {
  switch (alignment) {
    case 'left':
      return ':---'
    case 'center':
      return ':---:'
    case 'right':
      return '---:'
    default:
      return '---'
  }
}

export function renderGfmTable(
  rows: readonly MarkdownTableRow[],
  options: GfmTableOptions = {},
): string {
  if (rows.length === 0) return ''

  const columns: string[] = []
  const knownColumns = new Set<string>()
  for (const row of rows) {
    for (const column of Object.keys(row)) {
      if (knownColumns.has(column)) continue
      knownColumns.add(column)
      columns.push(column)
    }
  }
  if (columns.length === 0) return ''

  const header = `| ${columns.map(escapeMarkdownText).join(' | ')} |`
  const separator = `| ${columns.map(column => alignmentMarker(options.alignments?.[column])).join(' | ')} |`
  const plainTextColumns = new Set(options.plainTextColumns)
  const body = rows.map(
    row =>
      `| ${columns
        .map(column => {
          const value = row[column]
          return plainTextColumns.has(column) && typeof value === 'string'
            ? escapeMarkdownText(value)
            : scalarText(value)
        })
        .join(' | ')} |`,
  )
  return [header, separator, ...body].join('\n')
}

const emptyValue = (value: 'array' | 'object'): MarkdownTableRow => ({
  field: 'empty',
  value: value === 'array' ? [] : {},
})

const scalarRecord = (record: MarkdownTableRow): boolean =>
  Object.keys(record).length > 0 && Object.values(record).every(value => !isContainer(value))

const tableSection = (rows: readonly MarkdownTableRow[], indexColumn?: string): string =>
  renderGfmTable(rows, {
    alignments: indexColumn ? { [indexColumn]: 'right' } : undefined,
    plainTextColumns: ['field'],
  })

const heading = (path: string, depth: number): string =>
  `${'#'.repeat(Math.min(6, Math.max(1, depth)))} ${escapeMarkdownHeading(path)}`

const childPath = (path: string, key: string): string => `${path}.${key}`

const indexPath = (path: string, index: number): string => `${path}[${index}]`

const section = (path: string, depth: number, body: string): string =>
  `${heading(path, depth)}\n\n${body}`

const scalarObjectRows = (record: MarkdownTableRow): MarkdownTableRow[] =>
  Object.entries(record)
    .filter(([, value]) => !isContainer(value))
    .map(([field, value]) => ({ field, value }))

const renderValue = (value: unknown, path: string, depth: number): string[] => {
  if (Array.isArray(value)) return renderArray(value, path, depth)
  if (isRecord(value)) return renderObject(value, path, depth)
  return [section(path, depth, tableSection([{ value }]))]
}

const renderObject = (
  record: Readonly<Record<string, unknown>>,
  path: string,
  depth: number,
): string[] => {
  const scalarRows = scalarObjectRows(record)
  const nested = Object.entries(record).filter(([, value]) => isContainer(value))
  const blocks: string[] = []
  if (scalarRows.length > 0) blocks.push(section(path, depth, tableSection(scalarRows)))
  if (scalarRows.length === 0 && nested.length === 0) {
    blocks.push(section(path, depth, tableSection([emptyValue('object')])))
  }
  if (scalarRows.length === 0 && nested.length > 0) blocks.push(heading(path, depth))
  for (const [key, child] of nested)
    blocks.push(...renderValue(child, childPath(path, key), depth + 1))
  return blocks
}

const renderArray = (values: readonly unknown[], path: string, depth: number): string[] => {
  if (values.length === 0) return [section(path, depth, tableSection([emptyValue('array')]))]

  if (values.every(value => isRecord(value) && scalarRecord(value))) {
    const sourceColumns = new Set(
      values.flatMap(value => (isRecord(value) ? Object.keys(value) : [])),
    )
    let indexColumn = 'arrayIndex'
    while (sourceColumns.has(indexColumn)) indexColumn = `_${indexColumn}`
    const rows: MarkdownTableRow[] = []
    for (let index = 0; index < values.length; index += 1) {
      const value = values[index]
      if (!isRecord(value)) continue
      rows.push({ [indexColumn]: index + 1, ...value })
    }
    return [section(path, depth, tableSection(rows, indexColumn))]
  }

  if (values.every(value => !isContainer(value))) {
    const indexColumn = 'arrayIndex'
    const rows = values.map((value, index) => ({ [indexColumn]: index + 1, value }))
    return [section(path, depth, tableSection(rows, indexColumn))]
  }

  return [
    heading(path, depth),
    ...values.flatMap((value, index) => renderValue(value, indexPath(path, index), depth + 1)),
  ]
}

export function renderTitledSections(value: unknown, options: TitledSectionsOptions = {}): string {
  const path = options.title ?? 'data'
  const depth = options.headingDepth ?? 2
  return renderValue(value, path, depth).join('\n\n')
}
