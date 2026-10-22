import { QuartzPluginData } from '../../../../plugins/vfile'
import { BaseExpressionDiagnostic } from '../diagnostics'
import { ProgramIR } from '../ir'

export type NullValue = { kind: 'null' }
export type BooleanValue = { kind: 'boolean'; value: boolean }
export type NumberValue = { kind: 'number'; value: number }
export type StringValue = { kind: 'string'; value: string }
export type DateValue = { kind: 'date'; value: Date }
export type DurationValue = { kind: 'duration'; value: number; months: number }
export type ListValue = { kind: 'list'; value: Value[] }
export type ObjectValue = { kind: 'object'; value: Record<string, Value> }
export type FileValue = { kind: 'file'; value: QuartzPluginData }
export type LinkValue = { kind: 'link'; value: string; display?: string }
export type RegexValue = { kind: 'regex'; value: RegExp }
export type HtmlValue = { kind: 'html'; value: string }
export type IconValue = { kind: 'icon'; value: string }
export type ImageValue = { kind: 'image'; value: string }

export type Value =
  | NullValue
  | BooleanValue
  | NumberValue
  | StringValue
  | DateValue
  | DurationValue
  | ListValue
  | ObjectValue
  | FileValue
  | LinkValue
  | RegexValue
  | HtmlValue
  | IconValue
  | ImageValue

export type ValueKind = Value['kind']
export type ValueOf<K extends ValueKind> = Extract<Value, { kind: K }>

export function isValueKind(value: Value, kind: 'null'): value is NullValue
export function isValueKind(value: Value, kind: 'boolean'): value is BooleanValue
export function isValueKind(value: Value, kind: 'number'): value is NumberValue
export function isValueKind(value: Value, kind: 'string'): value is StringValue
export function isValueKind(value: Value, kind: 'date'): value is DateValue
export function isValueKind(value: Value, kind: 'duration'): value is DurationValue
export function isValueKind(value: Value, kind: 'list'): value is ListValue
export function isValueKind(value: Value, kind: 'object'): value is ObjectValue
export function isValueKind(value: Value, kind: 'file'): value is FileValue
export function isValueKind(value: Value, kind: 'link'): value is LinkValue
export function isValueKind(value: Value, kind: 'regex'): value is RegexValue
export function isValueKind(value: Value, kind: 'html'): value is HtmlValue
export function isValueKind(value: Value, kind: 'icon'): value is IconValue
export function isValueKind(value: Value, kind: 'image'): value is ImageValue
export function isValueKind(value: Value, kind: ValueKind): value is Value {
  return value.kind === kind
}

export type EvalContext = {
  file: QuartzPluginData
  thisFile?: QuartzPluginData
  allFiles: QuartzPluginData[]
  rows?: QuartzPluginData[]
  fileIndex?: Map<string, QuartzPluginData>
  backlinksIndex?: Map<string, string[]>
  formulas?: Record<string, ProgramIR>
  formulaSources?: Record<string, string>
  formulaCache?: Map<string, Value>
  formulaStack?: Set<string>
  locals?: Record<string, Value>
  values?: Value[]
  diagnostics?: BaseExpressionDiagnostic[]
  diagnosticContext?: string
  diagnosticSource?: string
  diagnosticSet?: Set<string>
  propertyCache?: Map<string, Value>
}

const nullValue: NullValue = { kind: 'null' }
export const trueValue: BooleanValue = { kind: 'boolean', value: true }
const falseValue: BooleanValue = { kind: 'boolean', value: false }
export const emptyValueList: Value[] = []

export const makeNull = (): NullValue => nullValue
export const makeBoolean = (value: boolean): BooleanValue => (value ? trueValue : falseValue)
export const makeNumber = (value: number): NumberValue => ({ kind: 'number', value })
export const makeString = (value: string): StringValue => ({ kind: 'string', value })
export const makeDate = (value: Date): DateValue => ({ kind: 'date', value })
export const makeDuration = (value: number, months = 0): DurationValue => ({
  kind: 'duration',
  value,
  months,
})
export const makeList = (value: Value[]): ListValue => ({ kind: 'list', value })
export const makeObject = (value: Record<string, Value>): ObjectValue => ({ kind: 'object', value })
export const makeFile = (value: QuartzPluginData): FileValue => ({ kind: 'file', value })
export const makeLink = (value: string, display?: string): LinkValue => ({
  kind: 'link',
  value,
  display,
})
export const makeRegex = (value: RegExp): RegexValue => ({ kind: 'regex', value })
export const makeHtml = (value: string): HtmlValue => ({ kind: 'html', value })
export const makeIcon = (value: string): IconValue => ({ kind: 'icon', value })
export const makeImage = (value: string): ImageValue => ({ kind: 'image', value })

export const isValue = (value: unknown): value is Value =>
  typeof value === 'object' && value !== null && 'kind' in value

export const isNumberValue = (value: Value): value is NumberValue => isValueKind(value, 'number')

export const isStringValue = (value: Value): value is StringValue => isValueKind(value, 'string')

export const isBooleanValue = (value: Value): value is BooleanValue => isValueKind(value, 'boolean')

export const isListValue = (value: Value): value is ListValue => isValueKind(value, 'list')

export const isObjectValue = (value: Value): value is ObjectValue => isValueKind(value, 'object')

export const isDateValue = (value: Value): value is DateValue => isValueKind(value, 'date')

export const isDurationValue = (value: Value): value is DurationValue =>
  isValueKind(value, 'duration')

export const isFileValue = (value: Value): value is FileValue => isValueKind(value, 'file')

export const isLinkValue = (value: Value): value is LinkValue => isValueKind(value, 'link')

export const isRegexValue = (value: Value): value is RegexValue => isValueKind(value, 'regex')
