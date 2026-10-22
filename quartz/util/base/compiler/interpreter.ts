import { Literal } from './ast'
import { evalGlobalCallValues, evalMethodCallValues } from './interpreter/builtin-calls'
import {
  accessProperty,
  getFrontmatterPropValue,
  resolveFileProperty,
  resolveIdentifier,
  vaultPath,
} from './interpreter/file-context'
import { literalToValue, toValue, valueToBoolean } from './interpreter/value-conversion'
import {
  emptyValueList,
  isFileValue,
  isListValue,
  isStringValue,
  makeBoolean,
  makeHtml,
  makeList,
  makeNull,
  makeNumber,
  trueValue,
  type EvalContext,
  type Value,
} from './interpreter/value-model'
import { accessIndex, applyBinary, applyUnary } from './interpreter/value-operators'
import { Instruction, ProgramIR } from './ir'
export type {
  EvalContext,
  Value,
  NullValue,
  BooleanValue,
  NumberValue,
  StringValue,
  DateValue,
  DurationValue,
  ListValue,
  ObjectValue,
  FileValue,
  LinkValue,
  RegexValue,
  HtmlValue,
  IconValue,
  ImageValue,
  ValueKind,
  ValueOf,
} from './interpreter/value-model'
export { isValueKind } from './interpreter/value-model'
export { valueToUnknown } from './interpreter/value-conversion'

const evaluateProgram = (program: ProgramIR, ctx: EvalContext): Value => {
  const stack: Value[] = []
  const instructions = program.instructions
  let ip = 0

  const popValue = (): Value => stack.pop() ?? makeNull()
  const popArgs = (count: number): Value[] => {
    if (count <= 0) return emptyValueList
    if (count === 1) return [stack.pop() ?? makeNull()]
    const start = Math.max(0, stack.length - count)
    return stack.splice(start, stack.length - start)
  }

  while (ip < instructions.length) {
    const instr = instructions[ip] as Instruction
    switch (instr.op) {
      case 'const':
        stack.push(literalToValue(instr.literal))
        break
      case 'ident': {
        if (instr.name === 'file' && !(ctx.locals && 'file' in ctx.locals)) {
          const nextInstr = instructions[ip + 1]
          if (nextInstr?.op === 'member') {
            stack.push(resolveFileProperty(ctx.file, nextInstr.property, ctx))
            ip += 2
            continue
          }
          const thirdInstr = instructions[ip + 2]
          if (
            nextInstr?.op === 'const' &&
            nextInstr.literal.kind === 'string' &&
            thirdInstr?.op === 'call_method' &&
            thirdInstr.name === 'inFolder' &&
            thirdInstr.argc === 1
          ) {
            const rawFolder = nextInstr.literal.value
            const folder =
              rawFolder.startsWith('/') || rawFolder.endsWith('/')
                ? rawFolder.replace(/^\/+|\/+$/g, '')
                : rawFolder
            if (!folder) {
              stack.push(trueValue)
            } else {
              const vp = vaultPath(ctx.file)
              stack.push(makeBoolean(vp.startsWith(folder) && vp.charCodeAt(folder.length) === 47))
            }
            ip += 3
            continue
          }
        }
        if (
          instr.name === 'note' &&
          !(ctx.locals && 'note' in ctx.locals) &&
          instructions[ip + 1]?.op === 'member'
        ) {
          const memberInstr = instructions[ip + 1] as { op: 'member'; property: string }
          stack.push(
            getFrontmatterPropValue(
              ctx.file.frontmatter as Record<string, unknown> | undefined,
              memberInstr.property,
            ),
          )
          ip += 2
          continue
        }
        if (
          instr.name === 'note' &&
          !(ctx.locals && 'note' in ctx.locals) &&
          instructions[ip + 1]?.op === 'const' &&
          (instructions[ip + 1] as { op: 'const'; literal: Literal }).literal.kind === 'string' &&
          instructions[ip + 2]?.op === 'index'
        ) {
          const key = (
            (instructions[ip + 1] as { op: 'const'; literal: Literal }).literal as {
              kind: 'string'
              value: string
            }
          ).value
          stack.push(
            getFrontmatterPropValue(
              ctx.file.frontmatter as Record<string, unknown> | undefined,
              key,
            ),
          )
          ip += 3
          continue
        }
        stack.push(resolveIdentifier(instr.name, ctx))
        break
      }
      case 'load_formula':
        stack.push(resolveFormulaProperty(instr.name, ctx))
        break
      case 'load_formula_index': {
        const indexValue = popValue()
        if (isStringValue(indexValue)) {
          stack.push(resolveFormulaProperty(indexValue.value, ctx))
        } else {
          stack.push(makeNull())
        }
        break
      }
      case 'member': {
        const objectValue = popValue()
        if (
          instr.property === 'properties' &&
          isFileValue(objectValue) &&
          instructions[ip + 1]?.op === 'member'
        ) {
          const nextMember = instructions[ip + 1] as { op: 'member'; property: string }
          stack.push(
            getFrontmatterPropValue(
              objectValue.value.frontmatter as Record<string, unknown> | undefined,
              nextMember.property,
            ),
          )
          ip += 2
          continue
        }
        stack.push(accessProperty(objectValue, instr.property, ctx))
        break
      }
      case 'index': {
        const indexValue = popValue()
        const objectValue = popValue()
        stack.push(accessIndex(objectValue, indexValue))
        break
      }
      case 'list': {
        const count = Math.max(0, instr.count)
        const items = count > 0 ? stack.splice(stack.length - count, count) : []
        stack.push(makeList(items))
        break
      }
      case 'unary': {
        const value = popValue()
        stack.push(applyUnary(instr.operator, value))
        break
      }
      case 'binary': {
        const right = popValue()
        const left = popValue()
        stack.push(applyBinary(instr.operator, left, right, ctx))
        break
      }
      case 'to_bool': {
        const value = popValue()
        stack.push(makeBoolean(valueToBoolean(value)))
        break
      }
      case 'call_global': {
        const args = popArgs(instr.argc)
        stack.push(evalGlobalCallValues(instr.name, args, ctx, instr.span))
        break
      }
      case 'call_method': {
        const args = popArgs(instr.argc)
        const receiver = popValue()
        stack.push(evalMethodCallValues(receiver, instr.name, args, ctx, instr.span))
        break
      }
      case 'call_dynamic': {
        const calleeValue = popValue()
        if (calleeValue.kind === 'html') {
          stack.push(makeHtml(calleeValue.value))
        } else {
          stack.push(makeNull())
        }
        break
      }
      case 'filter': {
        const receiver = popValue()
        stack.push(applyListFilter(receiver, instr.program, ctx))
        break
      }
      case 'map': {
        const receiver = popValue()
        stack.push(applyListMap(receiver, instr.program, ctx))
        break
      }
      case 'reduce': {
        const receiver = popValue()
        stack.push(applyListReduce(receiver, instr.program, instr.initial, ctx))
        break
      }
      case 'jump':
        ip = instr.target
        continue
      case 'jump_if_false': {
        const value = popValue()
        if (!valueToBoolean(value)) {
          ip = instr.target
          continue
        }
        break
      }
      case 'jump_if_true': {
        const value = popValue()
        if (valueToBoolean(value)) {
          ip = instr.target
          continue
        }
        break
      }
    }
    ip += 1
  }

  return popValue()
}

export const evaluateExpression = (program: ProgramIR, ctx: EvalContext): Value =>
  evaluateProgram(program, ctx)

export const evaluateFilterExpression = (program: ProgramIR, ctx: EvalContext): boolean =>
  valueToBoolean(evaluateExpression(program, ctx))

export const evaluateSummaryExpression = (
  program: ProgramIR,
  values: unknown[],
  ctx: EvalContext,
): Value => {
  const valueList = values.map(toValue)
  const summaryCtx: EvalContext = { ...ctx, values: valueList }
  return evaluateExpression(program, summaryCtx)
}

const applyListFilter = (receiver: Value, program: ProgramIR | null, ctx: EvalContext): Value => {
  if (!isListValue(receiver)) return makeNull()
  if (!program) return makeList(receiver.value)
  const list = receiver.value
  const filtered = list.filter((value, index) => {
    const locals: Record<string, Value> = { value, index: makeNumber(index) }
    const baseLocals = ctx.locals ? ctx.locals : {}
    const fileCtx = isFileValue(value)
      ? {
          ...ctx,
          file: value.value,
          propertyCache: undefined,
          formulaCache: undefined,
          formulaStack: new Set<string>(),
        }
      : ctx
    const nextCtx: EvalContext = { ...fileCtx, locals: { ...baseLocals, ...locals } }
    return valueToBoolean(evaluateProgram(program, nextCtx))
  })
  return makeList(filtered)
}

const applyListMap = (receiver: Value, program: ProgramIR | null, ctx: EvalContext): Value => {
  if (!isListValue(receiver)) return makeNull()
  if (!program) return makeList(receiver.value)
  const list = receiver.value
  const mapped = list.map((value, index) => {
    const locals: Record<string, Value> = { value, index: makeNumber(index) }
    const baseLocals = ctx.locals ? ctx.locals : {}
    const fileCtx = isFileValue(value)
      ? {
          ...ctx,
          file: value.value,
          propertyCache: undefined,
          formulaCache: undefined,
          formulaStack: new Set<string>(),
        }
      : ctx
    const nextCtx: EvalContext = { ...fileCtx, locals: { ...baseLocals, ...locals } }
    return evaluateProgram(program, nextCtx)
  })
  return makeList(mapped)
}

const applyListReduce = (
  receiver: Value,
  program: ProgramIR | null,
  initial: ProgramIR | null,
  ctx: EvalContext,
): Value => {
  if (!isListValue(receiver)) return makeNull()
  const initialValue = initial ? evaluateProgram(initial, ctx) : makeNull()
  if (!program) return initialValue
  let acc = initialValue
  const list = receiver.value
  for (let index = 0; index < list.length; index += 1) {
    const value = list[index]
    const locals: Record<string, Value> = { value, index: makeNumber(index), acc }
    const baseLocals = ctx.locals ? ctx.locals : {}
    const fileCtx = isFileValue(value)
      ? {
          ...ctx,
          file: value.value,
          propertyCache: undefined,
          formulaCache: undefined,
          formulaStack: new Set<string>(),
        }
      : ctx
    const nextCtx: EvalContext = { ...fileCtx, locals: { ...baseLocals, ...locals } }
    acc = evaluateProgram(program, nextCtx)
  }
  return acc
}

const resolveFormulaProperty = (name: string, ctx: EvalContext): Value => {
  if (!ctx.formulas || !ctx.formulas[name]) return makeNull()
  if (!ctx.formulaCache) ctx.formulaCache = new Map()
  if (!ctx.formulaStack) ctx.formulaStack = new Set()
  const cached = ctx.formulaCache.get(name)
  if (cached) return cached
  if (ctx.formulaStack.has(name)) return makeNull()
  ctx.formulaStack.add(name)
  const expr = ctx.formulas[name]
  const nextCtx: EvalContext = {
    ...ctx,
    diagnosticContext: `formula.${name}`,
    diagnosticSource: ctx.formulaSources?.[name] ?? ctx.diagnosticSource,
  }
  const value = evaluateExpression(expr, nextCtx)
  ctx.formulaCache.set(name, value)
  ctx.formulaStack.delete(name)
  return value
}
