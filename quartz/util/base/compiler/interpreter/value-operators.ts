import { BinaryExpr } from '../ast'
import { resolveLinkComparisonKey } from './file-context'
import {
  addDurationToDate,
  coerceToDate,
  coerceToNumber,
  isScalarValue,
  parseDurationValue,
  valueToBoolean,
  valueToNumber,
  valueToString,
} from './value-conversion'
import {
  isBooleanValue,
  isDateValue,
  isDurationValue,
  isFileValue,
  isLinkValue,
  isListValue,
  isNumberValue,
  isObjectValue,
  isRegexValue,
  isStringValue,
  makeBoolean,
  makeDate,
  makeDuration,
  makeNull,
  makeNumber,
  makeString,
  type EvalContext,
  type Value,
} from './value-model'

export const valueEquals = (left: Value, right: Value, ctx: EvalContext): boolean => {
  const leftLinkish = isLinkValue(left) || isFileValue(left)
  const rightLinkish = isLinkValue(right) || isFileValue(right)
  if (
    (leftLinkish && (rightLinkish || isStringValue(right))) ||
    (rightLinkish && isStringValue(left))
  ) {
    const leftKey = resolveLinkComparisonKey(left, ctx)
    const rightKey = resolveLinkComparisonKey(right, ctx)
    if (leftKey.slug && rightKey.slug) return leftKey.slug === rightKey.slug
    if (!leftKey.slug && !rightKey.slug) return leftKey.text === rightKey.text
    return false
  }
  if (left.kind !== right.kind) {
    if (isScalarValue(left) && isScalarValue(right)) {
      const leftDate = coerceToDate(left)
      const rightDate = coerceToDate(right)
      if (leftDate && rightDate) return leftDate.getTime() === rightDate.getTime()
      const leftNum = coerceToNumber(left)
      const rightNum = coerceToNumber(right)
      if (leftNum !== null && rightNum !== null) return leftNum === rightNum
      return valueToString(left) === valueToString(right)
    }
    return false
  }
  if (left.kind === 'null') return true
  if (isBooleanValue(left) && isBooleanValue(right)) return left.value === right.value
  if (isNumberValue(left) && isNumberValue(right)) return left.value === right.value
  if (isStringValue(left) && isStringValue(right)) return left.value === right.value
  if (isDateValue(left) && isDateValue(right)) return left.value.getTime() === right.value.getTime()
  if (isDurationValue(left) && isDurationValue(right)) {
    return left.value === right.value && left.months === right.months
  }
  if (isRegexValue(left) && isRegexValue(right)) return left.value.source === right.value.source
  if (isListValue(left) && isListValue(right)) {
    if (left.value.length !== right.value.length) return false
    for (let i = 0; i < left.value.length; i += 1) {
      if (!valueEquals(left.value[i], right.value[i], ctx)) return false
    }
    return true
  }
  if (isObjectValue(left) && isObjectValue(right)) {
    const leftKeys = Object.keys(left.value)
    const rightKeys = Object.keys(right.value)
    if (leftKeys.length !== rightKeys.length) return false
    for (const key of leftKeys) {
      const l = left.value[key]
      const r = right.value[key]
      if (!r || !valueEquals(l, r, ctx)) return false
    }
    return true
  }
  return false
}

export const applyUnary = (operator: '!' | '-', value: Value): Value => {
  if (operator === '!') {
    return makeBoolean(!valueToBoolean(value))
  }
  const num = valueToNumber(value)
  return Number.isFinite(num) ? makeNumber(-num) : makeNull()
}

export const applyBinary = (
  operator: BinaryExpr['operator'],
  left: Value,
  right: Value,
  ctx: EvalContext,
): Value => {
  if (operator === '==') return makeBoolean(valueEquals(left, right, ctx))
  if (operator === '!=') return makeBoolean(!valueEquals(left, right, ctx))

  if (operator === '+' || operator === '-') {
    return evalAdditive(operator, left, right)
  }
  if (operator === '*' || operator === '/' || operator === '%') {
    if (isDurationValue(left)) {
      const rightNum = valueToNumber(right)
      if (!Number.isFinite(rightNum)) return makeNull()
      if (operator === '*') return makeDuration(left.value * rightNum, left.months * rightNum)
      if (operator === '/') {
        if (rightNum === 0) return makeNull()
        return makeDuration(left.value / rightNum, left.months / rightNum)
      }
      return makeNull()
    }
    if (isDurationValue(right)) return makeNull()
    const leftNum = valueToNumber(left)
    const rightNum = valueToNumber(right)
    if (!Number.isFinite(leftNum) || !Number.isFinite(rightNum)) return makeNull()
    if (operator === '*') return makeNumber(leftNum * rightNum)
    if (operator === '/') return makeNumber(rightNum === 0 ? Number.NaN : leftNum / rightNum)
    return makeNumber(rightNum === 0 ? Number.NaN : leftNum % rightNum)
  }

  const compare = compareValues(left, right)
  if (compare === null) return makeNull()
  if (operator === '>') return makeBoolean(compare > 0)
  if (operator === '>=') return makeBoolean(compare >= 0)
  if (operator === '<') return makeBoolean(compare < 0)
  if (operator === '<=') return makeBoolean(compare <= 0)
  return makeNull()
}

const evalAdditive = (operator: '+' | '-', left: Value, right: Value): Value => {
  if (isDateValue(left) && isDateValue(right) && operator === '-') {
    return makeDuration(left.value.getTime() - right.value.getTime())
  }
  if (isDateValue(left)) {
    const duration = parseDurationValue(right)
    if (duration === null) return makeNull()
    return makeDate(addDurationToDate(left.value, duration, operator === '+' ? 1 : -1))
  }
  if (isDateValue(right) && operator === '+') {
    const duration = parseDurationValue(left)
    if (duration === null) return makeNull()
    return makeDate(addDurationToDate(right.value, duration, 1))
  }
  if (operator === '+' && (isStringValue(left) || isStringValue(right))) {
    return makeString(`${valueToString(left)}${valueToString(right)}`)
  }
  if (isDurationValue(left) && isDurationValue(right)) {
    return makeDuration(
      operator === '+' ? left.value + right.value : left.value - right.value,
      operator === '+' ? left.months + right.months : left.months - right.months,
    )
  }
  const leftNum = valueToNumber(left)
  const rightNum = valueToNumber(right)
  if (!Number.isFinite(leftNum) || !Number.isFinite(rightNum)) return makeNull()
  return makeNumber(operator === '+' ? leftNum + rightNum : leftNum - rightNum)
}

const compareValues = (left: Value, right: Value): number | null => {
  const leftDate = coerceToDate(left)
  const rightDate = coerceToDate(right)
  if (leftDate && rightDate) return leftDate.getTime() - rightDate.getTime()

  const leftNum = coerceToNumber(left)
  const rightNum = coerceToNumber(right)
  if (leftNum !== null && rightNum !== null) return leftNum - rightNum

  if (isScalarValue(left) && isScalarValue(right)) {
    const leftStr = valueToString(left)
    const rightStr = valueToString(right)
    if (leftStr === rightStr) return 0
    return leftStr > rightStr ? 1 : -1
  }

  return null
}

export const accessIndex = (objectValue: Value, indexValue: Value): Value => {
  if (isListValue(objectValue)) {
    const index = Math.trunc(valueToNumber(indexValue))
    if (!Number.isFinite(index)) return makeNull()
    const item = objectValue.value[index]
    return item ?? makeNull()
  }
  if (isObjectValue(objectValue) && isStringValue(indexValue)) {
    const item = objectValue.value[indexValue.value]
    return item ?? makeNull()
  }
  return makeNull()
}
