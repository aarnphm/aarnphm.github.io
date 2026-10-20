import { BooleanLiteral, Expr, Span } from './ast'

const MAX_ISOP_VARS = 12

type CanonicalAtom = { key: string; expr: Expr; cost: number }

type CubeLiteral = { varIndex: number; negated: boolean }

type Cube = CubeLiteral[]

const emptySpan: Span = {
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
}

const serializeCanonicalExpr = (node: Expr): string => {
  switch (node.type) {
    case 'Literal':
      return `L:${node.kind}:${String(node.value)}:${node.kind === 'regex' ? node.flags : ''}`
    case 'Identifier':
      return `I:${node.name}`
    case 'MemberExpr':
      return `M:${serializeCanonicalExpr(node.object)}.${node.property}`
    case 'IndexExpr':
      return `X:${serializeCanonicalExpr(node.object)}[${serializeCanonicalExpr(node.index)}]`
    case 'UnaryExpr':
      return `U:${node.operator}(${serializeCanonicalExpr(node.argument)})`
    case 'BinaryExpr': {
      const left = serializeCanonicalExpr(node.left)
      const right = serializeCanonicalExpr(node.right)
      if ((node.operator === '==' || node.operator === '!=') && left > right) {
        return `B:${node.operator}(${right},${left})`
      }
      return `B:${node.operator}(${left},${right})`
    }
    case 'LogicalExpr':
      return `G:${node.operator}(${serializeCanonicalExpr(node.left)},${serializeCanonicalExpr(node.right)})`
    case 'CallExpr':
      return `C:${serializeCanonicalExpr(node.callee)}(${node.args.map(serializeCanonicalExpr).join(',')})`
    case 'ListExpr':
      return `A:[${node.elements.map(serializeCanonicalExpr).join(',')}]`
    case 'ErrorExpr':
      return `E:${node.message}`
  }
}

const containsFormulaReference = (node: Expr): boolean => {
  switch (node.type) {
    case 'Literal':
    case 'ErrorExpr':
      return false
    case 'Identifier':
      return node.name === 'formula'
    case 'MemberExpr':
      return containsFormulaReference(node.object)
    case 'IndexExpr':
      return containsFormulaReference(node.object) || containsFormulaReference(node.index)
    case 'UnaryExpr':
      return containsFormulaReference(node.argument)
    case 'BinaryExpr':
    case 'LogicalExpr':
      return containsFormulaReference(node.left) || containsFormulaReference(node.right)
    case 'CallExpr':
      return containsFormulaReference(node.callee) || node.args.some(containsFormulaReference)
    case 'ListExpr':
      return node.elements.some(containsFormulaReference)
  }
}

const isFastFileProperty = (node: Expr): boolean =>
  node.type === 'MemberExpr' &&
  node.object.type === 'Identifier' &&
  node.object.name === 'file' &&
  (node.property === 'ext' ||
    node.property === 'name' ||
    node.property === 'basename' ||
    node.property === 'path' ||
    node.property === 'folder')

export const estimatePredicateCost = (node: Expr): number => {
  if (containsFormulaReference(node)) return 100

  switch (node.type) {
    case 'Literal':
    case 'ErrorExpr':
      return 1
    case 'Identifier':
      return 12
    case 'UnaryExpr':
      return estimatePredicateCost(node.argument) + 1
    case 'MemberExpr': {
      if (isFastFileProperty(node)) return 8
      if (node.object.type === 'Identifier' && node.object.name === 'file') {
        if (
          node.property === 'links' ||
          node.property === 'outlinks' ||
          node.property === 'backlinks' ||
          node.property === 'inlinks'
        ) {
          return 75
        }
        return 18
      }
      return 20
    }
    case 'IndexExpr':
      return 22
    case 'BinaryExpr': {
      if (
        (node.operator === '==' || node.operator === '!=') &&
        ((isFastFileProperty(node.left) && node.right.type === 'Literal') ||
          (isFastFileProperty(node.right) && node.left.type === 'Literal'))
      ) {
        return 10
      }
      return (
        Math.max(estimatePredicateCost(node.left), estimatePredicateCost(node.right)) +
        (node.operator === '==' || node.operator === '!=' ? 4 : 8)
      )
    }
    case 'CallExpr': {
      if (node.callee.type === 'MemberExpr') {
        const method = node.callee.property
        const obj = node.callee.object
        if (obj.type === 'Identifier' && obj.name === 'file') {
          if (method === 'inFolder') return 6
          if (method === 'hasProperty') return 16
          if (method === 'hasTag') return 42 + node.args.length * 2
          if (method === 'hasLink') return 70
        }
        if (method === 'isEmpty' || method === 'startsWith' || method === 'endsWith') {
          return 24
        }
        if (method === 'contains' || method === 'containsAny' || method === 'containsAll') {
          return 45 + node.args.length * 2
        }
        if (method === 'filter' || method === 'map' || method === 'reduce') {
          return 95
        }
      }
      return 50
    }
    case 'LogicalExpr':
      return estimatePredicateCost(node.left) + estimatePredicateCost(node.right)
    case 'ListExpr':
      return 30 + node.elements.length * 2
  }
}

type BoolFormula =
  | { kind: 'const'; value: boolean }
  | { kind: 'lit'; atomKey: string; negated: boolean }
  | { kind: 'not'; arg: BoolFormula }
  | { kind: 'and'; left: BoolFormula; right: BoolFormula }
  | { kind: 'or'; left: BoolFormula; right: BoolFormula }

const extractBoolFormula = (
  node: Expr,
  negated: boolean,
  atomsByKey: Map<string, CanonicalAtom>,
): BoolFormula => {
  if (node.type === 'Literal' && node.kind === 'boolean') {
    return { kind: 'const', value: negated ? !node.value : node.value }
  }
  if (node.type === 'UnaryExpr' && node.operator === '!') {
    return extractBoolFormula(node.argument, !negated, atomsByKey)
  }
  if (node.type === 'LogicalExpr') {
    if (node.operator === '&&') {
      const left = extractBoolFormula(node.left, negated, atomsByKey)
      const right = extractBoolFormula(node.right, negated, atomsByKey)
      return negated ? { kind: 'or', left, right } : { kind: 'and', left, right }
    }
    const left = extractBoolFormula(node.left, negated, atomsByKey)
    const right = extractBoolFormula(node.right, negated, atomsByKey)
    return negated ? { kind: 'and', left, right } : { kind: 'or', left, right }
  }

  let positiveExpr: Expr = node
  let atomNegated = negated
  if (node.type === 'BinaryExpr' && node.operator === '!=') {
    positiveExpr = { ...node, operator: '==' }
    atomNegated = !negated
  }

  const key = serializeCanonicalExpr(positiveExpr)
  if (!atomsByKey.has(key)) {
    atomsByKey.set(key, { key, expr: positiveExpr, cost: estimatePredicateCost(positiveExpr) })
  }
  return { kind: 'lit', atomKey: key, negated: atomNegated }
}

const evaluateFormulaOnAssignment = (
  formula: BoolFormula,
  assignment: number,
  keyToBit: Map<string, number>,
): boolean => {
  switch (formula.kind) {
    case 'const':
      return formula.value
    case 'lit': {
      const bit = keyToBit.get(formula.atomKey) ?? 0
      const val = ((assignment >>> bit) & 1) === 1
      return formula.negated ? !val : val
    }
    case 'not':
      return !evaluateFormulaOnAssignment(formula.arg, assignment, keyToBit)
    case 'and':
      return (
        evaluateFormulaOnAssignment(formula.left, assignment, keyToBit) &&
        evaluateFormulaOnAssignment(formula.right, assignment, keyToBit)
      )
    case 'or':
      return (
        evaluateFormulaOnAssignment(formula.left, assignment, keyToBit) ||
        evaluateFormulaOnAssignment(formula.right, assignment, keyToBit)
      )
  }
}

type IsopResult32 = { cubes: Cube[]; covered: number }

const FULL_MASK_32 = (bits: number): number => (bits === 32 ? 0xffffffff : (1 << bits) - 1)

/**
 * Minato-Morreale recursive Irredundant Sum-of-Products (ISOP) algorithm
 * for 32-bit truth-table intervals [lower, upper] over k <= 5 variables.
 *
 * Given an interval [L, U] with L => U:
 *   L0' = L0 & ~U1,  (G0, cov0) = ISOP(L0', U0)   (cubes requiring ~v)
 *   L1' = L1 & ~U0,  (G1, cov1) = ISOP(L1', U1)   (cubes requiring  v)
 *   L*  = (L0 & ~cov0) | (L1 & ~cov1), U* = U0 & U1
 *   (G*, cov*) = ISOP(L*, U*)                      (cubes independent of v)
 */
const isop32 = (lower: number, upper: number, numVars: number, varOffset: number): IsopResult32 => {
  const numMinterms = 1 << numVars
  const fullMask = FULL_MASK_32(numMinterms)
  const L = (lower & fullMask) >>> 0
  const U = (upper & fullMask) >>> 0

  if (L === 0) {
    return { cubes: [], covered: 0 }
  }
  if (U === fullMask) {
    return { cubes: [[]], covered: fullMask }
  }
  if (numVars === 0) {
    return { cubes: [[]], covered: 1 }
  }

  const halfBits = 1 << (numVars - 1)
  const halfMask = FULL_MASK_32(halfBits)
  const L0 = L & halfMask
  const L1 = (L >>> halfBits) & halfMask
  const U0 = U & halfMask
  const U1 = (U >>> halfBits) & halfMask

  const l0Prime = (L0 & ~U1 & halfMask) >>> 0
  const res0 = isop32(l0Prime, U0, numVars - 1, varOffset + 1)

  const l1Prime = (L1 & ~U0 & halfMask) >>> 0
  const res1 = isop32(l1Prime, U1, numVars - 1, varOffset + 1)

  const l0Rem = (L0 & ~res0.covered & halfMask) >>> 0
  const l1Rem = (L1 & ~res1.covered & halfMask) >>> 0
  const lStar = (l0Rem | l1Rem) >>> 0
  const uStar = (U0 & U1 & halfMask) >>> 0
  const resStar = isop32(lStar, uStar, numVars - 1, varOffset + 1)

  const cubes: Cube[] = []
  for (const cube of res0.cubes) {
    cubes.push([{ varIndex: varOffset, negated: true }, ...cube])
  }
  for (const cube of res1.cubes) {
    cubes.push([{ varIndex: varOffset, negated: false }, ...cube])
  }
  for (const cube of resStar.cubes) {
    cubes.push(cube)
  }

  const cov0 = (res0.covered | resStar.covered) & halfMask
  const cov1 = (res1.covered | resStar.covered) & halfMask
  const covered = (cov0 | (cov1 << halfBits)) >>> 0

  return { cubes, covered }
}

type IsopResultWords = { cubes: Cube[]; covered: Uint32Array }

const isAllZeroWords = (words: Uint32Array): boolean => {
  for (let i = 0; i < words.length; i += 1) {
    if (words[i] !== 0) return false
  }
  return true
}

const isAllOnesWords = (words: Uint32Array): boolean => {
  for (let i = 0; i < words.length; i += 1) {
    if (words[i] !== 0xffffffff) return false
  }
  return true
}

const isopWords = (
  lower: Uint32Array,
  upper: Uint32Array,
  numVars: number,
  varOffset: number,
): IsopResultWords => {
  if (numVars <= 5) {
    const res32 = isop32(lower[0] ?? 0, upper[0] ?? 0, numVars, varOffset)
    return { cubes: res32.cubes, covered: new Uint32Array([res32.covered]) }
  }

  if (isAllZeroWords(lower)) {
    return { cubes: [], covered: new Uint32Array(lower.length) }
  }
  if (isAllOnesWords(upper)) {
    const covered = new Uint32Array(upper.length)
    covered.fill(0xffffffff)
    return { cubes: [[]], covered }
  }

  const halfLen = lower.length >>> 1
  const L0 = lower.subarray(0, halfLen)
  const L1 = lower.subarray(halfLen)
  const U0 = upper.subarray(0, halfLen)
  const U1 = upper.subarray(halfLen)

  const l0Prime = new Uint32Array(halfLen)
  const l1Prime = new Uint32Array(halfLen)
  for (let i = 0; i < halfLen; i += 1) {
    l0Prime[i] = (L0[i] & ~U1[i]) >>> 0
    l1Prime[i] = (L1[i] & ~U0[i]) >>> 0
  }

  const res0 = isopWords(l0Prime, U0, numVars - 1, varOffset + 1)
  const res1 = isopWords(l1Prime, U1, numVars - 1, varOffset + 1)

  const lStar = new Uint32Array(halfLen)
  const uStar = new Uint32Array(halfLen)
  for (let i = 0; i < halfLen; i += 1) {
    const l0Rem = (L0[i] & ~res0.covered[i]) >>> 0
    const l1Rem = (L1[i] & ~res1.covered[i]) >>> 0
    lStar[i] = (l0Rem | l1Rem) >>> 0
    uStar[i] = (U0[i] & U1[i]) >>> 0
  }

  const resStar = isopWords(lStar, uStar, numVars - 1, varOffset + 1)

  const cubes: Cube[] = []
  for (const cube of res0.cubes) {
    cubes.push([{ varIndex: varOffset, negated: true }, ...cube])
  }
  for (const cube of res1.cubes) {
    cubes.push([{ varIndex: varOffset, negated: false }, ...cube])
  }
  for (const cube of resStar.cubes) {
    cubes.push(cube)
  }

  const covered = new Uint32Array(lower.length)
  for (let i = 0; i < halfLen; i += 1) {
    covered[i] = (res0.covered[i] | resStar.covered[i]) >>> 0
    covered[halfLen + i] = (res1.covered[i] | resStar.covered[i]) >>> 0
  }

  return { cubes, covered }
}

const literalToExpr = (lit: CubeLiteral, atoms: CanonicalAtom[]): Expr => {
  const atom = atoms[lit.varIndex]
  if (!lit.negated) {
    return atom.expr
  }
  if (atom.expr.type === 'BinaryExpr' && atom.expr.operator === '==') {
    return { ...atom.expr, operator: '!=' }
  }
  return { type: 'UnaryExpr', operator: '!', argument: atom.expr, span: atom.expr.span }
}

const literalCost = (lit: CubeLiteral, atoms: CanonicalAtom[]): number => {
  const base = atoms[lit.varIndex].cost
  // Negative literals (e.g. `!= "index"` or `!file.inFolder("library/posters")`) have high
  // pass-through probability, so evaluate positive folder/equality guards ahead of negated guards
  // within the same cost tier while keeping all fast file guards ahead of frontmatter/formulas.
  return lit.negated ? base + 9 : base
}

const cubeCost = (cube: Cube, atoms: CanonicalAtom[]): number => {
  let total = 0
  for (const lit of cube) {
    total += literalCost(lit, atoms)
  }
  return total
}

const makeBooleanLiteral = (value: boolean, span: Span = emptySpan): BooleanLiteral => ({
  type: 'Literal',
  kind: 'boolean',
  value,
  span,
})

const collectAssociativeOperands = (node: Expr, operator: '&&' | '||', out: Expr[]): void => {
  if (node.type === 'LogicalExpr' && node.operator === operator) {
    collectAssociativeOperands(node.left, operator, out)
    collectAssociativeOperands(node.right, operator, out)
    return
  }
  out.push(node)
}

const fallbackCostOrderLogicalTree = (node: Expr): Expr => {
  if (node.type === 'UnaryExpr' && node.operator === '!') {
    const inner = node.argument
    if (inner.type === 'UnaryExpr' && inner.operator === '!') {
      return fallbackCostOrderLogicalTree(inner.argument)
    }
    if (inner.type === 'BinaryExpr' && inner.operator === '==') {
      return { ...inner, operator: '!=' }
    }
    if (inner.type === 'BinaryExpr' && inner.operator === '!=') {
      return { ...inner, operator: '==' }
    }
  }
  if (node.type === 'LogicalExpr') {
    const operands: Expr[] = []
    collectAssociativeOperands(node, node.operator, operands)
    const optimizedOperands = operands
      .map(fallbackCostOrderLogicalTree)
      .sort((a, b) => estimatePredicateCost(a) - estimatePredicateCost(b))
    let current = optimizedOperands[0]
    for (let i = 1; i < optimizedOperands.length; i += 1) {
      current = {
        type: 'LogicalExpr',
        operator: node.operator,
        left: current,
        right: optimizedOperands[i],
        span: node.span,
      }
    }
    return current
  }
  return node
}

/**
 * Minimizes a boolean filter AST into an irredundant sum-of-products (ISOP)
 * via the Minato-Morreale recursive algorithm, with literals inside each
 * prime-implicant cube ordered by ascending evaluation cost.
 */
export const optimizeFilterExprWithIsop = (node: Expr): Expr => {
  const atomsByKey = new Map<string, CanonicalAtom>()
  const formula = extractBoolFormula(node, false, atomsByKey)

  if (atomsByKey.size === 0) {
    const val = evaluateFormulaOnAssignment(formula, 0, new Map())
    return makeBooleanLiteral(val, node.span)
  }

  if (atomsByKey.size > MAX_ISOP_VARS) {
    return fallbackCostOrderLogicalTree(node)
  }

  // Order variables by ascending evaluation cost so cheaper variables split first
  // and appear first inside every prime-implicant cube.
  const atoms = [...atomsByKey.values()].sort((a, b) =>
    a.cost !== b.cost ? a.cost - b.cost : a.key.localeCompare(b.key),
  )
  const numVars = atoms.length
  const keyToBit = new Map<string, number>()
  // Variable 0 is the top-level split variable (highest bit numVars - 1 in minterm index)
  for (let i = 0; i < numVars; i += 1) {
    keyToBit.set(atoms[i].key, numVars - 1 - i)
  }

  const numMinterms = 1 << numVars
  let cubes: Cube[]
  if (numVars <= 5) {
    let truthTable = 0
    for (let m = 0; m < numMinterms; m += 1) {
      if (evaluateFormulaOnAssignment(formula, m, keyToBit)) {
        truthTable = (truthTable | (1 << m)) >>> 0
      }
    }
    cubes = isop32(truthTable, truthTable, numVars, 0).cubes
  } else {
    const wordCount = numMinterms >>> 5
    const truthWords = new Uint32Array(wordCount)
    for (let m = 0; m < numMinterms; m += 1) {
      if (evaluateFormulaOnAssignment(formula, m, keyToBit)) {
        truthWords[m >>> 5] = (truthWords[m >>> 5] | (1 << (m & 31))) >>> 0
      }
    }
    cubes = isopWords(truthWords, truthWords, numVars, 0).cubes
  }

  if (cubes.length === 0) {
    return makeBooleanLiteral(false, node.span)
  }
  if (cubes.length === 1 && cubes[0].length === 0) {
    return makeBooleanLiteral(true, node.span)
  }

  // Sort literals inside each cube by ascending selectivity-adjusted cost, and cubes by ascending total cost
  for (const cube of cubes) {
    cube.sort((a, b) => {
      const diff = literalCost(a, atoms) - literalCost(b, atoms)
      return diff !== 0 ? diff : a.varIndex - b.varIndex
    })
  }
  cubes.sort((a, b) => {
    const costDiff = cubeCost(a, atoms) - cubeCost(b, atoms)
    return costDiff !== 0 ? costDiff : a.length - b.length
  })

  const cubeExprs = cubes.map(cube => {
    if (cube.length === 0) return makeBooleanLiteral(true, node.span)
    let conj = literalToExpr(cube[0], atoms)
    for (let i = 1; i < cube.length; i += 1) {
      conj = {
        type: 'LogicalExpr',
        operator: '&&',
        left: conj,
        right: literalToExpr(cube[i], atoms),
        span: node.span,
      }
    }
    return conj
  })

  let disj = cubeExprs[0]
  for (let i = 1; i < cubeExprs.length; i += 1) {
    disj = { type: 'LogicalExpr', operator: '||', left: disj, right: cubeExprs[i], span: node.span }
  }
  return disj
}
