import {
  defaultDevice,
  devicePut,
  grad,
  init,
  jit,
  makeJaxpr,
  nn,
  numpy as np,
  random,
  valueAndGrad,
} from '@jax-js/jax'

class NotebookMlUnsupportedError extends Error {
  constructor(message) {
    super(message)
    this.name = 'UnsupportedRuntimeFeature'
  }
}

let runtimePromise
let runtimeReady = false
let nextValueId = 1
let wrapPythonTree
let describePythonTree
const values = new Map()
const jittedFunctions = new WeakMap()

function textOf(value) {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  try {
    return String(value)
  } catch {
    return Object.prototype.toString.call(value)
  }
}

function ensureWebGpuSurface() {
  if (!globalThis.navigator || !globalThis.navigator.gpu) {
    throw new NotebookMlUnsupportedError('WebGPU is required for browser JAX/Torch notebook cells')
  }
}

export async function ensureNotebookMlRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      ensureWebGpuSurface()
      const available = await init('webgpu')
      if (!available.includes('webgpu')) {
        throw new NotebookMlUnsupportedError(
          'WebGPU is required for browser JAX/Torch notebook cells',
        )
      }
      defaultDevice('webgpu')
      runtimeReady = true
    })().catch(error => {
      runtimePromise = undefined
      runtimeReady = false
      if (error instanceof NotebookMlUnsupportedError) throw error
      throw new NotebookMlUnsupportedError(
        `WebGPU is required for browser JAX/Torch notebook cells: ${textOf(error)}`,
      )
    })
  }
  return runtimePromise
}

function storeValue(value) {
  if (!runtimeReady) {
    throw new NotebookMlUnsupportedError('WebGPU is required for browser JAX/Torch notebook cells')
  }
  const id = `ml-${nextValueId++}`
  values.set(id, value)
  return id
}

function getValue(id) {
  if (!values.has(id)) throw new Error(`unknown notebook ML value: ${id}`)
  return values.get(id)
}

function isObject(value) {
  return typeof value === 'object' && value !== null
}

function hasArrayShape(value) {
  return isObject(value) && Array.isArray(value.shape) && typeof value.dtype === 'string'
}

function refValue(value) {
  return hasArrayShape(value) && 'ref' in value ? value.ref : value
}

function dtypeOf(dtype) {
  if (dtype === undefined || dtype === null || dtype === '') return undefined
  const normalized = textOf(dtype)
  if (normalized === 'float32') return np.float32
  if (normalized === 'float64') return np.float64
  if (normalized === 'float16') return np.float16
  if (normalized === 'int32') return np.int32
  if (normalized === 'uint32') return np.uint32
  if (normalized === 'bool' || normalized === 'bool_') return np.bool
  return normalized
}

function dtypeOptions(dtype) {
  const normalized = dtypeOf(dtype)
  return normalized === undefined ? {} : { dtype: normalized }
}

function parseJson(value) {
  return typeof value === 'string' ? JSON.parse(value) : value
}

function leafFromDescriptor(value) {
  const descriptor = typeof value === 'string' ? JSON.parse(value) : value
  if (!descriptor || typeof descriptor !== 'object') return descriptor
  if (descriptor.kind === 'tensor') return refValue(getValue(descriptor.id))
  if (descriptor.kind === 'scalar') return descriptor.value
  if (descriptor.kind === 'none') return null
  if (descriptor.kind === 'list' || descriptor.kind === 'tuple') {
    return np.array(descriptor.items.map(item => scalarTreeFromDescriptor(item)))
  }
  return descriptor
}

function scalarTreeFromDescriptor(descriptor) {
  if (!descriptor || typeof descriptor !== 'object') return descriptor
  if (descriptor.kind === 'scalar') return descriptor.value
  if (descriptor.kind === 'none') return null
  if (descriptor.kind === 'list' || descriptor.kind === 'tuple') {
    return descriptor.items.map(item => scalarTreeFromDescriptor(item))
  }
  if (descriptor.kind === 'dict') {
    const out = {}
    for (const [key, value] of Object.entries(descriptor.items)) {
      out[key] = scalarTreeFromDescriptor(value)
    }
    return out
  }
  if (descriptor.kind === 'tensor') return getValue(descriptor.id).js()
  return descriptor
}

function treeFromDescriptor(descriptor) {
  if (!descriptor || typeof descriptor !== 'object') return descriptor
  if (descriptor.kind === 'tensor') return refValue(getValue(descriptor.id))
  if (descriptor.kind === 'scalar') return descriptor.value
  if (descriptor.kind === 'none') return null
  if (descriptor.kind === 'list' || descriptor.kind === 'tuple') {
    return descriptor.items.map(item => treeFromDescriptor(item))
  }
  if (descriptor.kind === 'dict') {
    const out = {}
    for (const [key, value] of Object.entries(descriptor.items)) {
      out[key] = treeFromDescriptor(value)
    }
    return out
  }
  return descriptor
}

function descriptorFromJsTree(tree) {
  if (hasArrayShape(tree)) return { kind: 'tensor', id: storeValue(tree) }
  if (Array.isArray(tree))
    return { kind: 'list', items: tree.map(item => descriptorFromJsTree(item)) }
  if (tree === null || tree === undefined) return { kind: 'none' }
  if (typeof tree === 'number' || typeof tree === 'boolean' || typeof tree === 'string') {
    return { kind: 'scalar', value: tree }
  }
  if (isObject(tree)) {
    const items = {}
    for (const [key, value] of Object.entries(tree)) {
      items[key] = descriptorFromJsTree(value)
    }
    return { kind: 'dict', items }
  }
  return { kind: 'scalar', value: textOf(tree) }
}

function describePython(value) {
  if (!describePythonTree) throw new Error('notebook ML Python descriptor is unavailable')
  return JSON.parse(textOf(describePythonTree(value)))
}

function wrapJsTree(tree) {
  if (!wrapPythonTree) throw new Error('notebook ML Python wrapper is unavailable')
  return wrapPythonTree(JSON.stringify(descriptorFromJsTree(tree)))
}

function pythonResultToJsTree(value) {
  return treeFromDescriptor(describePython(value))
}

function callPython(fn, args) {
  if (typeof fn === 'function') return fn(...args)
  if (fn && typeof fn.call === 'function') return fn.call(undefined, ...args)
  throw new Error('notebook ML Python callable is unavailable')
}

function pyTreeFunction(fn) {
  return (...args) =>
    pythonResultToJsTree(
      callPython(
        fn,
        args.map(arg => wrapJsTree(arg)),
      ),
    )
}

function transformOptions(optionsJson) {
  const raw = parseJson(optionsJson || '{}')
  const options = {}
  if (Array.isArray(raw.static_argnums)) options.staticArgnums = raw.static_argnums
  if (typeof raw.static_argnums === 'number') options.staticArgnums = [raw.static_argnums]
  if (raw.has_aux === true) options.hasAux = true
  if (typeof raw.argnums === 'number') options.argnums = raw.argnums
  if (Array.isArray(raw.argnums)) options.argnums = raw.argnums
  return options
}

function argArrayFromDescriptor(argsJson) {
  const descriptor = parseJson(argsJson)
  const tree = treeFromDescriptor(descriptor)
  return Array.isArray(tree) ? tree : [tree]
}

function callJitted(fn, argsJson, optionsJson) {
  let compiled = jittedFunctions.get(fn)
  if (!compiled) {
    compiled = jit(pyTreeFunction(fn), transformOptions(optionsJson))
    jittedFunctions.set(fn, compiled)
  }
  return JSON.stringify(descriptorFromJsTree(compiled(...argArrayFromDescriptor(argsJson))))
}

function callGrad(fn, argsJson, optionsJson) {
  const transformed = grad(pyTreeFunction(fn), transformOptions(optionsJson))
  return JSON.stringify(descriptorFromJsTree(transformed(...argArrayFromDescriptor(argsJson))))
}

function callValueAndGrad(fn, argsJson, optionsJson) {
  const transformed = valueAndGrad(pyTreeFunction(fn), transformOptions(optionsJson))
  return JSON.stringify(descriptorFromJsTree(transformed(...argArrayFromDescriptor(argsJson))))
}

function callMakeJaxpr(fn, argsJson) {
  const transformed = makeJaxpr(pyTreeFunction(fn))
  return textOf(transformed(...argArrayFromDescriptor(argsJson)).jaxpr)
}

function unaryOp(name, valueJson, optionsJson) {
  const value = leafFromDescriptor(valueJson)
  const options = parseJson(optionsJson || '{}')
  const ops = {
    astype: x => x.astype(dtypeOf(options.dtype)),
    block: x => {
      if (x && typeof x.dataSync === 'function') x.dataSync()
      return x
    },
    cos: x => np.cos(x),
    exp: x => np.exp(x),
    floor: x => np.floor(x),
    gelu: x => nn.gelu(x, { approximate: options.approximate !== false }),
    log: x => np.log(x),
    log_softmax: x => nn.logSoftmax(x, options.axis),
    neg: x => np.negative(x),
    relu: x => nn.relu(x),
    sigmoid: x => nn.sigmoid(x),
    sin: x => np.sin(x),
    softmax: x => nn.softmax(x, options.axis),
    sqrt: x => np.sqrt(x),
    squeeze: x => np.squeeze(x, options.axis),
    tanh: x => np.tanh(x),
    transpose: x => np.transpose(x, options.axes),
  }
  const op = ops[name]
  if (!op) throw new Error(`unsupported notebook ML unary op: ${name}`)
  return storeValue(op(value))
}

function binaryOp(name, leftJson, rightJson) {
  const left = leafFromDescriptor(leftJson)
  const right = leafFromDescriptor(rightJson)
  const ops = {
    add: np.add,
    div: np.trueDivide,
    eq: np.equal,
    ge: np.greaterEqual,
    gt: np.greater,
    le: np.lessEqual,
    lt: np.less,
    matmul: np.matmul,
    max: np.maximum,
    mul: np.multiply,
    ne: np.notEqual,
    pow: np.power,
    sub: np.subtract,
  }
  const op = ops[name]
  if (!op) throw new Error(`unsupported notebook ML binary op: ${name}`)
  return storeValue(op(left, right))
}

function reduceOp(name, valueJson, axisJson, keepdims) {
  const value = leafFromDescriptor(valueJson)
  const axis = parseJson(axisJson)
  const opts = keepdims ? { keepdims: true } : undefined
  if (name === 'mean') return storeValue(np.mean(value, axis, opts))
  if (name === 'sum') return storeValue(np.sum(value, axis, opts))
  throw new Error(`unsupported notebook ML reduce op: ${name}`)
}

function arrayFromJson(valuesJson, dtype) {
  const value = parseJson(valuesJson)
  return storeValue(np.array(value, dtypeOptions(dtype)))
}

function arrayFull(shapeJson, fillValueJson, dtype) {
  return storeValue(
    np.full(parseJson(shapeJson), leafFromDescriptor(fillValueJson), dtypeOptions(dtype)),
  )
}

function arrayZeros(shapeJson, dtype) {
  return storeValue(np.zeros(parseJson(shapeJson), dtypeOptions(dtype)))
}

function arrayOnes(shapeJson, dtype) {
  return storeValue(np.ones(parseJson(shapeJson), dtypeOptions(dtype)))
}

function arrayArange(start, stop, step, dtype) {
  return storeValue(
    np.arange(
      Number(start),
      stop === null || stop === undefined ? undefined : Number(stop),
      step === null || step === undefined ? undefined : Number(step),
      dtypeOptions(dtype),
    ),
  )
}

function arrayLike(name, valueJson, dtype) {
  const value = leafFromDescriptor(valueJson)
  if (name === 'zeros') return storeValue(np.zerosLike(value, dtypeOptions(dtype)))
  if (name === 'ones') return storeValue(np.onesLike(value, dtypeOptions(dtype)))
  throw new Error(`unsupported notebook ML like op: ${name}`)
}

function arrayShape(id) {
  return JSON.stringify(getValue(id).shape)
}

function arrayDType(id) {
  return textOf(getValue(id).dtype)
}

function arrayNDim(id) {
  return getValue(id).ndim
}

function arraySize(id) {
  return getValue(id).size
}

function arrayToJson(id) {
  return JSON.stringify(getValue(id).js())
}

function arrayItem(id) {
  return getValue(id).item()
}

function expandedIndexSpec(shape, spec) {
  const out = []
  let usedAxes = 0
  let sawEllipsis = false
  for (const part of spec) {
    if (part.kind === 'none') continue
    if (part.kind !== 'ellipsis') usedAxes += 1
  }
  for (const part of spec) {
    if (part.kind === 'ellipsis') {
      if (sawEllipsis) throw new Error('only one ellipsis is supported in notebook ML indexing')
      sawEllipsis = true
      const fill = shape.length - usedAxes
      for (let i = 0; i < fill; i++)
        out.push({ kind: 'slice', start: null, stop: null, step: null })
      continue
    }
    out.push(part)
  }
  let consumed = out.filter(part => part.kind !== 'none').length
  while (consumed < shape.length) {
    out.push({ kind: 'slice', start: null, stop: null, step: null })
    consumed += 1
  }
  return out
}

function sliceArgument(part) {
  if (part.kind === 'none') return null
  if (part.kind === 'int') return part.value
  if (part.kind === 'tensor') return refValue(getValue(part.id))
  if (part.kind === 'slice') {
    if (part.step !== null && part.step !== undefined && part.step !== 1) {
      throw new Error('strided slicing is only supported through .at[...].set(...)')
    }
    if (part.start === null && part.stop === null) return []
    if (part.stop === null) return [part.start]
    return [part.start ?? 0, part.stop]
  }
  throw new Error(`unsupported notebook ML index part: ${part.kind}`)
}

function arraySlice(id, specJson) {
  const value = getValue(id)
  const spec = expandedIndexSpec(value.shape, parseJson(specJson))
  return storeValue(value.ref.slice(...spec.map(sliceArgument)))
}

function rangeForSlice(part, size) {
  const step = part.step === null || part.step === undefined ? 1 : part.step
  const start =
    part.start === null || part.start === undefined ? (step < 0 ? size - 1 : 0) : part.start
  const stop = part.stop === null || part.stop === undefined ? (step < 0 ? -1 : size) : part.stop
  const normalizedStart = start < 0 ? size + start : start
  const normalizedStop =
    stop < 0 && part.stop !== null && part.stop !== undefined ? size + stop : stop
  const result = []
  if (step > 0) {
    for (let i = normalizedStart; i < normalizedStop; i += step) result.push(i)
  } else {
    for (let i = normalizedStart; i > normalizedStop; i += step) result.push(i)
  }
  return result
}

function selectedAxes(shape, spec) {
  const axes = []
  let axis = 0
  for (const part of expandedIndexSpec(shape, spec)) {
    if (part.kind === 'none') continue
    if (part.kind === 'int') {
      const index = part.value < 0 ? shape[axis] + part.value : part.value
      axes.push({ indices: [index], keep: false })
      axis += 1
      continue
    }
    if (part.kind !== 'slice') throw new Error('advanced .at[...].set(...) indexing is unavailable')
    axes.push({ indices: rangeForSlice(part, shape[axis]), keep: true })
    axis += 1
  }
  return axes
}

function valueAtSelection(value, keptCoords) {
  if (!Array.isArray(value)) return value
  let current = value
  for (const coord of keptCoords) {
    if (!Array.isArray(current)) return current
    current = current[coord]
  }
  return current
}

function setNested(root, coords, value) {
  let current = root
  for (let i = 0; i < coords.length - 1; i++) {
    current = current[coords[i]]
  }
  current[coords[coords.length - 1]] = value
}

function writeSelection(root, axes, replacement, axisIndex, coords, keptCoords) {
  if (axisIndex >= axes.length) {
    setNested(root, coords, valueAtSelection(replacement, keptCoords))
    return
  }
  const axis = axes[axisIndex]
  for (let i = 0; i < axis.indices.length; i++) {
    writeSelection(
      root,
      axes,
      replacement,
      axisIndex + 1,
      coords.concat(axis.indices[i]),
      axis.keep ? keptCoords.concat(i) : keptCoords,
    )
  }
}

function arrayAtSet(id, specJson, valueJson) {
  const target = getValue(id)
  const replacementValue = leafFromDescriptor(valueJson)
  const root = target.js()
  const replacement = hasArrayShape(replacementValue) ? replacementValue.js() : replacementValue
  writeSelection(root, selectedAxes(target.shape, parseJson(specJson)), replacement, 0, [], [])
  return storeValue(np.array(root, dtypeOptions(target.dtype)))
}

function arrayReshape(id, shapeJson) {
  return storeValue(getValue(id).ref.reshape(parseJson(shapeJson)))
}

function arraySplit(id, sections, axis) {
  return JSON.stringify(np.split(getValue(id).ref, parseJson(sections), axis).map(storeValue))
}

function arraySwapAxes(id, axis1, axis2) {
  return storeValue(np.swapaxes(getValue(id).ref, Number(axis1), Number(axis2)))
}

function arrayTake(id, indicesJson, axis) {
  return storeValue(np.take(getValue(id).ref, leafFromDescriptor(indicesJson), axis))
}

function arrayTril(id, k) {
  return storeValue(np.tril(getValue(id).ref, k === null || k === undefined ? 0 : Number(k)))
}

function arrayWhere(condJson, leftJson, rightJson) {
  return storeValue(
    np.where(
      leafFromDescriptor(condJson),
      leafFromDescriptor(leftJson),
      leafFromDescriptor(rightJson),
    ),
  )
}

function randomKey(seed) {
  return storeValue(random.key(Number(seed)))
}

function randomSplit(id, numJson) {
  return storeValue(random.split(getValue(id).ref, parseJson(numJson)))
}

function randomNormal(id, shapeJson, dtype) {
  const result = random.normal(getValue(id).ref, parseJson(shapeJson))
  return storeValue(dtype ? result.astype(dtypeOf(dtype)) : result)
}

function randomUniform(id, shapeJson, minval, maxval, dtype) {
  const result = random.uniform(getValue(id).ref, parseJson(shapeJson), {
    minval: Number(minval),
    maxval: Number(maxval),
  })
  return storeValue(dtype ? result.astype(dtypeOf(dtype)) : result)
}

function randomRandint(id, shapeJson, minval, maxval, dtype) {
  const uniform = random.uniform(getValue(id).ref, parseJson(shapeJson), {
    minval: Number(minval),
    maxval: Number(maxval),
  })
  return storeValue(np.floor(uniform).astype(dtypeOf(dtype || 'int32')))
}

function blockTree(argsJson) {
  const tree = treeFromDescriptor(parseJson(argsJson))
  const visit = value => {
    if (hasArrayShape(value)) {
      if (typeof value.dataSync === 'function') value.dataSync()
      return value
    }
    if (Array.isArray(value)) return value.map(visit)
    if (isObject(value)) {
      for (const nested of Object.values(value)) visit(nested)
    }
    return value
  }
  visit(tree)
  return JSON.stringify(descriptorFromJsTree(tree))
}

function devicePutTree(argsJson) {
  const tree = treeFromDescriptor(parseJson(argsJson))
  const placed = devicePut(tree, 'webgpu')
  if (placed && typeof placed.then === 'function') {
    throw new Error('asynchronous device_put is unavailable from synchronous notebook cells')
  }
  return JSON.stringify(descriptorFromJsTree(placed))
}

function devicesList() {
  return JSON.stringify(['webgpu'])
}

export function installNotebookMlBridge(target) {
  target.quartz_notebook_ml_register_callbacks = (wrapTreeCallback, describeTreeCallback) => {
    wrapPythonTree = wrapTreeCallback
    describePythonTree = describeTreeCallback
  }
  target.quartz_notebook_ml_array = arrayFromJson
  target.quartz_notebook_ml_full = arrayFull
  target.quartz_notebook_ml_zeros = arrayZeros
  target.quartz_notebook_ml_ones = arrayOnes
  target.quartz_notebook_ml_arange = arrayArange
  target.quartz_notebook_ml_like = arrayLike
  target.quartz_notebook_ml_shape = arrayShape
  target.quartz_notebook_ml_dtype = arrayDType
  target.quartz_notebook_ml_ndim = arrayNDim
  target.quartz_notebook_ml_size = arraySize
  target.quartz_notebook_ml_to_json = arrayToJson
  target.quartz_notebook_ml_item = arrayItem
  target.quartz_notebook_ml_unary = unaryOp
  target.quartz_notebook_ml_binary = binaryOp
  target.quartz_notebook_ml_reduce = reduceOp
  target.quartz_notebook_ml_slice = arraySlice
  target.quartz_notebook_ml_at_set = arrayAtSet
  target.quartz_notebook_ml_reshape = arrayReshape
  target.quartz_notebook_ml_split = arraySplit
  target.quartz_notebook_ml_swapaxes = arraySwapAxes
  target.quartz_notebook_ml_take = arrayTake
  target.quartz_notebook_ml_tril = arrayTril
  target.quartz_notebook_ml_where = arrayWhere
  target.quartz_notebook_ml_random_key = randomKey
  target.quartz_notebook_ml_random_split = randomSplit
  target.quartz_notebook_ml_random_normal = randomNormal
  target.quartz_notebook_ml_random_uniform = randomUniform
  target.quartz_notebook_ml_random_randint = randomRandint
  target.quartz_notebook_ml_jit = callJitted
  target.quartz_notebook_ml_grad = callGrad
  target.quartz_notebook_ml_value_and_grad = callValueAndGrad
  target.quartz_notebook_ml_make_jaxpr = callMakeJaxpr
  target.quartz_notebook_ml_block_until_ready = blockTree
  target.quartz_notebook_ml_device_put = devicePutTree
  target.quartz_notebook_ml_devices = devicesList
}
