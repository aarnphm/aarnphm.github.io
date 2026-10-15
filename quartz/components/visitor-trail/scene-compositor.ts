import { subscribeVisitorDaylight, type VisitorDaylight } from '../../util/visitor-daylight'
import { daylightTint, type DaylightTint } from './daylight-palette'
import { groundObjectSelector } from './depth-order'
import { trailOpeningCell, trailThemeDuration, type SceneReveal } from './scene-reveal'

// The SVG stays the scene model: journeys, depth order, the pet and camp hints still read and
// write it. This module rasterizes that model into textures and composites them with WebGL2, so
// Blink lays the SVG out but never paints, layerizes or rasterizes its eight thousand nodes.

type Matrix = [number, number, number, number, number, number]
type Layer = {
  texture: WebGLTexture | null
  width: number
  height: number
  // Maps the owner's user space onto texels at raster time.
  base: Matrix
  scale: number
  palette: number
  revision: number
  dirty: boolean
  // When it first needed a repaint; the oldest repaint goes first, so a busy texture cannot starve the rest.
  waiting: number
}
type Run = Layer & {
  kind: 'run'
  owner: Group
  nodes: Element[]
  box: Box | null
  hits: number[]
  hot: Set<Element>
  placed: Float32Array | null
}
type Surface = {
  kind: 'surface'
  frame: SVGForeignObjectElement
  canvas: HTMLCanvasElement
  style: CSSStyleDeclaration
  offset: Matrix
  shown: boolean
  placed: Float32Array | null
  texture: WebGLTexture | null
  dirty: boolean
}
type Group = {
  kind: 'group'
  element: SVGGraphicsElement
  items: (Run | Surface | Group)[]
  structure: boolean
  style: CSSStyleDeclaration
  mask: (Layer & { source: SVGMaskElement; placed: Float32Array | null }) | null
  clip: SVGClipPathElement | null
  // Cached between the mutations, animations and resizes that can change them.
  moved: boolean
  fresh: boolean
  matrix: Matrix
  screen: Matrix
  opacity: number
  shown: boolean
}
type Clip = { owner: Group; source: SVGClipPathElement }
type Box = { left: number; top: number; right: number; bottom: number }

const identity: Matrix = [1, 0, 0, 1, 0, 0]
const multiply = (m: Matrix, n: Matrix): Matrix => [
  m[0] * n[0] + m[2] * n[1],
  m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3],
  m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4],
  m[1] * n[4] + m[3] * n[5] + m[5],
]
const invert = (m: Matrix): Matrix => {
  const det = m[0] * m[3] - m[1] * m[2] || 1e-9
  return [
    m[3] / det,
    -m[1] / det,
    -m[2] / det,
    m[0] / det,
    (m[2] * m[5] - m[3] * m[4]) / det,
    (m[1] * m[4] - m[0] * m[5]) / det,
  ]
}
const scaleOf = (m: Matrix) => Math.sqrt(Math.abs(m[0] * m[3] - m[1] * m[2])) || 1
const scaling = (x: number, y = x): Matrix => [x, 0, 0, y, 0, 0]
const translation = (x: number, y: number): Matrix => [1, 0, 0, 1, x, y]

// Strokes and glyphs reach this far past the geometry box, in user units.
const padding = 3
const ambientLayers =
  '.trail-weather, .trail-hiker-tracks, .trail-reader-tracks, .trail-camp-tracks, .trail-visitor-tracks, .trail-ranger-tracks'
const motionProperties = new Set(['opacity', 'transform', 'translate', 'rotate', 'scale'])
const motionVariables = new Set([
  '--trail-tree-wind-scale',
  '--trail-weather-delay',
  '--trail-wind-period',
  '--trail-rain-strength',
  '--trail-wind-strength',
  '--trail-gust-strength',
  '--trail-wind-x',
  '--trail-wind-y',
  '--trail-rain-x',
  '--trail-leaf-push-x',
  '--trail-leaf-push-y',
  '--trail-leaf-rest-x',
  '--trail-leaf-rest-y',
  '--trail-leaf-flip',
  '--trail-leaf-lag-x',
  '--trail-leaf-lag-y',
  '--trail-leaf-travel-x',
  '--trail-leaf-travel-y',
])
const unpainted = new Set([
  'defs',
  'mask',
  'clipPath',
  'linearGradient',
  'radialGradient',
  'pattern',
  'marker',
  'symbol',
  'title',
  'desc',
  'metadata',
  'style',
  'script',
])
const shapes = new Set(['path', 'ellipse', 'circle', 'rect', 'line', 'polyline', 'polygon'])
const reference = (value: string | null) => value?.match(/url\(["']?#([^"')]+)["']?\)/)?.[1]

const vertexShader = `#version 300 es
in vec2 corner;
// The drawing target in canvas pixels: the canvas itself, or a cached segment within it.
uniform vec4 target;
uniform mat3 place;
flat out mat3 unplace;
void main() {
  vec2 point = (place * vec3(corner, 1.0)).xy;
  unplace = inverse(place);
  vec2 local = (point - target.xy) / target.zw;
  gl_Position = vec4(local.x * 2.0 - 1.0, 1.0 - local.y * 2.0, 0.0, 1.0);
}`
const fragmentShader = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform sampler2D maskImage;
uniform sampler2D clipImage;
uniform float alpha;
uniform vec2 viewport;
uniform vec3 reveal;
uniform vec3 sweep;
uniform vec4 themeReveal;
uniform sampler2D loadCoverage;
uniform float loadActive;
uniform vec3 solarTint;
uniform vec4 target;
uniform mat3 maskPlace;
flat in mat3 unplace;
out vec4 color;
vec2 clipUv;
float trailAlpha(float progress, float cellSize) {
  if (progress <= 0.0) return 0.0;
  if (progress >= 1.0) return 1.0;
  vec2 point = (floor(clipUv * viewport / cellSize) + 0.5) * cellSize;
  return step(distance(point, sweep.xy), progress * (sweep.z + cellSize));
}
void main() {
  // Sample from the pixel position, not interpolated coordinates, so a quad and a region
  // drawn over it pick the same nearest texels.
  vec2 point = vec2(target.x + gl_FragCoord.x, target.y + target.w - gl_FragCoord.y);
  vec2 uv = (unplace * vec3(point, 1.0)).xy;
  // Region draws cover a rect around the quad; only the quad itself paints.
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
  vec2 maskUv = (maskPlace * vec3(point, 1.0)).xy;
  clipUv = point / viewport;
  // Masks hold premultiplied luminance, so this is luminance times alpha, as in SVG.
  float mask = texture(maskImage, maskUv).r;
  color = texture(image, uv) * alpha * mask * texture(clipImage, clipUv).r;
  // Textures and the blend target hold premultiplied RGB.
  color.rgb = min(color.rgb * solarTint, vec3(color.a));
  float ready = 1.0;
  if (loadActive > 0.0 && themeReveal.x >= 0.0) {
    vec2 cell = floor(clipUv * viewport / themeReveal.w);
    ready = texture(loadCoverage, (cell + 0.5) / vec2(textureSize(loadCoverage, 0))).r;
  }
  if (themeReveal.x > 0.0) {
    float next = trailAlpha(themeReveal.y, themeReveal.w) * ready;
    color *= themeReveal.x < 1.5 ? next : 1.0 - next;
  } else color *= ready;
  if (reveal.x > 0.0) {
    vec2 cell = floor(clipUv * viewport / reveal.z);
    float noise = fract(dot(cell, vec2(0.754877666, 0.569840296)));
    float threshold = 0.7 * (1.0 - clipUv.y) + 0.3 * noise;
    float next = step(threshold, reveal.y);
    color *= reveal.x < 1.5 ? next : 1.0 - next;
  }
}`

type Classes = {
  CSSTransition: CSSTransition
  Element: Element
  KeyframeEffect: KeyframeEffect
  SVGAElement: SVGAElement
  SVGClipPathElement: SVGClipPathElement
  SVGElement: SVGElement
  SVGForeignObjectElement: SVGForeignObjectElement
  SVGGElement: SVGGElement
  SVGGeometryElement: SVGGeometryElement
  SVGGradientElement: SVGGradientElement
  SVGGraphicsElement: SVGGraphicsElement
  SVGMaskElement: SVGMaskElement
  SVGRadialGradientElement: SVGRadialGradientElement
  SVGSVGElement: SVGSVGElement
  SVGTextElement: SVGTextElement
}

export function mountSceneCompositor(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  opening: SceneReveal,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const svg = scene.querySelector<SVGSVGElement>('svg.trail-canvas')
  if (!view || !svg) throw new Error('The scene compositor needs the trail canvas')
  let resolution = Math.max(1, view.devicePixelRatio)
  const canvas = document.createElement('canvas')
  canvas.className = 'trail-compositor'
  canvas.setAttribute('aria-hidden', 'true')
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    premultipliedAlpha: true,
    // Deferred uploads keep the last complete frame through browser presentation.
    preserveDrawingBuffer: true,
    antialias: false,
    depth: false,
    stencil: false,
  })
  if (!gl) return false
  svg.after(canvas)
  const scratch = document.createElement('canvas')
  // Reuse CPU staging across uploads, then shed a large scene repaint's allocation once
  // recent paints only need small sprites. Durable pixels live in the WebGL textures.
  const paint = scratch.getContext('2d', { willReadFrequently: true })
  if (!paint) {
    gl.getExtension('WEBGL_lose_context')?.loseContext()
    canvas.remove()
    return false
  }
  let rasterWidth = 1
  let rasterHeight = 1
  let rasterSince = view.performance.now()
  const trimRaster = (release = false) => {
    const now = view.performance.now()
    if (!release && now - rasterSince < 1000) return
    const width = release ? 1 : rasterWidth
    const height = release ? 1 : rasterHeight
    if (release || scratch.width * scratch.height > 2 * width * height) {
      scratch.width = width
      scratch.height = height
    }
    rasterWidth = 1
    rasterHeight = 1
    rasterSince = now
  }
  const clearRaster = (width: number, height: number) => {
    const nextWidth = Math.ceil(width / 256) * 256
    const nextHeight = Math.ceil(height / 256) * 256
    rasterWidth = Math.max(rasterWidth, nextWidth)
    rasterHeight = Math.max(rasterHeight, nextHeight)
    if (scratch.width < width) scratch.width = nextWidth
    if (scratch.height < height) scratch.height = nextHeight
    paint.setTransform(1, 0, 0, 1, 0, 0)
    paint.clearRect(0, 0, width, height)
  }

  // The trail frame builds nodes in the page document, and some of them carry the frame's classes,
  // so a type test checks both realms.
  const realms = [view, globalThis as unknown as typeof view]
  const is = <K extends keyof Classes>(value: unknown, name: K): value is Classes[K] =>
    realms.some(realm => value instanceof (realm[name] as unknown as typeof Object))
  // The scene lives in a shadow root, so its ids and stylesheets belong to that tree.
  const tree = () => scene.getRootNode() as Document | ShadowRoot
  // Attributes that no selector reads cannot change a pixel. Journeys write dozens of them each frame.
  // The scene stylesheets load before the first render, so selectors are read then.
  let attributeSelectors: Map<string, string[]> | null = null
  const collect = (rules: CSSRuleList, into: Map<string, string[]>) => {
    for (const rule of rules) {
      if ('selectorText' in rule && typeof rule.selectorText === 'string')
        for (const name of new Set(
          Array.from(rule.selectorText.matchAll(/\[([\w-]+)/g), m => m[1]),
        ))
          into.set(name, [...(into.get(name) ?? []), rule.selectorText])
      if ('cssRules' in rule && rule.cssRules) collect(rule.cssRules as CSSRuleList, into)
    }
  }
  const readSelectors = () => {
    const found = new Map<string, string[]>()
    for (const sheet of tree().styleSheets) {
      try {
        collect(sheet.cssRules, found)
      } catch {
        // A cross-origin sheet cannot be read; its selectors do not target the trail.
      }
    }
    found.delete('data-trail-renderer')
    found.delete('data-trail-opening')
    found.delete('data-trail-rendered')
    found.delete('data-trail-visible')
    attributeSelectors = found
  }
  const inert = (name: string) =>
    attributeSelectors !== null && name.startsWith('data-') && !attributeSelectors.has(name)

  let program: WebGLProgram
  let vertexBuffer: WebGLBuffer | null = null
  let uniforms: Record<string, WebGLUniformLocation | null>
  let blank: WebGLTexture | null
  let maxTexture = 4096
  // One screen-space coverage texture per chain of clipped ancestors.
  let clipVersion = 0
  const clipTextures = new Map<string, { texture: WebGLTexture | null; key: string }>()
  const textures = new Set<WebGLTexture>()
  const contents = new WeakMap<WebGLTexture, number>()
  const allocations = new WeakMap<
    WebGLTexture,
    { width: number; height: number; single: boolean }
  >()
  const setup = () => {
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) throw new Error('WebGL could not create a shader')
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(shader) ?? 'WebGL shader failed')
      return shader
    }
    const next = gl.createProgram()
    const vertex = compile(gl.VERTEX_SHADER, vertexShader)
    const fragment = compile(gl.FRAGMENT_SHADER, fragmentShader)
    gl.attachShader(next, vertex)
    gl.attachShader(next, fragment)
    gl.bindAttribLocation(next, 0, 'corner')
    gl.linkProgram(next)
    gl.deleteShader(vertex)
    gl.deleteShader(fragment)
    if (!gl.getProgramParameter(next, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(next) ?? 'WebGL program failed')
    program = next
    gl.useProgram(program)
    uniforms = Object.fromEntries(
      [
        'viewport',
        'target',
        'place',
        'maskPlace',
        'alpha',
        'image',
        'maskImage',
        'clipImage',
        'reveal',
        'sweep',
        'themeReveal',
        'loadCoverage',
        'loadActive',
        'solarTint',
      ].map(name => [name, gl.getUniformLocation(program, name)]),
    )
    gl.uniform1i(uniforms.image, 0)
    gl.uniform1i(uniforms.maskImage, 1)
    gl.uniform1i(uniforms.clipImage, 2)
    gl.uniform1i(uniforms.loadCoverage, 3)
    gl.uniform3f(uniforms.solarTint, 1, 1, 1)
    vertexBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    blank = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, blank)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]),
    )
    filter()
    maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
    clipTextures.clear()
  }
  const filter = () => {
    const sampling = gl.NEAREST
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, sampling)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, sampling)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  }
  const upload = (
    texture: WebGLTexture | null,
    source: HTMLCanvasElement,
    single = false,
    width = source.width,
    height = source.height,
  ) => {
    const target = texture ?? gl.createTexture()
    if (!target) return null
    textures.add(target)
    gl.bindTexture(gl.TEXTURE_2D, target)
    const known = allocations.get(target)
    const format = single ? gl.RED : gl.RGBA
    if (known?.width === width && known.height === height && known.single === single)
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, format, gl.UNSIGNED_BYTE, source)
    else {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        single ? gl.R8 : gl.RGBA,
        width,
        height,
        0,
        format,
        gl.UNSIGNED_BYTE,
        source,
      )
      allocations.set(target, { width, height, single })
      filter()
    }
    // A cached segment holding this texture's previous pixels must be rebuilt.
    contents.set(target, (contents.get(target) ?? 0) + 1)
    return target
  }
  const release = (texture: WebGLTexture | null) => {
    if (!texture) return
    gl.deleteTexture(texture)
    textures.delete(texture)
    allocations.delete(texture)
  }

  const ctm = (element: SVGGraphicsElement): Matrix => {
    const matrix = element.getCTM()
    return matrix ? [matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f] : identity
  }

  // Elements whose own transform, opacity or position changes keep their own textures. A fade
  // keeps its element apart only while it runs, so the opening does not leave the scene in shards.
  const promoted = new Map<Element, 'shape' | 'motion' | 'hot'>()
  const fading = new WeakMap<Element, number>()
  let carriers = new WeakSet<Element>()
  const groups = new Map<Element, Group>()
  const runOf = new WeakMap<Element, Run>()
  const carry = (element: Element) => {
    for (
      let parent: Element | null = element.parentElement;
      parent && parent !== svg;
      parent = parent.parentElement
    )
      carriers.add(parent)
  }
  const promote = (element: Element, reason: 'shape' | 'motion' | 'hot' = 'shape') => {
    if (reason === 'motion') fading.set(element, clock)
    const known = promoted.get(element)
    if (known) {
      if (known === 'motion' && reason !== 'motion') promoted.set(element, reason)
      return
    }
    if (element === svg || !svg.contains(element)) return
    if (element.closest('defs, mask, clipPath')) return
    promoted.set(element, reason)
    carry(element)
    let owner = element.parentElement
    while (owner && !groups.has(owner)) owner = owner.parentElement
    const group = owner && groups.get(owner)
    if (group) group.structure = true
  }
  const discover = (root: Element) => {
    // Depth sorting moves known objects every frame; only new markup needs a search.
    if (!is(root, 'SVGElement')) return
    if (promoted.has(root)) {
      carry(root)
      let owner = root.parentElement
      while (owner && !groups.has(owner)) owner = owner.parentElement
      const group = owner && groups.get(owner)
      if (group) group.structure = true
      return
    }
    if (
      root.matches(`${groundObjectSelector}, .trail-tree-canopy`) ||
      root.hasAttribute('data-trail-depth') ||
      root.hasAttribute('data-trail-chunk') ||
      root.hasAttribute('data-trail-layer')
    )
      promote(root)
    if (root.hasAttribute('mask') || is(root, 'SVGForeignObjectElement')) promote(root)
    for (const element of root.querySelectorAll(
      `${groundObjectSelector}, .trail-tree-canopy, [data-trail-depth], [data-trail-chunk], [data-trail-layer], [mask], foreignObject`,
    ))
      promote(element)
  }

  // A node's geometry box in the user space that `toSpace` maps the viewport into.
  let bboxes = new WeakMap<Element, DOMRect | null>()
  const invalidateBox = (element: Element) => {
    for (let node: Element | null = element; node && node !== svg; node = node.parentElement)
      bboxes.delete(node)
  }
  const localBox = (node: SVGGraphicsElement): DOMRect | null => {
    const cached = bboxes.get(node)
    if (cached !== undefined) return cached
    let box: DOMRect | null = null
    try {
      box = node.getBBox()
    } catch {
      box = null
    }
    if (box && (box.width || box.height)) bboxes.set(node, box)
    return box
  }
  const boxOf = (node: Element, toSpace: Matrix, self = false): Box | null => {
    if (!is(node, 'SVGGraphicsElement')) return null
    const box = localBox(node)
    if (!box || (!box.width && !box.height)) return null
    const matrix = self ? identity : multiply(toSpace, ctm(node))
    let left = Infinity
    let top = Infinity
    let right = -Infinity
    let bottom = -Infinity
    for (const [x, y] of [
      [box.x, box.y],
      [box.x + box.width, box.y],
      [box.x, box.y + box.height],
      [box.x + box.width, box.y + box.height],
    ]) {
      const px = matrix[0] * x + matrix[2] * y + matrix[4]
      const py = matrix[1] * x + matrix[3] * y + matrix[5]
      left = Math.min(left, px)
      top = Math.min(top, py)
      right = Math.max(right, px)
      bottom = Math.max(bottom, py)
    }
    return { left, top, right, bottom }
  }
  const group = (element: SVGGraphicsElement): Group => {
    const known = groups.get(element)
    if (known) return known
    const next: Group = {
      kind: 'group',
      element,
      items: [],
      structure: true,
      style: view.getComputedStyle(element),
      mask: null,
      clip: null,
      moved: true,
      fresh: false,
      matrix: identity,
      screen: identity,
      opacity: 1,
      shown: true,
    }
    groups.set(element, next)
    return next
  }
  const layer = (): Layer => ({
    texture: null,
    width: 0,
    height: 0,
    base: identity,
    scale: 0,
    palette: -1,
    revision: -1,
    dirty: true,
    waiting: 0,
  })
  const drop = (item: Run | Surface | Group) => {
    if (item.kind === 'group') {
      // A retired container can still reference descendants moved into another layer.
      if (
        svg.contains(item.element) &&
        (promoted.has(item.element) || carriers.has(item.element))
      ) {
        item.moved = true
        item.fresh = false
        return
      }
      for (const child of item.items) drop(child)
      release(item.mask?.texture ?? null)
      groups.delete(item.element)
    } else release(item.texture)
  }
  const partition = (owner: Group) => {
    owner.structure = false
    const previous = owner.items
    const reuse = new Map<Element, Run>()
    for (const item of previous) if (item.kind === 'run') reuse.set(item.nodes[0], item)
    const kept = new Set<Run | Surface | Group>()
    const items: (Run | Surface | Group)[] = []
    let pending: Element[] = []
    let extent: Box | null = null
    let covered = 0
    const flush = () => {
      const box = extent
      extent = null
      covered = 0
      if (!pending.length) return
      const known = reuse.get(pending[0])
      const same =
        known &&
        known.nodes.length === pending.length &&
        known.nodes.every((node, index) => node === pending[index])
      const run: Run = same
        ? known
        : {
            kind: 'run',
            owner,
            nodes: pending,
            box,
            hits: [],
            hot: new Set(),
            placed: null,
            ...layer(),
          }
      if (same && box) run.box = box
      for (const node of pending) runOf.set(node, run)
      items.push(run)
      kept.add(run)
      pending = []
    }
    const element = owner.element
    const leaf = !(
      is(element, 'SVGGElement') ||
      is(element, 'SVGSVGElement') ||
      is(element, 'SVGAElement')
    )
    const children = leaf ? [element] : [...element.children]
    const space = leaf ? identity : invert(ctm(element))
    for (const child of children) {
      if (
        child !== element &&
        (unpainted.has(child.localName) || child.hasAttribute('data-trail-model'))
      )
        continue
      if (child !== element && is(child, 'SVGForeignObjectElement')) {
        flush()
        const surface = child.querySelector('canvas')
        if (!surface) continue
        const known = previous.find(
          (item): item is Surface => item.kind === 'surface' && item.frame === child,
        )
        const shift = view.getComputedStyle(surface).transform
        const offset = new view.DOMMatrixReadOnly(shift === 'none' ? undefined : shift)
        const item: Surface = known ?? {
          kind: 'surface',
          frame: child,
          canvas: surface,
          style: view.getComputedStyle(child),
          offset: translation(child.x.baseVal.value + offset.e, child.y.baseVal.value + offset.f),
          shown: true,
          placed: null,
          texture: null,
          dirty: true,
        }
        items.push(item)
        kept.add(item)
        continue
      }
      if (child !== element && (promoted.has(child) || carriers.has(child))) {
        flush()
        const item = group(child as SVGGraphicsElement)
        items.push(item)
        kept.add(item)
        continue
      }
      // Siblings share a texture while they overlap; far-apart ones would leave it mostly empty.
      const box = leaf ? null : boxOf(child, space)
      if (box) {
        const area = (box.right - box.left) * (box.bottom - box.top)
        const union: Box = extent
          ? {
              left: Math.min(extent.left, box.left),
              top: Math.min(extent.top, box.top),
              right: Math.max(extent.right, box.right),
              bottom: Math.max(extent.bottom, box.bottom),
            }
          : box
        if (
          extent &&
          (union.right - union.left) * (union.bottom - union.top) > 2 * (covered + area) + 4096
        )
          flush()
        extent = extent ? union : box
        covered += area
      }
      pending.push(child)
    }
    flush()
    for (const item of previous) {
      if (kept.has(item)) continue
      drop(item)
    }
    owner.items = items
    const clipId = element === svg ? undefined : reference(owner.style.clipPath)
    const clip = clipId ? tree().getElementById(clipId) : null
    owner.clip = is(clip, 'SVGClipPathElement') ? clip : null
    owner.fresh = false
    const maskId =
      element === svg ? undefined : reference(owner.style.mask === 'none' ? null : owner.style.mask)
    const source = maskId ? tree().getElementById(maskId) : null
    if (is(source, 'SVGMaskElement')) {
      if (owner.mask?.source !== source) {
        release(owner.mask?.texture ?? null)
        owner.mask = { ...layer(), source, placed: null }
      }
    } else if (owner.mask) {
      release(owner.mask.texture)
      owner.mask = null
    }
  }
  const root = group(svg)

  // The run that paints an element: its ancestor whose parent keeps its own textures.
  const ownerOf = (element: Element) => {
    for (let node: Element | null = element; node && node !== svg; node = node.parentElement) {
      const parent = node.parentElement
      if (parent && groups.has(parent)) return { run: runOf.get(node), top: node }
    }
    return { run: undefined, top: null }
  }
  const touch = (element: Element) => {
    invalidateBox(element)
    const group = groups.get(element)
    if (group) {
      dirtyGroup(group)
      return
    }
    const { run, top } = ownerOf(element)
    if (!run || !top) return
    run.dirty = true
    run.box = null
    run.hot.add(top)
    if (run.hits.at(-1) !== clock) run.hits.push(clock)
  }
  const dirtyGroup = (group: Group) => {
    invalidateBox(group.element)
    group.moved = true
    group.fresh = false
    if (group.mask) group.mask.dirty = true
    for (const item of group.items) {
      if (item.kind === 'group') dirtyGroup(item)
      else {
        item.dirty = true
        if (item.kind === 'run') item.box = null
      }
    }
  }
  const dirtyMask = (mask: Element) => {
    for (const group of groups.values()) if (group.mask?.source === mask) group.mask.dirty = true
  }
  const clipVersions = new WeakMap<SVGClipPathElement, number>()
  const touchDefinition = (element: Element) => {
    const mask = element.closest('mask')
    if (mask) return dirtyMask(mask)
    const clip = element.closest('clipPath')
    if (clip) {
      clipVersions.set(clip, (clipVersions.get(clip) ?? 0) + 1)
      return
    }
    if (element.closest('linearGradient, radialGradient')) return dirtyGroup(root)
    const id = element.closest('[id]')?.id
    if (id)
      for (const path of svg.querySelectorAll(`textPath[href="#${CSS.escape(id)}"]`)) touch(path)
  }

  // Raster: draws an SVG subtree into the scratch canvas from computed styles.
  const paths = new WeakMap<Element, { d: string; grow: number; path: Path2D }>()
  // Pixel art is one square per cell, `M x y h s v s h -s Z`. The canvas antialiases every edge,
  // and cells of different tones abut on a dither grid at fractional texels, so each shared edge
  // left a blended line and some let the layer behind through. Each cell grows by one texel to
  // the right and below, under the cells drawn after it, so solid paint covers every shared edge.
  // A stroke in the fill's paint did the same and cost half as much again per raster.
  const cells = (d: string, grow: number) => {
    if (!d.startsWith('M') || !d.endsWith('Z')) return new view.Path2D(d)
    const firstH = d.indexOf('h', 1)
    const firstV = firstH > 0 ? d.indexOf('v', firstH + 1) : -1
    if (firstV < 0) return new view.Path2D(d)
    const step = d.slice(firstH + 1, firstV)
    const stepNum = Number(step)
    if (!step || !Number.isFinite(stepNum)) return new view.Path2D(d)
    const suffix = `h${step}v${step}h-${step}Z`
    const size = stepNum + grow
    const path = new view.Path2D()
    let index = 0
    while (index < d.length) {
      if (d.charCodeAt(index) !== 77) return new view.Path2D(d)
      const space = d.indexOf(' ', index + 1)
      if (space < 0) return new view.Path2D(d)
      const h = d.indexOf('h', space + 1)
      if (h < 0 || !d.startsWith(suffix, h)) return new view.Path2D(d)
      const x = Number(d.slice(index + 1, space))
      const y = Number(d.slice(space + 1, h))
      if (!Number.isFinite(x) || !Number.isFinite(y)) return new view.Path2D(d)
      path.rect(x, y, size, size)
      index = h + suffix.length
    }
    return path
  }
  const geometry = (element: SVGGraphicsElement, style: CSSStyleDeclaration, grow = 0) => {
    const number = (value: string) => Number.parseFloat(value) || 0
    const path = new view.Path2D()
    switch (element.localName) {
      case 'path': {
        const d = element.getAttribute('d') ?? ''
        const known = paths.get(element)
        if (known?.d === d && known.grow === grow) return known.path
        const next = grow ? cells(d, grow) : new view.Path2D(d)
        paths.set(element, { d, grow, path: next })
        return next
      }
      case 'ellipse': {
        const rx = number(style.rx)
        const ry = style.ry === 'auto' ? rx : number(style.ry)
        path.ellipse(number(style.cx), number(style.cy), rx, ry, 0, 0, Math.PI * 2)
        return path
      }
      case 'circle':
        path.arc(number(style.cx), number(style.cy), number(style.r), 0, Math.PI * 2)
        return path
      case 'rect': {
        const width = number(style.width)
        const height = number(style.height)
        const rx = Math.min(width / 2, number(style.rx === 'auto' ? style.ry : style.rx))
        const ry = Math.min(height / 2, number(style.ry === 'auto' ? style.rx : style.ry))
        if (rx || ry)
          path.roundRect(number(style.x), number(style.y), width, height, [{ x: rx, y: ry }])
        else path.rect(number(style.x), number(style.y), width, height)
        return path
      }
      case 'line': {
        const line = element as SVGLineElement
        path.moveTo(line.x1.baseVal.value, line.y1.baseVal.value)
        path.lineTo(line.x2.baseVal.value, line.y2.baseVal.value)
        return path
      }
      default: {
        const points = [...(element as SVGPolylineElement).points]
        points.forEach((point, index) =>
          index ? path.lineTo(point.x, point.y) : path.moveTo(point.x, point.y),
        )
        if (element.localName === 'polygon') path.closePath()
        return path
      }
    }
  }
  const paintFor = (
    value: string,
    element: SVGGraphicsElement,
    shape: Path2D,
  ): { style: string | CanvasGradient; shape: Path2D; box: Matrix } | null => {
    const id = reference(value)
    if (!id) return value === 'none' ? null : { style: value, shape, box: identity }
    const source = tree().getElementById(id)
    if (!is(source, 'SVGGradientElement')) {
      const fallback = value.replace(/url\([^)]*\)/, '').trim()
      return fallback && fallback !== 'none' ? { style: fallback, shape, box: identity } : null
    }
    const bounding = source.getAttribute('gradientUnits') !== 'userSpaceOnUse'
    let box = identity
    let local = shape
    if (bounding) {
      const bounds = localBox(element)
      if (!bounds || !bounds.width || !bounds.height) return null
      box = [bounds.width, 0, 0, bounds.height, bounds.x, bounds.y]
      local = new view.Path2D()
      const inverse = invert(box)
      local.addPath(shape, {
        a: inverse[0],
        b: inverse[1],
        c: inverse[2],
        d: inverse[3],
        e: inverse[4],
        f: inverse[5],
      })
    }
    const length = (name: string, fallback: string) => {
      const raw = source.getAttribute(name) ?? fallback
      const value = Number.parseFloat(raw)
      if (!raw.endsWith('%')) return value
      return bounding
        ? value / 100
        : (value / 100) *
            (name.includes('y') ? svg.viewBox.baseVal.height : svg.viewBox.baseVal.width)
    }
    const gradient = is(source, 'SVGRadialGradientElement')
      ? paint.createRadialGradient(
          length('fx', source.getAttribute('cx') ?? '50%'),
          length('fy', source.getAttribute('cy') ?? '50%'),
          0,
          length('cx', '50%'),
          length('cy', '50%'),
          length('r', '50%'),
        )
      : paint.createLinearGradient(
          length('x1', '0%'),
          length('y1', '0%'),
          length('x2', '100%'),
          length('y2', '0%'),
        )
    let offset = 0
    for (const stop of source.querySelectorAll('stop')) {
      const raw = stop.getAttribute('offset') ?? '0'
      offset = Math.min(
        1,
        Math.max(offset, Number.parseFloat(raw) / (raw.endsWith('%') ? 100 : 1) || 0),
      )
      const stopStyle = view.getComputedStyle(stop)
      const opacity = Number(stopStyle.stopOpacity)
      const color = stopStyle.stopColor
      gradient.addColorStop(
        offset,
        opacity < 1 ? `color-mix(in srgb, ${color} ${opacity * 100}%, transparent)` : color,
      )
    }
    return { style: gradient, shape: local, box }
  }
  const dashes = (element: SVGGraphicsElement, style: CSSStyleDeclaration) => {
    if (style.strokeDasharray === 'none') return { list: [] as number[], offset: 0 }
    const length = Number(element.getAttribute('pathLength'))
    const factor =
      length > 0 && is(element, 'SVGGeometryElement') ? element.getTotalLength() / length : 1
    const list = style.strokeDasharray
      .split(/[\s,]+/)
      .map(part => (Number.parseFloat(part) || 0) * factor)
    return {
      list: list.length % 2 ? [...list, ...list] : list,
      offset: (Number.parseFloat(style.strokeDashoffset) || 0) * factor,
    }
  }
  const draw = (
    matrix: Matrix,
    alpha: number,
    style: CSSStyleDeclaration,
    element: SVGGraphicsElement,
    shape: Path2D,
  ) => {
    let painted = false
    const fill = paintFor(style.fill, element, shape)
    if (fill) {
      const opacity = alpha * Number(style.fillOpacity)
      if (opacity > 0) {
        paint.setTransform(...multiply(matrix, fill.box))
        paint.globalAlpha = opacity
        paint.fillStyle = fill.style
        paint.fill(fill.shape, style.fillRule === 'evenodd' ? 'evenodd' : 'nonzero')
        painted = true
      }
    }
    const width = Number.parseFloat(style.strokeWidth) || 0
    const stroke = width > 0 ? paintFor(style.stroke, element, shape) : null
    if (stroke) {
      const opacity = alpha * Number(style.strokeOpacity)
      if (opacity > 0) {
        paint.setTransform(...multiply(matrix, stroke.box))
        paint.globalAlpha = opacity
        paint.strokeStyle = stroke.style
        const fixed = style.vectorEffect === 'non-scaling-stroke'
        paint.lineWidth =
          (fixed ? (width * resolution) / scaleOf(matrix) : width) / scaleOf(stroke.box)
        paint.lineCap = style.strokeLinecap as CanvasLineCap
        paint.lineJoin = (
          style.strokeLinejoin === 'arcs' || style.strokeLinejoin === 'miter-clip'
            ? 'miter'
            : style.strokeLinejoin
        ) as CanvasLineJoin
        paint.miterLimit = Number(style.strokeMiterlimit) || 4
        const dash = dashes(element, style)
        paint.setLineDash(dash.list)
        paint.lineDashOffset = dash.offset
        paint.stroke(stroke.shape)
        paint.setLineDash([])
        painted = true
      }
    }
    return painted
  }
  const write = (
    element: SVGTextElement,
    style: CSSStyleDeclaration,
    matrix: Matrix,
    alpha: number,
  ) => {
    const along = element.querySelector('textPath')
    const textStyle = along ? view.getComputedStyle(along) : style
    const fill = paintFor(textStyle.fill, element, new view.Path2D())
    const content = element.textContent ?? ''
    if (!fill || !content.trim() || textStyle.visibility !== 'visible') return false
    paint.setTransform(...matrix)
    paint.globalAlpha = alpha * Number(textStyle.fillOpacity)
    paint.fillStyle = fill.style
    paint.font = `${textStyle.fontStyle} ${textStyle.fontWeight} ${textStyle.fontSize} ${textStyle.fontFamily}`
    paint.textBaseline = 'alphabetic'
    const anchor = textStyle.textAnchor
    const first = (list: SVGAnimatedLengthList) =>
      list.baseVal.numberOfItems ? list.baseVal.getItem(0).value : 0
    const dx = first(element.dx)
    const dy = first(element.dy)
    if (!along) {
      paint.textAlign = anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left'
      paint.fillText(content, first(element.x) + dx, first(element.y) + dy)
      return true
    }
    const id = reference(`url(${along.getAttribute('href') ?? ''})`)
    const route = id ? tree().getElementById(id) : null
    if (!is(route, 'SVGGeometryElement') || !route.getAttribute('d')) return false
    const length = route.getTotalLength()
    const startRaw = along.getAttribute('startOffset') ?? '0'
    const start = startRaw.endsWith('%')
      ? (Number.parseFloat(startRaw) / 100) * length
      : Number.parseFloat(startRaw) || 0
    const total = paint.measureText(content).width
    let distance = start + dx - (anchor === 'middle' ? total / 2 : anchor === 'end' ? total : 0)
    paint.textAlign = 'center'
    for (const glyph of content) {
      const advance = paint.measureText(glyph).width
      const middle = distance + advance / 2
      distance += advance
      if (middle < 0 || middle > length) continue
      const point = route.getPointAtLength(middle)
      const ahead = route.getPointAtLength(Math.min(length, middle + 0.5))
      const behind = route.getPointAtLength(Math.max(0, middle - 0.5))
      const angle = Math.atan2(ahead.y - behind.y, ahead.x - behind.x)
      const cos = Math.cos(angle)
      const sin = Math.sin(angle)
      paint.setTransform(...multiply(matrix, [cos, sin, -sin, cos, point.x, point.y]))
      paint.fillText(glyph, 0, dy)
    }
    return true
  }
  const clipPathOf = (source: SVGClipPathElement, matrix: Matrix) => {
    const shape = new view.Path2D()
    let rule: CanvasFillRule = 'nonzero'
    for (const child of source.children) {
      if (!is(child, 'SVGGeometryElement')) continue
      const style = view.getComputedStyle(child)
      if (style.display === 'none') continue
      if (style.clipRule === 'evenodd') rule = 'evenodd'
      shape.addPath(geometry(child, style), {
        a: matrix[0],
        b: matrix[1],
        c: matrix[2],
        d: matrix[3],
        e: matrix[4],
        f: matrix[5],
      })
    }
    return { shape, rule }
  }
  const render = (
    element: Element,
    matrix: Matrix,
    alpha: number,
    toTexels: Matrix,
    self: boolean,
  ): boolean => {
    if (
      !is(element, 'SVGGraphicsElement') ||
      unpainted.has(element.localName) ||
      element.hasAttribute('data-trail-model')
    )
      return false
    if (is(element, 'SVGForeignObjectElement')) return false
    const style = view.getComputedStyle(element)
    if (style.display === 'none') return false
    const local = !self && style.transform !== 'none' ? multiply(toTexels, ctm(element)) : matrix
    const opacity = self ? alpha : alpha * Number(style.opacity)
    if (opacity <= 0) return false
    const clipId = reference(style.clipPath)
    const clipSource = clipId ? tree().getElementById(clipId) : null
    if (is(clipSource, 'SVGClipPathElement')) {
      paint.save()
      paint.setTransform(1, 0, 0, 1, 0, 0)
      const { shape, rule } = clipPathOf(clipSource, local)
      paint.clip(shape, rule)
    }
    let painted = false
    if (shapes.has(element.localName)) {
      if (style.visibility === 'visible')
        painted = draw(
          local,
          opacity,
          style,
          element,
          geometry(element, style, style.shapeRendering === 'crispedges' ? 1 / scaleOf(local) : 0),
        )
    } else if (is(element, 'SVGTextElement')) {
      painted = write(element, style, local, opacity)
    } else
      for (const child of element.children)
        painted = render(child, local, opacity, toTexels, false) || painted
    if (is(clipSource, 'SVGClipPathElement')) paint.restore()
    return painted
  }
  const bounds = (nodes: Element[], space: Matrix, self: boolean) => {
    const toSpace = invert(space)
    let total: Box | null = null
    for (const node of nodes) {
      const box = boxOf(node, toSpace, self)
      if (!box) continue
      total = total
        ? {
            left: Math.min(total.left, box.left),
            top: Math.min(total.top, box.top),
            right: Math.max(total.right, box.right),
            bottom: Math.max(total.bottom, box.bottom),
          }
        : box
    }
    return total
  }
  const rasterize = (target: Run, space: SVGGraphicsElement, nodes: Element[], self: boolean) => {
    const spaceMatrix = space === target.owner.element ? target.owner.matrix : ctm(space)
    const sceneScale = scaleOf(spaceMatrix) * resolution
    // A palette change keeps the geometry and texel allocation from the previous paint.
    const recolor = !target.dirty && target.width > 0 && target.scale === sceneScale
    const box = recolor ? null : (target.box ?? bounds(nodes, spaceMatrix, self))
    if (box) target.box = box
    target.dirty = false
    target.waiting = 0
    target.palette = palette
    target.revision = revision
    const scale = Math.max(0.25, sceneScale)
    target.scale = sceneScale
    if (!recolor && !box) {
      release(target.texture)
      target.texture = null
      return
    }
    if (box) {
      const margin = padding + 2 / scale
      target.width = Math.min(maxTexture, Math.ceil((box.right - box.left + margin * 2) * scale))
      target.height = Math.min(maxTexture, Math.ceil((box.bottom - box.top + margin * 2) * scale))
      target.base = multiply(scaling(scale), translation(margin - box.left, margin - box.top))
      target.placed = null
    }
    const { width, height } = target
    if (width <= 0 || height <= 0) return
    clearRaster(width, height)
    const toTexels = multiply(target.base, invert(spaceMatrix))
    let painted = false
    for (const node of nodes) painted = render(node, target.base, 1, toTexels, self) || painted
    if (!painted) {
      release(target.texture)
      target.texture = null
      return
    }
    target.texture = upload(target.texture, scratch, false, width, height)
  }
  const rasterizeMask = (owner: Group) => {
    const mask = owner.mask
    if (!mask) return
    const box = localBox(owner.element)
    if (!box) return
    mask.dirty = false
    mask.waiting = 0
    mask.palette = palette
    mask.revision = revision
    const spaceMatrix = owner.matrix
    const scale = Math.max(0.25, scaleOf(spaceMatrix) * resolution)
    mask.scale = scaleOf(spaceMatrix) * resolution
    const margin = padding + 2 / scale
    const width = Math.min(maxTexture, Math.max(1, Math.ceil((box.width + margin * 2) * scale)))
    const height = Math.min(maxTexture, Math.max(1, Math.ceil((box.height + margin * 2) * scale)))
    mask.base = multiply(scaling(scale), translation(margin - box.x, margin - box.y))
    mask.placed = null
    mask.width = width
    mask.height = height
    clearRaster(width, height)
    const toTexels = multiply(mask.base, invert(spaceMatrix))
    // Grayscale keeps luminance in the red channel, so the mask fits one byte per texel.
    paint.filter = 'grayscale(1)'
    for (const child of mask.source.children) render(child, mask.base, 1, toTexels, false)
    paint.filter = 'none'
    mask.texture = upload(mask.texture, scratch, true, width, height)
  }

  const clipFor = (clips: Clip[]) => {
    const chain = clips.map(clip => clip.source.id).join('|')
    const key = `${clipVersion}:${clips.map(clip => `${clipVersions.get(clip.source) ?? 0}@${clip.owner.matrix.map(value => value.toFixed(2)).join(',')}`).join('|')}`
    const known = clipTextures.get(chain)
    if (known?.key === key && known.texture) return known.texture
    clearRaster(canvas.width, canvas.height)
    paint.save()
    for (const clip of clips) {
      const { shape, rule } = clipPathOf(clip.source, clip.owner.screen)
      paint.clip(shape, rule)
    }
    paint.globalAlpha = 1
    paint.fillStyle = '#fff'
    paint.fillRect(0, 0, canvas.width, canvas.height)
    paint.restore()
    const texture = upload(known?.texture ?? null, scratch, true, canvas.width, canvas.height)
    clipTextures.set(chain, { texture, key })
    return texture
  }

  // Frame loop
  const pausedAnimations = new Set<Animation>()
  const pauseAnimations = (animations: Animation[]) => {
    const running = animations.filter(animation => animation.playState === 'running')
    for (const animation of running) {
      pausedAnimations.add(animation)
      animation.pause()
    }
  }
  const resumeAnimations = () => {
    const paused = [...pausedAnimations].filter(animation => animation.playState === 'paused')
    for (const animation of paused) animation.play()
    pausedAnimations.clear()
  }
  let frame = 0
  let timer = 0
  let ready = false
  let surfaceExposed = false
  let surfaceTimer = 0
  let generationPending = true
  let palettePending = false
  let paletteTransition = false
  let palette = 0
  const pageThemeReveal = () => document.documentElement.dataset.homeThemeReveal === 'revealing'
  // Each cell switches between complete palettes after uploads finish. The scene owns its
  // timing, so a completed body transition cannot cut a late palette reveal short.
  let fade: {
    texture: WebGLTexture
    started: number
    updated: number
    progress: number
    origin: number
    scrollX: number
    scrollY: number
  } | null = null
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let resizeFrame: {
    texture: WebGLTexture
    width: number
    height: number
    started: number | null
  } | null = null
  const resizeProgress = () =>
    resizeFrame?.started === null || !resizeFrame
      ? 0
      : Math.min(1, Math.max(0, (clock - resizeFrame.started) / 420))
  const size = { width: canvas.width, height: canvas.height }
  let sizeDirty = true
  // Hold the complete canvas while a whole-scene change fills within the normal raster budget.
  let flush = false
  let revision = 0
  const repaint = () => {
    dirtyGroup(root)
    flush = true
    revision += 1
  }
  // The current frame's time; one frame counts once however many records it brings.
  let clock = 0
  let solar: VisitorDaylight | null = null
  let tintFrom: DaylightTint = [1, 1, 1]
  let tintTo: DaylightTint = [1, 1, 1]
  let tintStarted = 0
  const preparePresentation = () => {
    if (!host.hidden) return
    scene.dataset.trailOpening = 'pending'
    scene.dataset.trailOpeningProgress = '0'
    scene.dataset.trailPaintReady = 'false'
  }
  const tintAt = (now: number): DaylightTint => {
    const progress = reducedMotion.matches
      ? 1
      : Math.min(1, Math.max(0, (now - tintStarted) / 1200))
    const share = progress * progress * (3 - 2 * progress)
    return [
      tintFrom[0] + (tintTo[0] - tintFrom[0]) * share,
      tintFrom[1] + (tintTo[1] - tintFrom[1]) * share,
      tintFrom[2] + (tintTo[2] - tintFrom[2]) * share,
    ]
  }
  const updateTint = () => {
    const next = daylightTint(solar, document.documentElement.dataset.themeMode === 'daylight')
    if (next.every((value, index) => value === tintTo[index])) return
    const now = view.performance.now()
    tintFrom = host.hidden ? next : tintAt(now)
    tintTo = next
    tintStarted = host.hidden ? now - 1200 : now
    preparePresentation()
    scene.dataset.solarTint = next.map(value => value.toFixed(4)).join(' ')
  }
  let settled = 0
  // Groups stand for their mask here.
  const pending: (Run | Group)[] = []
  const coverageBounds = new WeakMap<Run | Group, { revision: number; box: Box | null }>()
  let coverage: {
    texture: WebGLTexture | null
    cells: Uint8Array
    width: number
    height: number
    active: boolean
  } = { texture: null, cells: new Uint8Array(), width: 0, height: 0, active: false }
  const paintBounds = (target: Run | Group): Box | null => {
    const owner = target.kind === 'group' ? target : target.owner
    const layer = target.kind === 'group' ? target.mask : target
    if (!layer) return null
    let matrix = owner.screen
    let box: Box
    if (layer.width && layer.height) {
      matrix = multiply(owner.screen, invert(layer.base))
      box = { left: 0, top: 0, right: layer.width, bottom: layer.height }
    } else if (target.kind === 'run' && target.box) {
      box = target.box
    } else {
      const known = coverageBounds.get(target)
      let next = known?.revision === revision ? known.box : undefined
      if (next === undefined) {
        // Empty runs can share an owner with the whole valley. Measure their own nodes,
        // otherwise an invisible container reserves that owner's entire reveal area.
        next = bounds(
          target.kind === 'run' ? target.nodes : [owner.element],
          owner.matrix,
          target.kind === 'group' || target.nodes[0] === owner.element,
        )
        if (target.kind === 'run') target.box = next
        coverageBounds.set(target, { revision, box: next })
      }
      if (!next) return null
      box = next
    }
    const x = (box.left + box.right) / 2
    const y = (box.top + box.bottom) / 2
    const cx = matrix[0] * x + matrix[2] * y + matrix[4]
    const cy = matrix[1] * x + matrix[3] * y + matrix[5]
    const halfWidth = (box.right - box.left) / 2
    const halfHeight = (box.bottom - box.top) / 2
    const reachX = Math.abs(matrix[0]) * halfWidth + Math.abs(matrix[2]) * halfHeight
    const reachY = Math.abs(matrix[1]) * halfWidth + Math.abs(matrix[3]) * halfHeight
    return { left: cx - reachX, right: cx + reachX, top: cy - reachY, bottom: cy + reachY }
  }
  const paintCoverage = (unfinished: (Run | Group)[]) => {
    if (!unfinished.length && !coverage.active) return
    const cell = trailOpeningCell * resolution
    const width = Math.ceil(size.width / cell)
    const height = Math.ceil(size.height / cell)
    if (coverage.width !== width || coverage.height !== height) {
      coverage.width = width
      coverage.height = height
      coverage.cells = new Uint8Array(width * height)
    }
    coverage.cells.fill(255)
    for (const target of unfinished) {
      const box = paintBounds(target)
      if (!box) continue
      const left = Math.max(0, Math.floor(box.left / cell))
      const right = Math.min(width, Math.ceil(box.right / cell))
      const top = Math.max(0, Math.floor(box.top / cell))
      const bottom = Math.min(height, Math.ceil(box.bottom / cell))
      if (right <= left || bottom <= top) continue
      for (let y = top; y < bottom; y++) coverage.cells.fill(0, y * width + left, y * width + right)
    }
    coverage.active = false
    let available = 0
    for (let index = 0; index < coverage.cells.length; index++) {
      if (!coverage.cells[index]) coverage.active = true
      else available++
    }
    scene.dataset.trailLoadProgress = (available / coverage.cells.length).toFixed(3)
    if (!coverage.active) return
    coverage.texture ??= gl.createTexture()
    if (!coverage.texture) return
    textures.add(coverage.texture)
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D, coverage.texture)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R8,
      width,
      height,
      0,
      gl.RED,
      gl.UNSIGNED_BYTE,
      coverage.cells,
    )
    filter()
  }
  const paintPriority = (target: Run | Group) => {
    const owner = target.kind === 'group' ? target : target.owner
    if (target.kind === 'group' || owner.element.closest('.trail-paper, .trail-ground-base'))
      return -Infinity
    if (owner.element.closest('.trail-camp, .trail-fire, .trail-camp-approach')) return -1e9
    // Full-width ground and route rasters cover the lowest reveal cells. Fill them before
    // individual crowns, otherwise those cells hold the whole opening behind the forest.
    if (owner.element.closest('.trail-ground, .trail-steps, .trail-hinterland')) return -1e8
    if (owner.element.closest('.trail-weather')) return Infinity
    const ground = owner.element.closest<SVGElement>('[data-ground-y]')
    // Untranslated paths can still reach the camp. Use their painted extent rather than
    // the owner's transform origin, so bottom-row work cannot sit behind the mountains.
    return -(ground
      ? Number(ground.dataset.groundY) * resolution
      : (paintBounds(target)?.bottom ?? -Infinity))
  }
  const coverageCeiling = () => {
    const origin = opening.origin()
    const radius = opening.radius()
    if (!coverage.active) return 1
    let ceiling = 1
    const cell = trailOpeningCell * resolution
    for (let index = 0; index < coverage.cells.length; index++) {
      if (coverage.cells[index]) continue
      const y = Math.floor(index / coverage.width)
      const x = index % coverage.width
      const dx = Math.max(0, Math.abs((x + 0.5) * cell - origin.x * resolution) - cell / 2)
      const dy = Math.max(0, Math.abs((y + 0.5) * cell - origin.y * resolution) - cell / 2)
      ceiling = Math.min(ceiling, Math.hypot(dx, dy) / (radius * resolution + cell))
    }
    return Math.max(0, ceiling - 0.000001)
  }
  const stale = (layer: Layer, scale: number) => {
    const due =
      layer.dirty ||
      layer.palette !== palette ||
      (layer.scale > 0 && Math.abs(scale / layer.scale - 1) > 0.2)
    if (due && !layer.waiting) layer.waiting = clock
    return due
  }
  const prepare = (owner: Group, moved: boolean) => {
    if (!chunkVisible(owner)) return
    if (owner.structure) partition(owner)
    if (!owner.fresh) {
      owner.fresh = true
      owner.shown = owner.element === svg || owner.style.display !== 'none'
      owner.opacity = owner.element === svg ? 1 : Number(owner.style.opacity)
      for (const item of owner.items)
        if (item.kind === 'surface')
          item.shown = item.style.display !== 'none' && item.style.visibility === 'visible'
    }
    if (!owner.shown) return
    moved ||= owner.moved
    if (moved) {
      owner.moved = false
      owner.matrix = ctm(owner.element)
      owner.screen = multiply(scaling(resolution), owner.matrix)
      if (owner.mask) owner.mask.placed = null
      for (const item of owner.items) if (item.kind !== 'group') item.placed = null
    }
    const scale = scaleOf(owner.matrix) * resolution
    if (owner.mask && stale(owner.mask, scale)) pending.push(owner)
    for (const item of owner.items) {
      if (item.kind === 'group') prepare(item, moved)
      else if (item.kind === 'surface') {
        if (!item.dirty && item.texture) continue
        item.dirty = false
        if (item.canvas.width && item.canvas.height) {
          item.texture = upload(item.texture, item.canvas)
        }
      } else if (stale(item, scale)) pending.push(item)
    }
  }

  const split = () => {
    for (const group of groups.values())
      for (const item of group.items) {
        if (item.kind !== 'run' || !item.hits.length) continue
        item.hits = item.hits.filter(time => clock - time < 3000)
        if (item.hits.length < 3) {
          if (!item.hits.length) item.hot.clear()
          continue
        }
        // A node that changes again and again gets its own texture, so its run stops redrawing.
        const single = item.nodes.length === 1 && item.nodes[0] === group.element
        if (!single) for (const node of item.hot) promote(node, 'hot')
        item.hits = []
        item.hot.clear()
      }
  }
  // Returns finished fades and removed elements to their neighbours' textures.
  const settle = () => {
    let changed = false
    for (const [element, reason] of promoted) {
      const idle = reason === 'motion' && clock - (fading.get(element) ?? 0) > 2000
      if (!idle && element.isConnected) continue
      promoted.delete(element)
      changed = true
    }
    if (!changed) return
    carriers = new WeakSet()
    for (const element of promoted.keys()) carry(element)
    for (const group of groups.values()) group.structure = true
  }
  // Whether an animation changes pixels inside its target, rather than only moving or fading it.
  const paints = new WeakMap<Animation, { content: boolean; state: AnimationPlayState }>()
  const timing = new Set(['offset', 'computedOffset', 'easing', 'composite'])
  const scan = (animations: Animation[]) => {
    if (palettePending || paletteTransition) {
      const fullPalette = palettePending
      palettePending = false
      // Palette changes during a resize join its replacement instead of stacking two reveals.
      if (resizeFrame?.started !== null && resizeFrame) holdResize()
      if (fullPalette) palette += 1
      if (paletteTransition && !resizeFrame) capturePalette()
      paletteTransition = false
      // Theme colors do not move the scene. Preserve matrices and surface uploads;
      // the palette version already invalidates the rasters that need new colors.
      if (fullPalette) {
        flush = true
        for (const owner of groups.values()) owner.fresh = false
      }
      revision += 1
    }
    for (const animation of animations) {
      const state = animation.playState
      let sampled = paints.get(animation)
      // A stepped fade reaches its final opacity when it finishes. Sample that state once
      // before retaining it, otherwise footprints stay invisible until their textures regroup.
      if (state !== 'running' && sampled?.state === state) continue
      const effect = animation.effect
      if (!is(effect, 'KeyframeEffect') || !effect.target) continue
      const target = effect.target
      if (target.closest('defs')) {
        touchDefinition(target)
        continue
      }
      if (!sampled) {
        const properties = is(animation, 'CSSTransition')
          ? [animation.transitionProperty]
          : effect.getKeyframes().flatMap(frame => Object.keys(frame))
        sampled = {
          content: properties.some(name => !timing.has(name) && !motionProperties.has(name)),
          state,
        }
        paints.set(animation, sampled)
      }
      sampled.state = state
      const known = groups.get(target)
      if (known) {
        if (sampled.content) dirtyGroup(known)
        else {
          fading.set(target, clock)
          known.moved = true
          known.fresh = false
        }
      } else if (!sampled.content) promote(target, 'motion')
      else touch(target)
    }
    if (clock - settled > 1000) {
      settled = clock
      settle()
    }
  }
  const place = (matrix: Matrix) =>
    new Float32Array([matrix[0], matrix[1], 0, matrix[2], matrix[3], 0, matrix[4], matrix[5], 1])
  const none = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0])
  const chunkVisible = (owner: Group) => {
    const element = owner.element
    if (!element.hasAttribute('data-trail-chunk')) return true
    const width = size.width / resolution
    const height = size.height / resolution
    if (element.hasAttribute('data-ground-x')) {
      const x = Number(element.getAttribute('data-ground-x'))
      const y = Number(element.getAttribute('data-ground-y'))
      return x > -256 && x < width + 256 && y > -256 && y < height + 256
    }
    if (!element.hasAttribute('data-patch-left')) return true
    return (
      Number(element.getAttribute('data-patch-left')) < width + 8 &&
      Number(element.getAttribute('data-patch-top')) < height + 8 &&
      Number(element.getAttribute('data-patch-right')) > -8 &&
      Number(element.getAttribute('data-patch-bottom')) > -8
    )
  }
  const surface = (width: number, height: number, clear = true) => {
    if (canvas.width !== width || canvas.height !== height) clipVersion += 1
    if (canvas.width !== width) canvas.width = width
    if (canvas.height !== height) canvas.height = height
    const cssWidth = `${width / resolution}px`
    const cssHeight = `${height / resolution}px`
    if (canvas.style.width !== cssWidth) canvas.style.width = cssWidth
    if (canvas.style.height !== cssHeight) canvas.style.height = cssHeight
    gl.viewport(0, 0, canvas.width, canvas.height)
    gl.clearColor(0, 0, 0, 0)
    if (clear) gl.clear(gl.COLOR_BUFFER_BIT)
    gl.uniform2f(uniforms.viewport, canvas.width, canvas.height)
    gl.uniform4f(uniforms.target, 0, 0, canvas.width, canvas.height)
  }
  const composite = (width = size.width, height = size.height) => {
    // The scene clears what it repaints; the retained canvas keeps the rest.
    surface(width, height, false)
    paintScene()
  }
  type Draw = {
    item: Run | Surface
    texture: WebGLTexture
    placed: Float32Array
    alpha: number
    mask: Float32Array
    maskTexture: WebGLTexture | null
    clipTexture: WebGLTexture | null
  }
  // What a draw last showed, and when that changed.
  type Trace = {
    texture: WebGLTexture
    content: number
    placed: Float32Array
    alpha: number
    mask: Float32Array
    maskTexture: WebGLTexture | null
    maskContent: number
    clipTexture: WebGLTexture | null
    clipContent: number
    changed: number
    // How long the draw must stay unchanged before it joins a cached segment.
    hold: number
  }
  // Canvas pixels: left, top, right, bottom.
  type Rect = [number, number, number, number]
  // A run of unchanged draws composited once into a texture covering their canvas extent.
  type Segment = {
    items: (Run | Surface)[]
    rect: Rect
    texture: WebGLTexture | null
    framebuffer: WebGLFramebuffer | null
    placed: Float32Array
    built: number
  }
  type Unit = { draw: Draw; rect: Rect } | { segment: Segment; rect: Rect }
  const traces = new WeakMap<Run | Surface, Trace>()
  const segments = new Map<Run | Surface, Segment>()
  // The scene state the retained canvas shows, so a frame can redraw only what changed.
  let shown = new Map<Run | Surface, { rect: Rect; below: Run | Surface | null }>()
  let shownScene = ''
  // The canvas size the segment textures were built for. They hold untinted pixels, so a
  // daylight tint change keeps them.
  let segmentScene = ''
  let rectBuffer: WebGLBuffer | null = null
  let rectVertices = new Float32Array(4096)
  const segmentHold = 500
  const segmentMinimum = 4
  const same = (a: Float32Array, b: Float32Array) => {
    if (a === b) return true
    for (let index = 0; index < 9; index++) if (a[index] !== b[index]) return false
    return true
  }
  const trace = (draw: Draw) => {
    const content = contents.get(draw.texture) ?? 0
    const maskContent = draw.maskTexture ? (contents.get(draw.maskTexture) ?? 0) : 0
    const clipContent = draw.clipTexture ? (contents.get(draw.clipTexture) ?? 0) : 0
    const known = traces.get(draw.item)
    if (
      known &&
      known.texture === draw.texture &&
      known.content === content &&
      known.alpha === draw.alpha &&
      known.maskTexture === draw.maskTexture &&
      known.maskContent === maskContent &&
      known.clipTexture === draw.clipTexture &&
      known.clipContent === clipContent &&
      same(known.placed, draw.placed) &&
      same(known.mask, draw.mask)
    )
      return known
    // A draw that moves again after joining a segment waits longer next time, so
    // intermittent motion cannot rebuild its segment every few frames.
    const hold = !known
      ? segmentHold
      : clock - known.changed >= known.hold
        ? Math.min(8000, known.hold * 2)
        : known.hold
    const next: Trace = {
      texture: draw.texture,
      content,
      placed: draw.placed,
      alpha: draw.alpha,
      mask: draw.mask,
      maskTexture: draw.maskTexture,
      maskContent,
      clipTexture: draw.clipTexture,
      clipContent,
      changed: clock,
      hold,
    }
    traces.set(draw.item, next)
    return next
  }
  // The canvas pixels a placed unit quad can touch.
  const extent = (p: Float32Array): Rect => {
    const x = p[6] + Math.min(0, p[0]) + Math.min(0, p[3])
    const y = p[7] + Math.min(0, p[1]) + Math.min(0, p[4])
    return [
      Math.floor(x) - 1,
      Math.floor(y) - 1,
      Math.ceil(x + Math.abs(p[0]) + Math.abs(p[3])) + 1,
      Math.ceil(y + Math.abs(p[1]) + Math.abs(p[4])) + 1,
    ]
  }
  const meets = (a: Rect, b: Rect, gap = 0) =>
    a[0] < b[2] + gap && b[0] < a[2] + gap && a[1] < b[3] + gap && b[1] < a[3] + gap
  const dropSegment = (segment: Segment) => {
    gl.deleteFramebuffer(segment.framebuffer)
    release(segment.texture)
  }
  const dropSegments = () => {
    for (const segment of segments.values()) dropSegment(segment)
    segments.clear()
  }
  const paintScene = () => {
    const tint = tintAt(clock)
    gl.uniform3fv(uniforms.solarTint, tint)
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D, coverage.texture ?? blank)
    gl.uniform1f(uniforms.loadActive, Number(!!fade && coverage.active))
    gl.uniform4f(
      uniforms.themeReveal,
      fade && !pageThemeReveal() ? 1 : 0,
      themeProgress(),
      fade?.origin ?? 1,
      trailOpeningCell * resolution,
    )
    const origin = opening.origin()
    gl.uniform3f(
      uniforms.sweep,
      origin.x * resolution,
      origin.y * resolution,
      opening.radius() * resolution,
    )
    gl.uniform3f(
      uniforms.reveal,
      resizeFrame && resizeFrame.started !== null ? 1 : 0,
      resizeProgress(),
      6 * resolution,
    )
    const draws: Draw[] = []
    const visit = (
      owner: Group,
      alpha: number,
      mask: Float32Array,
      maskTexture: WebGLTexture | null,
      clipped: Clip[],
      clipTexture: WebGLTexture | null,
    ) => {
      if (scene.dataset.trailAmbientReady !== 'true' && owner.element.matches(ambientLayers)) return
      if (!chunkVisible(owner)) return
      alpha *= owner.opacity
      if (!owner.shown || alpha <= 0) return
      if (owner.clip) {
        clipped = [...clipped, { owner, source: owner.clip }]
        clipTexture = clipFor(clipped)
      }
      // A mask still waiting for its texture would show everything it hides.
      if (owner.mask && !owner.mask.texture) return
      if (owner.mask) {
        owner.mask.placed ??= place(
          multiply(
            scaling(1 / owner.mask.width, 1 / owner.mask.height),
            multiply(owner.mask.base, invert(owner.screen)),
          ),
        )
        maskTexture = owner.mask.texture
        mask = owner.mask.placed
      }
      for (const item of owner.items) {
        if (item.kind === 'group') {
          visit(item, alpha, mask, maskTexture, clipped, clipTexture)
          continue
        }
        if (!item.texture) continue
        if (coverage.active && item.kind === 'run' && item.palette !== palette) continue
        if (item.kind === 'run')
          item.placed ??= place(
            multiply(multiply(owner.screen, invert(item.base)), scaling(item.width, item.height)),
          )
        else {
          if (!item.shown) continue
          item.placed ??= place(
            multiply(
              multiply(owner.screen, item.offset),
              scaling(item.canvas.width, item.canvas.height),
            ),
          )
        }
        draws.push({
          item,
          texture: item.texture,
          placed: item.placed,
          alpha,
          mask: maskTexture ? mask : none,
          maskTexture,
          clipTexture,
        })
      }
    }
    visit(root, 1, none, null, [], null)
    // WebGL validates every call; skip the ones that would not change state.
    const state = {
      alpha: -1,
      mask: null as Float32Array | null,
      unit: -1,
      plain: false,
      bound: [null, null, null, null] as (WebGLTexture | null)[],
    }
    const bind = (unit: number, texture: WebGLTexture | null) => {
      if (state.bound[unit] === texture) return
      if (state.unit !== unit) gl.activeTexture(gl.TEXTURE0 + (state.unit = unit))
      gl.bindTexture(gl.TEXTURE_2D, (state.bound[unit] = texture))
    }
    // Segment textures hold untinted pixels; the tint applies when they reach the canvas.
    let building = false
    const shade = (plain: boolean) => {
      if (state.plain === plain) return
      state.plain = plain
      gl.uniform3fv(uniforms.solarTint, plain ? [1, 1, 1] : tint)
    }
    // Draws a unit quad, or only its share of `first..first + count` rect vertices.
    const paint = (draw: Draw, first = -1, count = 0) => {
      shade(building)
      bind(3, coverage.texture ?? blank)
      bind(1, draw.maskTexture ?? blank)
      bind(2, draw.clipTexture ?? blank)
      bind(0, draw.texture)
      if (draw.alpha !== state.alpha) gl.uniform1f(uniforms.alpha, (state.alpha = draw.alpha))
      if (draw.mask !== state.mask)
        gl.uniformMatrix3fv(uniforms.maskPlace, false, (state.mask = draw.mask))
      gl.uniformMatrix3fv(uniforms.place, false, draw.placed)
      if (first < 0) gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      else gl.drawArrays(gl.TRIANGLES, first, count)
    }
    const paintSegment = (segment: Segment, first = -1, count = 0) => {
      shade(false)
      bind(3, coverage.texture ?? blank)
      bind(1, blank)
      bind(2, blank)
      bind(0, segment.texture)
      if (state.alpha !== 1) gl.uniform1f(uniforms.alpha, (state.alpha = 1))
      if (state.mask !== none) gl.uniformMatrix3fv(uniforms.maskPlace, false, (state.mask = none))
      gl.uniformMatrix3fv(uniforms.place, false, segment.placed)
      if (first < 0) gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      else gl.drawArrays(gl.TRIANGLES, first, count)
    }
    // Retained pixels and cached segments hold finished draws, so the theme and resize overlays
    // and the load coverage must be inactive.
    const surfaceKey = `${canvas.width}x${canvas.height}@${resolution}`
    const key = `${surfaceKey}|${tint.join()}`
    const cacheable = !fade && !resizeFrame && !coverage.active
    if (!cacheable || surfaceKey !== segmentScene) {
      dropSegments()
      segmentScene = cacheable ? surfaceKey : ''
    }
    if (!cacheable) {
      shownScene = ''
      shown = new Map()
      gl.clear(gl.COLOR_BUFFER_BIT)
      for (const draw of draws) paint(draw)
      return
    }
    // A new tint changes every pixel: repaint the whole canvas from the segments that remain.
    if (key !== shownScene) {
      shownScene = key
      shown = new Map()
    }
    // What changed since the retained frame: moved or repainted draws at both positions,
    // removed draws, and draws whose neighbour below changed through depth sorting.
    const rects = draws.map(draw => extent(draw.placed))
    const next = new Map<Run | Surface, { rect: Rect; below: Run | Surface | null }>()
    let dirty: Rect[] = []
    const idle = draws.map((draw, index) => {
      const known = trace(draw)
      const rect = rects[index]
      const below = index ? draws[index - 1].item : null
      const before = shown.get(draw.item)
      next.set(draw.item, { rect, below })
      if (!before) dirty.push(rect)
      else if (known.changed === clock || before.below !== below) dirty.push(rect, before.rect)
      return clock - known.changed >= known.hold
    })
    for (const [item, before] of shown) if (!next.has(item)) dirty.push(before.rect)
    shown = next
    // Merge nearby regions: one rect costs a draw per layer it crosses.
    const gap = 8 * resolution
    const regions: Rect[] = []
    for (const [left, top, right, bottom] of dirty) {
      const rect: Rect = [
        Math.max(0, left),
        Math.max(0, top),
        Math.min(canvas.width, right),
        Math.min(canvas.height, bottom),
      ]
      if (rect[2] <= rect[0] || rect[3] <= rect[1]) continue
      // A grown rect can reach regions already passed, so the search restarts after a merge.
      for (let index = 0; index < regions.length; ) {
        const other = regions[index]
        if (!meets(other, rect, gap)) {
          index++
          continue
        }
        rect[0] = Math.min(rect[0], other[0])
        rect[1] = Math.min(rect[1], other[1])
        rect[2] = Math.max(rect[2], other[2])
        rect[3] = Math.max(rect[3], other[3])
        regions.splice(index, 1)
        index = 0
      }
      regions.push(rect)
    }
    dirty = regions
    const covered = dirty.reduce((sum, rect) => sum + (rect[2] - rect[0]) * (rect[3] - rect[1]), 0)
    const whole = covered > 0.5 * canvas.width * canvas.height
    if (!dirty.length) return
    // Premultiplied source-over is associative, so a run of draws composited into a
    // transparent texture, then drawn once, matches drawing each of them in turn.
    // A built segment stays while its draws are unchanged; draws that settle beside it
    // form new segments instead of rebuilding it. Area budgets bound texture memory and
    // the offscreen work one frame may add.
    const area = canvas.width * canvas.height
    let memory = 0
    for (const segment of segments.values())
      memory += (segment.rect[2] - segment.rect[0]) * (segment.rect[3] - segment.rect[1])
    let rebuilt = 0
    const units: Unit[] = []
    const used = new Set<Segment>()
    const build = (from: number, to: number) => {
      const rect: Rect = [canvas.width, canvas.height, 0, 0]
      for (let index = from; index < to; index++) {
        rect[0] = Math.min(rect[0], Math.max(0, rects[index][0]))
        rect[1] = Math.min(rect[1], Math.max(0, rects[index][1]))
        rect[2] = Math.max(rect[2], Math.min(canvas.width, rects[index][2]))
        rect[3] = Math.max(rect[3], Math.min(canvas.height, rects[index][3]))
      }
      const width = rect[2] - rect[0]
      const height = rect[3] - rect[1]
      const items = draws.slice(from, to).map(draw => draw.item)
      const known = segments.get(items[0])
      const reuse =
        known &&
        !used.has(known) &&
        known.rect[2] - known.rect[0] === width &&
        known.rect[3] - known.rect[1] === height
      const added = reuse ? 0 : width * height
      if (
        to - from < segmentMinimum ||
        width <= 0 ||
        height <= 0 ||
        width > maxTexture ||
        height > maxTexture ||
        memory + added > 3 * area ||
        rebuilt + width * height > area
      ) {
        for (let index = from; index < to; index++)
          units.push({ draw: draws[index], rect: rects[index] })
        return
      }
      let segment = reuse ? known : undefined
      if (segment) gl.bindFramebuffer(gl.FRAMEBUFFER, segment.framebuffer)
      else {
        if (known && !used.has(known)) {
          memory -= (known.rect[2] - known.rect[0]) * (known.rect[3] - known.rect[1])
          dropSegment(known)
          segments.delete(items[0])
        }
        const texture = gl.createTexture()
        const framebuffer = gl.createFramebuffer()
        if (!texture || !framebuffer) {
          gl.deleteTexture(texture)
          gl.deleteFramebuffer(framebuffer)
          for (let index = from; index < to; index++)
            units.push({ draw: draws[index], rect: rects[index] })
          return
        }
        textures.add(texture)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, texture)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
        filter()
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
        segment = { items, rect, texture, framebuffer, placed: none, built: clock }
        memory += added
      }
      rebuilt += width * height
      segment.items = items
      segment.rect = rect
      // The texture holds the target's bottom row first, so the quad is flipped to match.
      segment.placed = place([width, 0, 0, -height, rect[0], rect[3]])
      segment.built = clock
      segments.set(items[0], segment)
      state.unit = -1
      state.bound.fill(null)
      gl.viewport(0, 0, width, height)
      gl.uniform4f(uniforms.target, rect[0], rect[1], width, height)
      gl.clear(gl.COLOR_BUFFER_BIT)
      building = true
      for (let index = from; index < to; index++) paint(draws[index])
      building = false
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.uniform4f(uniforms.target, 0, 0, canvas.width, canvas.height)
      units.push({ segment, rect })
      used.add(segment)
    }
    let pending = -1
    for (let index = 0; index < draws.length; ) {
      const segment = segments.get(draws[index].item)
      const kept =
        segment &&
        !used.has(segment) &&
        index + segment.items.length <= draws.length &&
        segment.items.every(
          (item, offset) =>
            draws[index + offset].item === item &&
            (traces.get(item)?.changed ?? Infinity) < segment.built,
        )
      if (segment && kept) {
        if (pending >= 0) build(pending, index)
        pending = -1
        units.push({ segment, rect: segment.rect })
        used.add(segment)
        index += segment.items.length
        continue
      }
      if (idle[index]) {
        if (pending < 0) pending = index
      } else {
        if (pending >= 0) build(pending, index)
        pending = -1
        units.push({ draw: draws[index], rect: rects[index] })
      }
      index++
    }
    if (pending >= 0) build(pending, draws.length)
    for (const [first, segment] of segments) {
      if (used.has(segment)) continue
      dropSegment(segment)
      segments.delete(first)
    }
    if (whole) {
      gl.clear(gl.COLOR_BUFFER_BIT)
      for (const unit of units)
        if ('segment' in unit) paintSegment(unit.segment)
        else paint(unit.draw)
      return
    }
    // Clear the changed regions, then draw each unit once over just the regions it crosses.
    // Its vertices carry their own quad coordinates, so the shaders sample as for a whole quad.
    gl.enable(gl.SCISSOR_TEST)
    for (const [left, top, right, bottom] of dirty) {
      gl.scissor(left, canvas.height - bottom, right - left, bottom - top)
      gl.clear(gl.COLOR_BUFFER_BIT)
    }
    gl.disable(gl.SCISSOR_TEST)
    const spans: { unit: Unit; first: number; count: number }[] = []
    let length = 0
    for (const unit of units) {
      const p = 'segment' in unit ? unit.segment.placed : unit.draw.placed
      const det = p[0] * p[4] - p[1] * p[3]
      if (!det) continue
      const first = length / 2
      for (const rect of dirty) {
        if (!meets(rect, unit.rect)) continue
        const left = Math.max(rect[0], unit.rect[0])
        const top = Math.max(rect[1], unit.rect[1])
        const right = Math.min(rect[2], unit.rect[2])
        const bottom = Math.min(rect[3], unit.rect[3])
        if (rectVertices.length < length + 12) {
          const grown = new Float32Array(rectVertices.length * 2)
          grown.set(rectVertices)
          rectVertices = grown
        }
        for (const [x, y] of [
          [left, top],
          [right, top],
          [left, bottom],
          [left, bottom],
          [right, top],
          [right, bottom],
        ]) {
          rectVertices[length++] = (p[4] * (x - p[6]) - p[3] * (y - p[7])) / det
          rectVertices[length++] = (p[0] * (y - p[7]) - p[1] * (x - p[6])) / det
        }
      }
      if (length / 2 > first) spans.push({ unit, first, count: length / 2 - first })
    }
    rectBuffer ??= gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, rectBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, rectVertices.subarray(0, length), gl.STREAM_DRAW)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    for (const { unit, first, count } of spans)
      if ('segment' in unit) paintSegment(unit.segment, first, count)
      else paint(unit.draw, first, count)
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  }
  const drawCopy = (
    texture: WebGLTexture,
    alpha: number,
    {
      width = canvas.width,
      height = canvas.height,
      left = 0,
      mirror = false,
      reveal = false,
      theme = false,
    }: {
      width?: number
      height?: number
      left?: number
      mirror?: boolean
      reveal?: boolean
      theme?: boolean
    } = {},
  ) => {
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, blank)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, blank)
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D, coverage.texture ?? blank)
    gl.uniform1f(uniforms.loadActive, Number(theme && coverage.active))
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.uniform1f(uniforms.alpha, alpha)
    gl.uniform3f(uniforms.reveal, reveal ? 2 : 0, resizeProgress(), 6 * resolution)
    gl.uniform4f(
      uniforms.themeReveal,
      theme ? 2 : -1,
      themeProgress(),
      fade?.origin ?? 1,
      trailOpeningCell * resolution,
    )
    // Theme and resize copies already contain the lighting of their captured frame.
    gl.uniform3f(uniforms.solarTint, 1, 1, 1)
    gl.uniformMatrix3fv(uniforms.maskPlace, false, none)
    // The copy holds the drawing buffer bottom row first, so the quad is flipped to match.
    gl.uniformMatrix3fv(
      uniforms.place,
      false,
      place([mirror ? -width : width, 0, 0, -height, mirror ? left + width : left, height]),
    )
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
  const trimSurface = () => {
    surfaceTimer = 0
    if (
      !surfaceExposed ||
      !host.hidden ||
      !ready ||
      !active() ||
      canvas.width > maxTexture ||
      canvas.height > maxTexture
    )
      return
    // The retained pixels come back unchanged below, so only the segment textures are released.
    dropSegments()
    const texture = gl.createTexture()
    if (!texture) return
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, canvas.width, canvas.height, 0)
    filter()
    if (gl.getError() !== gl.NO_ERROR) {
      gl.deleteTexture(texture)
      return
    }
    textures.add(texture)
    const { width, height } = canvas
    // Release pooled browser buffers, then restore while hidden so reopening needs no copy.
    canvas.width = 1
    canvas.height = 1
    // Restoring the same pixels does not invalidate full-resolution clip textures.
    canvas.width = width
    canvas.height = height
    gl.viewport(0, 0, canvas.width, canvas.height)
    gl.uniform2f(uniforms.viewport, canvas.width, canvas.height)
    gl.uniform4f(uniforms.target, 0, 0, canvas.width, canvas.height)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    drawCopy(texture, 1)
    release(texture)
    surfaceExposed = false
  }
  const cancelSurfaceTrim = () => {
    view.clearTimeout(surfaceTimer)
    surfaceTimer = 0
  }
  const scheduleSurfaceTrim = () => {
    if (!host.hidden || !surfaceExposed) return
    cancelSurfaceTrim()
    // Finish pending hidden paints before releasing the browser's drawing-buffer pool.
    surfaceTimer = view.setTimeout(trimSurface, 250)
  }
  const endFade = () => {
    if (!fade) return
    release(fade.texture)
    fade = null
    scene.dataset.trailThemePhase = 'complete'
    scene.dataset.trailThemeProgress = '1'
  }
  const themeProgress = () => fade?.progress ?? 0
  const capturePalette = () => {
    scene.dataset.trailAmbientReady = 'false'
    if (
      host.hidden ||
      !ready ||
      reducedMotion.matches ||
      scene.dataset.trailOpening !== 'complete'
    ) {
      endFade()
      return
    }
    const texture = gl.createTexture()
    if (!texture) return
    // The retained buffer includes any interrupted pixel reveal. Copy it before replacing
    // textures or releasing that reveal, so a reversal starts at the actual visible frame.
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, canvas.width, canvas.height, 0)
    filter()
    textures.add(texture)
    endFade()
    fade = {
      texture,
      started: Number(scene.dataset.trailThemeStarted) || clock,
      updated: clock,
      progress: 0,
      origin: opening.radius() * resolution,
      scrollX: view.scrollX,
      scrollY: view.scrollY,
    }
    scene.dataset.trailThemePhase = 'revealing'
    scene.dataset.trailThemeProgress = '0'
  }
  const advanceFade = () => {
    if (!fade) return
    const duration = trailThemeDuration
    const elapsed = Math.min(1, Math.max(0, (clock - fade.started) / duration))
    // Preparation and reveal share one deadline. Recover time spent behind an upload
    // without jumping the frontier across the scene after a delayed frame.
    const advance = Math.min(32, Math.max(0, clock - fade.updated)) / (duration / 2)
    const progress = Math.max(
      fade.progress,
      Math.min(elapsed, coverageCeiling(), fade.progress + advance),
    )
    fade.updated = clock
    fade.progress = progress
    scene.dataset.trailThemeProgress = progress.toFixed(3)
    if (progress >= 1 && !coverage.active) endFade()
  }
  const overlay = () => {
    // The homepage snapshot masks the UI and scene together using this fade's progress.
    if (fade && !pageThemeReveal()) drawCopy(fade.texture, 1, { theme: true })
  }
  const endResize = () => {
    if (resizeFrame) release(resizeFrame.texture)
    resizeFrame = null
    delete scene.dataset.trailResizePhase
  }
  const overlayResize = () => {
    if (!resizeFrame) return
    // Reflect the held edge into newly exposed space. Every cell keeps its native dimensions;
    // the generated landscape replaces this temporary fill when its textures are complete.
    for (let left = 0, tile = 0; left < canvas.width; left += resizeFrame.width, tile++)
      drawCopy(resizeFrame.texture, 1, {
        width: resizeFrame.width,
        height: resizeFrame.height,
        left,
        mirror: tile % 2 !== 0,
        reveal: resizeFrame.started !== null,
      })
  }
  const presentResize = () => {
    if (!resizeFrame || resizeFrame.started !== null) return
    // Grow in buckets and clip on compression, avoiding a drawing-buffer allocation on each
    // pixel of a drag. The committed frame returns to the exact settled dimensions.
    const width =
      host.clientWidth * resolution > canvas.width
        ? Math.ceil(host.clientWidth / 256) * 256 * resolution
        : canvas.width
    const height = Math.max(canvas.height, Math.round(host.clientHeight * resolution))
    surface(width, height)
    overlayResize()
  }
  const holdResize = () => {
    scene.dataset.trailAmbientReady = 'false'
    if (host.hidden) {
      preparePresentation()
      endResize()
      return
    }
    if (!ready || !active()) return
    if (resizeFrame?.started === null) {
      presentResize()
      return
    }
    if (canvas.width > maxTexture || canvas.height > maxTexture) return
    clock = view.performance.now()
    // Recompose the cached frame before geometry changes, including active theme overlays.
    composite(canvas.width, canvas.height)
    overlay()
    overlayResize()
    const texture = resizeFrame?.texture ?? gl.createTexture()
    if (!texture) return
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, canvas.width, canvas.height, 0)
    filter()
    textures.add(texture)
    endFade()
    resizeFrame = { texture, width: canvas.width, height: canvas.height, started: null }
    scene.dataset.trailResizePhase = 'holding'
    presentResize()
  }
  const resize = () => {
    const nextResolution = Math.max(1, view.devicePixelRatio)
    if (!sizeDirty && resolution === nextResolution) return
    sizeDirty = false
    if (resolution !== nextResolution) {
      resolution = nextResolution
      repaint()
    }
    const width = Math.max(1, Math.round(svg.clientWidth * resolution))
    const height = Math.max(1, Math.round(svg.clientHeight * resolution))
    if (size.width === width && size.height === height) return
    size.width = width
    size.height = height
    clipVersion += 1
    endFade()
    root.moved = true
  }
  const active = () =>
    !signal.aborted && !document.hidden && scene.isConnected && !gl.isContextLost()
  const tick = () => {
    frame = 0
    if (!active()) return
    if (scene.dataset.trailRendered !== 'true' && scene.dataset.trailRasterReady !== 'true') return
    // The native reveal can present the retained buffer without resampling the live SVG.
    if (
      scene.dataset.trailReveal !== undefined &&
      scene.dataset.trailOpening === 'complete' &&
      !sizeDirty &&
      !flush &&
      !palettePending &&
      !paletteTransition &&
      !fade &&
      !resizeFrame
    )
      return
    frame = -1
    // Queue the next frame after this one paints, so it runs after the journeys' frame callbacks
    // and reads their writes in the one style and layout pass the frame needs anyway.
    if (!host.hidden || scene.dataset.trailPaintReady !== 'true')
      timer = view.setTimeout(() => {
        timer = 0
        if (scene.dataset.trailOpening === 'complete') frame = view.requestAnimationFrame(tick)
        else tick()
      }, 0)
    else frame = 0
    clock = view.performance.now()
    record(observer.takeRecords())
    // Resizes retain their frame. A newly opened scene starts painting as soon as the camp is placed.
    if (
      host.dataset.trailResizing === 'true' ||
      scene.dataset.campPositioned !== 'true' ||
      (scene.dataset.trailRendered !== 'true' && scene.dataset.trailOpening === 'complete')
    ) {
      return
    }
    if (!attributeSelectors) readSelectors()
    resize()
    const animations = svg.getAnimations({ subtree: true })
    if (scene.dataset.trailOpening !== 'complete' || host.hidden) pauseAnimations(animations)
    scan(animations)
    split()
    pending.length = 0
    prepare(root, false)
    // Spend the saved layout work on bounded palette uploads during the shared sweep.
    const budget = fade || scene.dataset.trailOpening !== 'complete' ? 10 : 6
    const unfinished: (Run | Group)[] = []
    let deferred = 0
    let holes = 0
    let replacements = 0
    const unpainted = (target: Run | Group) =>
      (target.kind === 'group' ? (target.mask?.palette ?? -1) : target.palette) < 0
    const outdated = (target: Run | Group) => {
      const layer = target.kind === 'group' ? target.mask : target
      return layer?.palette !== palette || layer.revision !== revision
    }
    // A scene switch already holds a complete frame. Keep its camp-first order even when
    // winter clothing or crowns introduce new textures higher up the valley.
    const priorities = new Map(pending.map(target => [target, paintPriority(target)]))
    pending.sort(
      (a, b) =>
        Number(outdated(b)) - Number(outdated(a)) ||
        (fade ? 0 : Number(unpainted(b)) - Number(unpainted(a))) ||
        (priorities.get(a) ?? 0) - (priorities.get(b) ?? 0) ||
        (a.kind === 'group' ? (a.mask?.waiting ?? 0) : a.waiting) -
          (b.kind === 'group' ? (b.mask?.waiting ?? 0) : b.waiting),
    )
    const painting = view.performance.now()
    for (const target of pending) {
      const owner = target.kind === 'group' ? target : target.owner
      const layer = target.kind === 'group' ? target.mask : target
      // Weather and tracks resume after a palette transition finishes.
      if (fade && owner.element.closest(ambientLayers)) continue
      const painted = !unpainted(target)
      if (view.performance.now() - painting > budget) {
        if (layer && (layer.palette !== palette || layer.revision !== revision))
          unfinished.push(target)
        if ((target.kind === 'group' ? target.mask?.revision : target.revision) !== revision)
          replacements += 1
        if (!painted) holes += 1
        else if ((target.kind === 'group' ? target.mask?.palette : target.palette) !== palette)
          deferred += 1
        continue
      }
      if (target.kind === 'group') rasterizeMask(target)
      else
        rasterize(
          target,
          target.owner.element,
          target.nodes,
          target.nodes[0] === target.owner.element,
        )
    }
    // Ongoing motion can dirty a replacement again; it must not hold the completed scene forever.
    const painted =
      holes || deferred || replacements || scene.dataset.trailRendered !== 'true' ? 'false' : 'true'
    if (scene.dataset.trailPaintReady !== painted) scene.dataset.trailPaintReady = painted
    // Starting motion can split a retained layer into new textures. Keep the last complete
    // canvas until those textures exist, so budgeted uploads cannot flash empty patches.
    if ((ready && holes && !fade) || (resizeFrame && (holes || (ready && flush && replacements)))) {
      return
    }
    if (fade) paintCoverage(unfinished)
    else {
      coverage.active = false
      scene.dataset.trailLoadProgress = painted === 'true' ? '1.000' : '0.000'
    }
    flush = false
    advanceFade()
    if (resizeFrame) {
      if (reducedMotion.matches || resizeProgress() >= 1) endResize()
      else if (resizeFrame.started === null) {
        resizeFrame.started = clock
        scene.dataset.trailResizePhase = 'revealing'
      }
    }
    const ambient = String(
      scene.dataset.trailRendered === 'true' &&
        (scene.dataset.trailOpening === 'complete' || painted === 'true') &&
        !fade &&
        !resizeFrame,
    )
    if (scene.dataset.trailAmbientReady !== ambient) scene.dataset.trailAmbientReady = ambient
    // The first visible frame uses the finished scene, with every texture at its final scale.
    if (scene.dataset.trailOpening !== 'complete' && painted !== 'true') {
      if (!holes && !deferred && !replacements && scene.dataset.trailRendered !== 'true') {
        view.clearTimeout(timer)
        timer = 0
        frame = 0
      }
      return
    }
    composite()
    overlay()
    overlayResize()
    if (!host.hidden) surfaceExposed = true
    if (painted === 'true' && !fade && !resizeFrame)
      scene.dataset.trailPaintTheme =
        document.documentElement.getAttribute('saved-theme') ?? 'light'
    if (!ready) {
      ready = true
      scene.dataset.trailRenderer = 'webgl'
      for (const owner of groups.values()) owner.fresh = false
    }
    const presenting = scene.dataset.trailOpening !== 'complete'
    opening.present()
    if (presenting && !host.hidden && scene.dataset.trailReveal !== 'revealing') resumeAnimations()
    trimRaster(host.hidden && painted === 'true')
    scheduleSurfaceTrim()
    if (painted !== 'true') schedule()
  }
  const schedule = () => {
    if (!frame && active()) frame = view.requestAnimationFrame(tick)
  }

  const scratchStyle = document.createElement('div').style
  const motionOnly = (previous: string | null, next: string | null) => {
    const read = (value: string | null) => {
      scratchStyle.cssText = value ?? ''
      return new Map(
        [...scratchStyle]
          .filter(name => !motionProperties.has(name) && !motionVariables.has(name))
          .map(name => [name, scratchStyle.getPropertyValue(name)]),
      )
    }
    const before = read(previous)
    const after = read(next)
    return (
      before.size === after.size && [...before].every(([name, value]) => after.get(name) === value)
    )
  }
  // An attribute outside the SVG, such as a hovered blaze on the scene, restyles only the elements
  // its rules select. Without the attribute test, each selector matches them before and after the
  // change. Returns false when the whole scene may change.
  const restyle = (name: string) => {
    const selectors = attributeSelectors?.get(name)
    if (!selectors || !name.startsWith('data-')) return false
    const test = new RegExp(
      `:not\\(\\[${name}(?![\\w-])[^\\]]*\\]\\)|\\[${name}(?![\\w-])[^\\]]*\\]`,
      'g',
    )
    const touched = new Set<Element>()
    for (const selector of selectors) {
      let matches: NodeListOf<Element>
      try {
        matches = tree().querySelectorAll(selector.replace(test, ':is(*)'))
      } catch {
        return false
      }
      for (const element of matches) {
        if (element.contains(svg)) return false
        if (svg.contains(element)) touched.add(element)
      }
    }
    for (const element of touched) touch(element)
    return true
  }
  const record = (records: MutationRecord[]) => {
    if (scene.dataset.trailRendered !== 'true' && scene.dataset.trailRasterReady !== 'true') {
      generationPending = true
      return
    }
    if (generationPending) {
      generationPending = false
      bboxes = new WeakMap()
      discover(svg)
      for (const owner of groups.values()) {
        owner.structure = true
        owner.moved = true
        owner.fresh = false
      }
      repaint()
    }
    let changed = false
    for (const entry of records) {
      const target = is(entry.target, 'Element') ? entry.target : entry.target.parentElement
      if (!target) continue
      if (entry.type === 'attributes') {
        const name = entry.attributeName ?? ''
        const paletteChange = target === document.documentElement
        // A holiday or a chosen season restyles the whole scene: crossfade once it is painted.
        const holidayChange =
          target === scene && (name === 'data-holiday' || name === 'data-foliage-season')
        if (
          target.getAttribute(name) === entry.oldValue ||
          (!paletteChange && !holidayChange && inert(name))
        )
          continue
        changed = true
        if (paletteChange) {
          preparePresentation()
          palettePending = true
          paletteTransition = true
        } else if (holidayChange) {
          paletteTransition ||= scene.dataset.trailOpening === 'complete'
          if (!restyle(name)) repaint()
        } else if (
          !svg.contains(target) &&
          name === 'style' &&
          motionOnly(entry.oldValue, target.getAttribute('style'))
        ) {
          for (const owner of groups.values()) {
            owner.moved = true
            owner.fresh = false
          }
        } else if (
          target === scene &&
          (name === 'data-trail-rendered' || name === 'data-trail-visible')
        )
          schedule()
        else if (!svg.contains(target) && !restyle(name)) repaint()
        if (!svg.contains(target)) continue
        if (target === svg && (name === 'viewBox' || name === 'style')) {
          root.moved = true
          continue
        }
        if (target.closest('defs')) {
          touchDefinition(target)
          continue
        }
        const known = groups.get(target)
        if (
          name === 'transform' ||
          (name === 'style' && motionOnly(entry.oldValue, target.getAttribute('style')))
        ) {
          if (name === 'transform' && target.parentElement) invalidateBox(target.parentElement)
          // Scripted motion has the same texture boundary as CSS motion. Waiting for repeated
          // repaints makes the first gust split and reraster each crown several times.
          if (known) {
            known.moved = true
            if (name === 'style') known.fresh = false
          } else promote(target, 'hot')
          continue
        }
        invalidateBox(target)
        if (known && (name === 'mask' || name === 'clip-path' || name === 'class'))
          known.structure = true
        if (name === 'mask') promote(target)
        touch(target)
        continue
      }
      changed = true
      if (target.closest('defs')) {
        touchDefinition(target)
        continue
      }
      if (entry.type === 'childList') {
        invalidateBox(target)
        for (const node of entry.addedNodes) {
          if (!is(node, 'Element')) continue
          discover(node)
          const movedGroup = groups.get(node)
          if (movedGroup) movedGroup.moved = true
        }
        const known = groups.get(target)
        const leaf = known?.items[0]?.kind === 'run' && known.items[0].nodes[0] === target
        if (known && !leaf) {
          known.structure = true
          continue
        }
      }
      touch(target)
    }
    if (changed) schedule()
  }
  const observer = new MutationObserver(record)
  observer.observe(svg, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeOldValue: true,
    characterData: true,
  })
  observer.observe(scene, { attributes: true, attributeOldValue: true })
  observer.observe(document.documentElement, {
    attributes: true,
    attributeOldValue: true,
    attributeFilter: ['saved-theme'],
  })
  host.addEventListener('trail:prepare', schedule, { signal })
  host.addEventListener('trail:ready', schedule, { signal })

  scene.addEventListener(
    'trail:canopy-paint',
    event => {
      const frameElement = (event.target as Element | null)?.closest('foreignObject')
      if (!frameElement) return
      let owner = frameElement.parentElement
      while (owner && !groups.has(owner)) owner = owner.parentElement
      const item =
        owner &&
        groups
          .get(owner)
          ?.items.find(
            (item): item is Surface => item.kind === 'surface' && item.frame === frameElement,
          )
      if (item) item.dirty = true
    },
    { signal },
  )
  document.addEventListener('visibilitychange', schedule, { signal })
  view.addEventListener('resize', schedule, { signal })
  reducedMotion.addEventListener(
    'change',
    () => {
      if (reducedMotion.matches) {
        endFade()
        tintFrom = [...tintTo]
        tintStarted = view.performance.now() - 1200
      }
      schedule()
    },
    { signal },
  )
  document.addEventListener(
    'themechange',
    () => {
      updateTint()
      schedule()
    },
    { signal },
  )
  subscribeVisitorDaylight(
    document,
    current => {
      solar = current
      updateTint()
      schedule()
    },
    signal,
  )
  host.addEventListener('trail:resize-start', holdResize, { signal })
  host.addEventListener(
    'trail:open',
    () => {
      cancelSurfaceTrim()
      surfaceExposed = true
    },
    { signal },
  )
  host.addEventListener('trail:reveal-end', schedule, { signal })
  host.addEventListener(
    'trail:close',
    () => pauseAnimations(svg.getAnimations({ subtree: true })),
    { signal },
  )
  host.addEventListener(
    'trail:show',
    () => {
      if (scene.dataset.trailOpening === 'complete') resumeAnimations()
      schedule()
    },
    { signal },
  )
  host.addEventListener(
    'trail:hide',
    () => {
      scheduleSurfaceTrim()
      trimRaster(true)
      endFade()
      endResize()
      tintFrom = [...tintTo]
      tintStarted = view.performance.now() - 1200
      schedule()
    },
    { signal },
  )
  const resizeObserver = new ResizeObserver(() => {
    sizeDirty = true
    preparePresentation()
    presentResize()
    root.moved = true
    schedule()
  })
  resizeObserver.observe(svg)
  resizeObserver.observe(host)
  canvas.addEventListener(
    'webglcontextlost',
    event => {
      event.preventDefault()
      cancelSurfaceTrim()
      delete scene.dataset.trailRenderer
      opening.useFallback()
      ready = false
      surfaceExposed = false
      endResize()
      fade = null
    },
    { signal },
  )
  canvas.addEventListener(
    'webglcontextrestored',
    () => {
      textures.clear()
      // The lost context already freed these; deleting them here would raise errors.
      segments.clear()
      shownScene = ''
      segmentScene = ''
      rectBuffer = null
      coverage.texture = null
      coverage.active = false
      endResize()
      fade = null
      setup()
      const reset = (owner: Group) => {
        if (owner.mask) Object.assign(owner.mask, { texture: null, dirty: true, palette: -1 })
        for (const item of owner.items) {
          if (item.kind === 'group') reset(item)
          else Object.assign(item, { texture: null, dirty: true, palette: -1 })
        }
      }
      reset(root)
      opening.useCompositor()
      scene.dataset.trailRenderer = 'webgl'
      schedule()
    },
    { signal },
  )
  setup()
  opening.useCompositor()
  scene.dataset.trailRenderer = 'webgl'
  schedule()
  signal.addEventListener(
    'abort',
    () => {
      view.cancelAnimationFrame(frame)
      view.clearTimeout(timer)
      cancelSurfaceTrim()
      observer.disconnect()
      resizeObserver.disconnect()
      endResize()
      for (const texture of textures) gl.deleteTexture(texture)
      textures.clear()
      groups.clear()
      promoted.clear()
      clipTextures.clear()
      trimRaster(true)
      gl.deleteTexture(blank)
      gl.deleteBuffer(vertexBuffer)
      gl.deleteBuffer(rectBuffer)
      gl.deleteProgram(program)
      canvas.width = 1
      canvas.height = 1
      gl.getExtension('WEBGL_lose_context')?.loseContext()
      delete scene.dataset.trailRenderer
      canvas.remove()
    },
    { once: true },
  )
  return true
}
