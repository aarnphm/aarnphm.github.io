// A small relief model of a ride loop: raw WebGL2, no dependencies, drawn only on demand.
// Frame: x east, y SOUTH in metres (the SVG map's frame). World: X east, Y up, Z south.

export interface TerrainData {
  x0: number
  y0: number
  step: number
  cols: number
  rows: number
  min: number
  scale: number
  h: string
}

export interface Palette {
  paper: string
  ink: string
  muted: string
  faint: string
  accent: string
  /** hypsometric tint, lowest ground to highest; each defaults to `paper` */
  low?: string
  mid?: string
  high?: string
}

export interface Pin {
  id: string
  x: number
  y: number
  /** model metres from the top of the ribbon to the label anchor; default `TerrainOptions.lift` */
  lift?: number
}

export interface TerrainView {
  setPalette(p: Palette): void
  rotate(deltaRadians: number): void
  resize(): void
  render(): void
  pins(): { id: string; left: number; top: number; visible: boolean }[]
  dispose(): void
}

export interface TerrainOptions {
  terrain: TerrainData
  route: string
  ghosts?: string[]
  pins?: Pin[]
  palette: Palette
  onFrame?: () => void
  /** vertical exaggeration of the terrain, default 5 */
  exaggeration?: number
  /** contour interval in real metres, default 2; every fifth line is an index line */
  interval?: number
  /** terrain kept around the route's bounding box, metres: [west, north, east, south] */
  margin?: [number, number, number, number]
  /** height of the route ribbon in model metres (not exaggerated), default 40 */
  ribbon?: number
  /** default pin length above the ribbon, model metres, default 70 */
  lift?: number
  /** CSS px kept clear between the model (pin tops included) and the canvas edge, default 14 */
  pad?: number
  /**
   * camera, radians. yaw 0 looks due north; pitch is the angle above the horizon and stays put:
   * drags only turn the block
   */
  yaw?: number
  pitch?: number
}

type V3 = [number, number, number]
type Batch = [first: number, count: number, colour: number, alpha: number, width: number]

const { min, max, hypot, cos, sin, round, floor, abs, exp } = Math

const VS = `#version 300 es
uniform mat4 m;uniform vec2 s;layout(location=0)in vec3 p;layout(location=1)in vec3 n;
out vec3 N;out float H;
void main(){N=n;H=p.y*s.x+s.y;gl_Position=m*vec4(p,1);}`

// c = paper, ink, muted, faint, accent, low, mid, high. u = (curtain alpha or 0, dark theme flag).
// H is elevation in contour intervals; r maps it to 0..1 over the plate's ground for the tint,
// which is 55 % stepped at each contour so the block reads as stacked sheets. k() is one
// anti-aliased line per unit, faded where lines crowd. N.y == 0 marks a wall; a flat top
// (|N.xz| ~ 0) drops its contours. Slopes away from the light take ink (paper in the dark
// theme), slopes towards it take the other.
const FS = `#version 300 es
precision highp float;
uniform vec3 c[8],l;uniform vec2 u,r;in vec3 N;in float H;out vec4 o;
float k(float f){float g=fwidth(f);
return(1.-smoothstep(.6,1.4,abs(fract(f-.5)-.5)/g))*(1.-smoothstep(.2,.45,g));}
vec3 R(float z){return z<.5?mix(c[5],c[6],z*2.):mix(c[6],c[7],z*2.-1.);}
void main(){
if(u.x>0.){o=vec4(c[4],u.x);return;}
vec3 n=normalize(N);
float w=step(n.y,.01),z=clamp((mix(H,floor(H)+.5,.55)-r.x)*r.y,0.,1.),
t=mix((dot(n,l)-l.y)*1.1,dot(n,l)*.35-.2,w),
e=max(w,smoothstep(.012,.035,length(N.xz)))*(.55-w*.2);
vec3 d=R(z),s=u.y>0.?c[0]:c[1],h=u.y>0.?c[1]:c[0];
d=t<0.?mix(d,s,min(-t,u.y>0.?.6:.45)):mix(d,h,min(t,u.y>0.?.15:.35));
o=vec4(mix(mix(d,c[3],k(H)*e),c[2],k(H/5.)*e),1);}`

// one instanced quad per segment a→b, widened in screen space; v = (viewport px, half width px)
const LVS = `#version 300 es
uniform mat4 m;uniform vec3 v;layout(location=0)in vec3 a;layout(location=1)in vec3 b;
void main(){vec4 p=m*vec4(a,1),q=m*vec4(b,1);
vec2 d=normalize((q.xy-p.xy)*v.xy+1e-5);
p=gl_VertexID<2?p:q;
p.xy+=vec2(-d.y,d.x)/v.xy*(float(gl_VertexID&1)*2.-1.)*v.z;
p.z-=.004;gl_Position=p;}`

const LFS = `#version 300 es
precision mediump float;uniform vec4 c;out vec4 o;void main(){o=c;}`

const clamp = (v: number, lo: number, hi: number): number => min(hi, max(lo, v))

/** `M x y l dx dy dx dy ...`: an absolute start, then relative steps. */
const parsePath = (d: string): number[][] => {
  const n = (d.match(/-?[\d.]+/g) ?? []).map(Number)
  const out = [[n[0], n[1]]]
  for (let i = 2; i + 1 < n.length; i += 2) {
    const p = out[out.length - 1]
    out.push([p[0] + n[i], p[1] + n[i + 1]])
  }
  return out
}

/** `rounds` of Chaikin corner cutting, then an even resample every `ds` metres. */
const smooth = (p: number[][], ds: number, rounds: number, closed?: boolean): number[][] => {
  while (rounds--) {
    const q: number[][] = []
    for (let i = 0; i < p.length - (closed ? 0 : 1); i++) {
      const [ax, ay] = p[i]
      const [bx, by] = p[(i + 1) % p.length]
      q.push(
        [ax * 0.75 + bx * 0.25, ay * 0.75 + by * 0.25],
        [ax * 0.25 + bx * 0.75, ay * 0.25 + by * 0.75],
      )
    }
    p = q
  }
  if (closed) p.push(p[0])
  const out = [p[0]]
  let need = ds
  for (let i = 1; i < p.length; i++) {
    let [ax, ay] = p[i - 1]
    const [bx, by] = p[i]
    let len = hypot(bx - ax, by - ay)
    while (len >= need) {
      ax += ((bx - ax) * need) / len
      ay += ((by - ay) * need) / len
      out.push([ax, ay])
      len -= need
      need = ds
    }
    need -= len
  }
  if (closed) out.push(out[0])
  return out
}

export function mountTerrain(canvas: HTMLCanvasElement, opts: TerrainOptions): TerrainView | null {
  const gl = canvas.getContext('webgl2', { antialias: true })
  if (!gl) return null

  const t = opts.terrain
  const ex = opts.exaggeration ?? 5
  const interval = opts.interval ?? 2
  const ribbon = opts.ribbon ?? 40
  const pad = opts.pad ?? 14

  // ---- the route decides the plate: its bounding box plus a margin, clamped to the data
  const corner = parsePath(opts.route)
  const [fx, fy] = corner[0]
  const [lx, ly] = corner[corner.length - 1]
  const route = smooth(corner, 6, 2, hypot(fx - lx, fy - ly) < 80)
  const xs = route.map(p => p[0])
  const ys = route.map(p => p[1])
  const mg = opts.margin ?? [110, 200, 50, 200]
  const cell = t.step / 2
  const west = max(min(...xs) - mg[0], t.x0)
  const north = max(min(...ys) - mg[1], t.y0)
  const nx = floor((min(max(...xs) + mg[2], t.x0 + (t.cols - 1) * t.step) - west) / cell) + 1
  const nz = floor((min(max(...ys) + mg[3], t.y0 + (t.rows - 1) * t.step) - north) / cell) + 1
  const hw = ((nx - 1) * cell) / 2
  const hd = ((nz - 1) * cell) / 2

  // ---- heightfield: resampled at half the source step, then softened
  const raw = Uint8Array.from(atob(t.h), ch => ch.charCodeAt(0))
  const lerp2 = (a: ArrayLike<number>, w: number, h: number, u: number, v: number): number => {
    u = clamp(u, 0, w - 1.001)
    v = clamp(v, 0, h - 1.001)
    const i = u | 0
    const k = (v | 0) * w + i
    const top = a[k] + (a[k + 1] - a[k]) * (u - i)
    return top + (a[k + w] + (a[k + w + 1] - a[k + w]) * (u - i) - top) * (v % 1)
  }
  let G = new Float32Array(nx * nz)
  for (let k = 0; k < nx * nz; k++) {
    G[k] = lerp2(
      raw,
      t.cols,
      t.rows,
      (west + (k % nx) * cell - t.x0) / t.step,
      (north + floor(k / nx) * cell - t.y0) / t.step,
    )
  }
  // two [1 2 1] passes each way: removes the 8-bit terracing and the bilinear creases
  for (let pass = 0; pass < 4; pass++) {
    const B = new Float32Array(nx * nz)
    for (let k = 0; k < nx * nz; k++) {
      const i = k % nx
      const lo = pass & 1 ? (k < nx ? 0 : nx) : i ? 1 : 0
      const hi = pass & 1 ? (k < nx * nz - nx ? nx : 0) : i < nx - 1 ? 1 : 0
      B[k] = (G[k - lo] + 2 * G[k] + G[k + hi]) / 4
    }
    G = B
  }
  // the block's underside sits on the first index contour at least 2 m under the lowest ground
  const metres = (g: number): number => t.min + g * t.scale
  const base = floor((metres(G.reduce((a, b) => min(a, b))) - 2) / (interval * 5)) * interval * 5
  const Y = (g: number): number => (metres(g) - base) * ex
  const top = Y(G.reduce((a, b) => max(a, b)))
  // lowest and highest ground in the crop, metres, for the tint
  const ground = [metres(G.reduce((a, b) => min(a, b))), metres(G.reduce((a, b) => max(a, b)))]

  // ---- geometry
  const verts: number[] = []
  const index: number[] = []
  const lines: number[] = []
  const batches: Batch[] = []
  const batch = (colour: number, alpha: number, width: number, fill: () => void): void => {
    const from = lines.length / 6
    fill()
    batches.push([from, lines.length / 6 - from, colour, alpha, width])
  }
  const seg = (a: V3, b: V3): number => lines.push(...a, ...b)
  const quad = (): void => {
    const b = verts.length / 6
    index.push(b - 4, b - 3, b - 2, b - 2, b - 3, b - 1)
  }
  const g = (i: number, j: number): number => Y(G[clamp(j, 0, nz - 1) * nx + clamp(i, 0, nx - 1)])
  const at = (i: number, j: number): V3 => [i * cell - hw, g(i, j), j * cell - hd]

  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      verts.push(
        ...at(i, j),
        (g(i - 1, j) - g(i + 1, j)) / (2 * cell),
        1,
        (g(i, j - 1) - g(i, j + 1)) / (2 * cell),
      )
      if (i && j) {
        const k = j * nx + i
        index.push(k - nx - 1, k - nx, k - 1, k - 1, k - nx, k)
      }
    }
  }

  // side walls: the cut faces of the block, and its ink outline
  batch(1, 0.9, 1, () => {
    const wall = (n: number, f: (k: number) => V3, wx: number, wz: number): void => {
      let prev: V3 | undefined
      for (let k = 0; k < n; k++) {
        const p = f(k)
        const low: V3 = [p[0], 0, p[2]]
        verts.push(...p, wx, 0, wz, ...low, wx, 0, wz)
        if (prev) {
          quad()
          seg(prev, p)
          seg([prev[0], 0, prev[2]], low)
        } else seg(low, p)
        prev = p
      }
    }
    wall(nx, k => at(k, 0), 0, -1)
    wall(nz, k => at(nx - 1, k), 1, 0)
    wall(nx, k => at(nx - 1 - k, nz - 1), 0, 1)
    wall(nz, k => at(0, nz - 1 - k), -1, 0)
  })
  const solid = index.length

  /** a frame point (x east, y south) set on the surface, `up` model metres above it */
  const drape = (p: number[], up: number): V3 => {
    const u = (p[0] - west) / cell
    const v = (p[1] - north) / cell
    return [u * cell - hw, Y(lerp2(G, nx, nz, u, v)) + up, v * cell - hd]
  }
  const inside = (p: V3): boolean => abs(p[0]) <= hw && abs(p[2]) <= hd
  const stroke = (pts: V3[]): void => {
    for (let i = 1; i < pts.length; i++) {
      if (inside(pts[i - 1]) && inside(pts[i])) seg(pts[i - 1], pts[i])
    }
  }

  // ghost laps: hairlines lying on the surface
  batch(2, 0.13, 1, () => {
    for (const d of opts.ghosts ?? []) stroke(smooth(parsePath(d), 12, 1).map(p => drape(p, 1)))
  })
  const under = batches.length

  // the fastest lap: a curtain standing on the ground, its top edge the hero line
  const foot = route.map(p => drape(p, 0))
  const head = route.map(p => drape(p, ribbon))
  foot.forEach((p, i) => {
    verts.push(...p, 0, 1, 0, ...head[i], 0, 1, 0)
    if (i) quad()
  })
  batch(4, 0.55, 1, () => stroke(foot))
  batch(4, 1, 2, () => stroke(head))

  // pins: snapped to the route, a hairline from the ground up to the label anchor
  const anchors: { id: string; p: V3 }[] = []
  batch(1, 1, 1, () => {
    for (const pin of opts.pins ?? []) {
      const at = drape([pin.x, pin.y], 0)
      let a = at
      let dist = 40
      for (const p of foot) {
        const d = hypot(p[0] - at[0], p[2] - at[2])
        if (d < dist) {
          dist = d
          a = p
        }
      }
      const p: V3 = [a[0], a[1] + ribbon + (pin.lift ?? opts.lift ?? 70), a[2]]
      anchors.push({ id: pin.id, p })
      seg(a, p)
    }
  })

  // what the camera must keep in frame: the block and the pin tops
  const corners: V3[] = anchors.map(a => a.p)
  for (let i = 0; i < 8; i++) corners.push([i & 1 ? hw : -hw, i & 2 ? top : 0, i & 4 ? hd : -hd])

  // ---- GL state (rebuilt after a context loss)
  let main: WebGLProgram
  let line: WebGLProgram
  let meshBuf: WebGLBuffer | null
  let lineBuf: WebGLBuffer | null

  const program = (vs: string, fs: string): WebGLProgram => {
    const p = gl.createProgram() as WebGLProgram
    ;[vs, fs].forEach((src, i) => {
      const s = gl.createShader(i ? gl.FRAGMENT_SHADER : gl.VERTEX_SHADER) as WebGLShader
      gl.shaderSource(s, src)
      gl.compileShader(s)
      gl.attachShader(p, s)
    })
    gl.linkProgram(p)
    return p
  }
  const buffer = (target: number, data: BufferSource): WebGLBuffer | null => {
    const b = gl.createBuffer()
    gl.bindBuffer(target, b)
    gl.bufferData(target, data, gl.STATIC_DRAW)
    return b
  }
  /** make `p` current and return its uniform lookup */
  const use = (p: WebGLProgram): ((name: string) => WebGLUniformLocation | null) => {
    gl.useProgram(p)
    return name => gl.getUniformLocation(p, name)
  }
  /** both programs read two vec3 attributes, 24 bytes apart: per vertex or per instance */
  const point = (buf: WebGLBuffer | null, offset: number, divisor: number): void => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    for (const i of [0, 1]) {
      gl.enableVertexAttribArray(i)
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 24, offset + i * 12)
      gl.vertexAttribDivisor(i, divisor)
    }
  }
  const build = (): void => {
    main = program(VS, FS)
    line = program(LVS, LFS)
    buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(index))
    meshBuf = buffer(gl.ARRAY_BUFFER, new Float32Array(verts))
    lineBuf = buffer(gl.ARRAY_BUFFER, new Float32Array(lines))
    gl.enable(gl.DEPTH_TEST)
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
  }

  // ---- palette: any CSS colour string, resolved by a 1px 2D canvas
  const colours = new Float32Array(24)
  let dark = 0
  const probe = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  const setPalette = (p: Palette): void => {
    const tint = [p.low, p.mid, p.high].map(css => css ?? p.paper)
    ;[p.paper, p.ink, p.muted, p.faint, p.accent, ...tint].forEach((css, i) => {
      if (!probe) return
      probe.clearRect(0, 0, 1, 1)
      probe.fillStyle = css
      probe.fillRect(0, 0, 1, 1)
      probe.getImageData(0, 0, 1, 1).data.forEach((v, k) => k < 3 && (colours[i * 3 + k] = v / 255))
    })
    dark = colours[3] > colours[0] ? 1 : 0
    invalidate()
  }

  // ---- camera: orthographic, fitted to the canvas at every angle
  let yaw = opts.yaw ?? 0.64
  const pitch = opts.pitch ?? 0.6
  let goal = yaw
  let width = 1
  let height = 1
  const mat = new Float32Array(16)

  const fit = (): void => {
    const c = cos(yaw)
    const s = sin(yaw)
    const cp = cos(pitch)
    const sp = sin(pitch)
    let x0 = 1e9
    let x1 = -x0
    let y0 = x0
    let y1 = x1
    for (const p of corners) {
      const x = p[0] * c + p[2] * s
      const y = p[1] * cp - (p[2] * c - p[0] * s) * sp
      x0 = min(x0, x)
      x1 = max(x1, x)
      y0 = min(y0, y)
      y1 = max(y1, y)
    }
    const k = max(0, min((width - 2 * pad) / (x1 - x0), (height - 2 * pad) / (y1 - y0)))
    const kx = (2 * k) / width
    const ky = (2 * k) / height
    const kz = -0.8 / hypot(hw, hd, top)
    // column-major; clip = (kx * xView + ox, ky * yView + oy, kz * depthTowardsCamera, 1)
    mat.set([
      kx * c,
      ky * s * sp,
      -kz * s * cp,
      0,
      0,
      ky * cp,
      kz * sp,
      0,
      kx * s,
      -ky * c * sp,
      kz * c * cp,
      0,
      (-(x0 + x1) / 2) * kx,
      (-(y0 + y1) / 2) * ky,
      0,
      1,
    ])
  }

  const render = (): void => {
    if (gl.isContextLost()) return
    fit()
    gl.disable(gl.BLEND)
    gl.depthMask(true) // before the clear: a masked depth buffer is not cleared
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    const mesh = (alpha: number, from: number, count: number): void => {
      const u = use(main)
      const c = cos(yaw)
      const s = sin(yaw)
      point(meshBuf, 0, 0)
      gl.uniformMatrix4fv(u('m'), false, mat)
      gl.uniform2f(u('s'), 1 / (ex * interval), base / interval)
      gl.uniform3fv(u('c'), colours)
      // light fixed to the viewer (upper left, slightly in front), so relief never inverts
      gl.uniform3f(u('l'), -0.48 * c - 0.36 * s, 0.8, 0.36 * c - 0.48 * s)
      gl.uniform2f(u('u'), alpha, dark)
      gl.uniform2f(u('r'), ground[0] / interval, interval / (ground[1] - ground[0]))
      gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, from * 4)
    }
    const strokes = (from: number, to: number): void => {
      const u = use(line)
      gl.uniformMatrix4fv(u('m'), false, mat)
      for (const [start, count, colour, alpha, w] of batches.slice(from, to)) {
        const k = colour * 3
        point(lineBuf, start * 24, 1)
        gl.uniform3f(u('v'), canvas.width, canvas.height, (w * canvas.width) / width)
        gl.uniform4f(u('c'), colours[k], colours[k + 1], colours[k + 2], alpha)
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count)
      }
    }

    mesh(0, 0, solid)
    gl.enable(gl.BLEND)
    gl.depthMask(false)
    strokes(0, under)
    mesh(dark ? 0.2 : 0.26, solid, index.length - solid)
    strokes(under, 9)
    if (live) opts.onFrame?.()
  }

  // ---- scheduling: one frame per request, a chain of frames only while easing
  let raf = 0
  let live = false
  let then = 0
  const frame = (now: number): void => {
    raf = 0
    const d = goal - yaw
    if (abs(d) > 0.002 && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      // time-based ease, about half a second whatever the frame rate
      yaw += d * (1 - exp(-min(now - then, 50) / 90))
      invalidate()
    } else yaw = goal
    then = now
    render()
  }
  function invalidate(): void {
    raf ||= requestAnimationFrame(frame)
  }

  const resize = (): void => {
    const dpr = min(devicePixelRatio || 1, 2)
    width = canvas.clientWidth || 1
    height = canvas.clientHeight || 1
    canvas.width = round(width * dpr)
    canvas.height = round(height * dpr)
    gl.viewport(0, 0, canvas.width, canvas.height)
    render()
  }

  // ---- input: drags orbit; `touch-action: pan-y` leaves vertical swipes to the page
  let pointer = -1
  let px = 0
  const down = (e: PointerEvent): void => {
    if (e.button) return
    pointer = e.pointerId
    px = e.clientX
    canvas.setPointerCapture(pointer)
  }
  const move = (e: PointerEvent): void => {
    if (e.pointerId != pointer) return
    goal = yaw += (e.clientX - px) * 0.008
    px = e.clientX
    invalidate()
  }
  const up = (): void => {
    pointer = -1
  }
  const events: [string, (e: never) => void][] = [
    ['pointerdown', down],
    ['pointermove', move],
    ['pointerup', up],
    ['pointercancel', up],
    ['webglcontextlost', (e: Event) => e.preventDefault()],
    [
      'webglcontextrestored',
      () => {
        build()
        resize()
      },
    ],
  ]
  for (const [name, fn] of events) canvas.addEventListener(name, fn as EventListener)
  canvas.style.touchAction = 'pan-y'
  canvas.style.cursor = 'grab'

  const observer = new ResizeObserver(resize)

  build()
  setPalette(opts.palette)
  resize()
  live = true
  observer.observe(canvas)

  return {
    setPalette,
    rotate(delta) {
      goal += delta
      invalidate()
    },
    resize,
    render,
    pins() {
      return anchors.map(({ id, p }) => {
        const at = (r: number): number =>
          (mat[r] * p[0] + mat[4 + r] * p[1] + mat[8 + r] * p[2] + mat[12 + r]) / 2 + 0.5
        const left = at(0) * width
        const y = (1 - at(1)) * height
        return { id, left, top: y, visible: left >= 0 && left <= width && y >= 0 && y <= height }
      })
    },
    dispose() {
      cancelAnimationFrame(raf)
      observer.disconnect()
      for (const [name, fn] of events) canvas.removeEventListener(name, fn as EventListener)
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    },
  }
}
