import type { JSX } from 'preact'
import { i18n } from '../../i18n'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { GATE, MISSING, STEP_TOP, hourAngle, stoneOuter } from '../scripts/404-gate'

// Every layer shares this viewBox with `slice`, so geometry lines up across the stacked SVGs and the
// pixel canvases. Style rules: flat plates under one crisp ink keyline, texture only from the
// dithered landscape canvases, and stop-motion steps for motion.
const VIEW = '0 0 1600 1000'
const WATER = 870
const TABLET = { x: 805, y: 786, rot: -3, w: 244, h: 60 }

type Rng = () => number
type Pt = [number, number]

function mulberry(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const f = (n: number) => Math.round(n * 10) / 10
const snap = (n: number, g = 10) => Math.round(n / g) * g
const box = (x: number, y: number, w: number, h: number) =>
  `M${f(x)} ${f(y)}h${f(w)}v${f(h)}h${f(-w)}Z`
const polyline = (pts: Pt[]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${f(x)} ${f(y)}`).join('')

type Vine = { stem: string; leaves: JSX.Element[]; delay: number; dur: number; frames: number }

// Square leaf tiles on alternating sides of the stem, each timed to pop as the stem draws past it.
function grow(r: Rng, pts: Pt[], delay: number): Vine {
  const lens = pts.slice(1).map(([x, y], i) => Math.hypot(x - pts[i][0], y - pts[i][1]))
  const total = lens.reduce((s, l) => s + l, 0)
  const dur = 1.2 + total / 180
  const leaves: JSX.Element[] = []
  let walked = 0
  let side = 1
  lens.forEach((len, i) => {
    const [ax, ay] = pts[i]
    const ux = (pts[i + 1][0] - ax) / (len || 1)
    const uy = (pts[i + 1][1] - ay) / (len || 1)
    for (let t = 6; t < len; t += 13) {
      side = -side
      const s = r() < 0.2 ? 13 : 10
      const off = side * (s / 2 + 1)
      const bloom = r() < 0.09
      leaves.push(
        <rect
          class={bloom ? 'nf-leaf nf-bloom' : 'nf-leaf'}
          x={f(ax + ux * t - uy * off - s / 2)}
          y={f(ay + uy * t + ux * off - s / 2)}
          width={s}
          height={s}
          style={`animation-delay:${f(delay + ((walked + t) / total) * dur)}s`}
        />,
      )
    }
    walked += len
  })
  return { stem: polyline(pts), leaves, delay, dur, frames: Math.max(6, Math.round(dur * 8)) }
}

// Ivy climbing the ring: short chords along the rim, between the hour bars and the outer edge.
function creeper(seed: number, from: number, to: number, delay: number): Vine {
  const r = mulberry(seed)
  const pts: Pt[] = []
  const way = Math.sign(to - from)
  for (let a = from; way * (to - a) > 0; a += way * (0.05 + r() * 0.06)) {
    const radius = GATE.r + GATE.ring - 6 + (r() - 0.5) * 6
    pts.push([GATE.x + Math.cos(a) * radius, GATE.y + Math.sin(a) * radius])
  }
  return grow(r, pts, delay)
}

const at = (cx: number, cy: number, a: number, r: number) =>
  `${f(cx + Math.cos(a) * r)} ${f(cy + Math.sin(a) * r)}`

// One voussoir: the annular sector spanning half an hour either side of its hour.
function voussoir(h: number, cx: number, cy: number, out = stoneOuter(h)) {
  const a0 = hourAngle(h) - Math.PI / 12
  const a1 = hourAngle(h) + Math.PI / 12
  const r = GATE.r
  return `M${at(cx, cy, a0, out)}A${out} ${out} 0 0 1 ${at(cx, cy, a1, out)}L${at(cx, cy, a1, r)}A${r} ${r} 0 0 0 ${at(cx, cy, a0, r)}Z`
}

// Hilfiker's station dial cut into the stone: a bar for the hour, a tick for each minute beside it.
function dial(h: number, cx: number, cy: number) {
  const mark = (a: number, len: number, w: number) => {
    const c = Math.cos(a)
    const s = Math.sin(a)
    const r0 = GATE.r + 7
    const r1 = r0 + len
    const nx = (-s * w) / 2
    const ny = (c * w) / 2
    return `M${f(cx + c * r0 + nx)} ${f(cy + s * r0 + ny)}L${f(cx + c * r1 + nx)} ${f(cy + s * r1 + ny)}L${f(cx + c * r1 - nx)} ${f(cy + s * r1 - ny)}L${f(cx + c * r0 - nx)} ${f(cy + s * r0 - ny)}Z`
  }
  const a = hourAngle(h)
  return mark(a, 26, 9) + [-2, -1, 1, 2].map(m => mark(a + (m * Math.PI) / 30, 9, 3)).join('')
}

// The clock gate: twelve voussoirs, one per hour, around an opening onto the rift. The keystone is
// rose, the mossy five o'clock stone sage, and the eight o'clock stone is gone.
function Ruin() {
  const { x, y, r, depth } = GATE
  const hours = Array.from({ length: 12 }, (_, h) => h).filter(h => h !== MISSING)
  const plates = { stone: '', rose: '', sage: '' }
  for (const h of hours) plates[h === 0 ? 'rose' : h === 5 ? 'sage' : 'stone'] += voussoir(h, x, y)
  const ring = hours.map(h => voussoir(h, x, y)).join('')
  const marks = hours.map(h => dial(h, x, y)).join('')
  // The underside of the opening, seen from below its axis: the front edge less the back edge,
  // which sits lower by the ring's depth.
  const half = Math.sqrt(r * r - (depth / 2) ** 2)
  const yi = y + depth / 2
  const soffit = `M${f(x - half)} ${f(yi)}A${r} ${r} 0 1 1 ${f(x + half)} ${f(yi)}A${r} ${r} 0 0 0 ${f(x - half)} ${f(yi)}Z`
  const plinth = box(716, 656, 168, 16) + box(690, 672, 220, STEP_TOP - 672)

  let stepFill = ''
  let spill = ''
  for (let i = 0; i < 4; i++) {
    stepFill += box(620 - i * 50, STEP_TOP + i * 25, 360 + i * 100, 25)
    spill += box(700 - i * 50, STEP_TOP + i * 25, 200 + i * 100, 25)
  }

  const vines: Vine[] = [
    creeper(21, hourAngle(5.6), hourAngle(2.4), 0.3),
    creeper(23, hourAngle(9.7), hourAngle(11.1), 1.1),
  ]

  return (
    <svg
      class="nf-layer nf-temple"
      data-depth="0.42"
      viewBox={VIEW}
      preserveAspectRatio="xMidYMid slice"
    >
      <g id="nf-temple-art">
        <path class="nf-fill-stone" d={soffit} />
        <path class="nf-key" d={soffit} />
        <g class="nf-ring">
          <path class="nf-fill-stone" d={plates.stone} />
          <path class="nf-fill-rose" d={plates.rose} />
          <path class="nf-fill-sage" d={plates.sage} />
          <path class="nf-key" d={ring} />
          <path class="nf-ink" d={marks} />
        </g>
        <path class="nf-light-spill" d={spill} />
        <g class="nf-steps">
          <path class="nf-fill-stone" d={stepFill} />
          <path class="nf-key" d={stepFill} />
        </g>
        <path class="nf-fill-stone" d={plinth} />
        <path class="nf-key" d={plinth} />
        <g class="nf-vines">
          {vines.map(v => (
            <g class="nf-vine">
              <path
                class="nf-stem"
                pathLength={1}
                style={`animation-delay:${f(v.delay)}s;animation-duration:${f(v.dur)}s;animation-timing-function:steps(${v.frames})`}
                d={v.stem}
              />
              {v.leaves}
            </g>
          ))}
        </g>
        <Shore />
        <g class="nf-tablet" transform={`translate(${TABLET.x} ${TABLET.y}) rotate(${TABLET.rot})`}>
          <path class="nf-fill-stone" d={box(-TABLET.w / 2, -TABLET.h / 2, TABLET.w, TABLET.h)} />
          <path class="nf-key" d={box(-TABLET.w / 2, -TABLET.h / 2, TABLET.w, TABLET.h)} />
        </g>
      </g>
      <a class="nf-door" href="/" aria-label="Franchir la porte : retour à l'accueil">
        <circle cx={x} cy={y} r={r} />
      </a>
    </svg>
  )
}

const HUB = 15

// Station-clock hands pivoting on the pole star. The hub is hollow so the star shows through it.
function Hands() {
  const { x, y } = GATE
  const arm = (w: number, len: number, tail: number) =>
    box(x - w / 2, y - len, w, len - HUB) + box(x - w / 2, y + HUB, w, tail - HUB)
  const hour = arm(14, 112, 40)
  const minute = arm(10, GATE.r + 12, 44)
  const second = arm(5, 102, 52)
  return (
    <g id="nf-clock-hands">
      <g data-hand="hour">
        <path class="nf-fill-stone" d={hour} />
        <path class="nf-key" d={hour} />
      </g>
      <g data-hand="minute">
        <path class="nf-fill-stone" d={minute} />
        <path class="nf-key" d={minute} />
      </g>
      <g data-hand="second">
        <path class="nf-fill-rose" d={second} />
        <path class="nf-key nf-hair" d={second} />
        <circle class="nf-fill-rose" cx={x} cy={y - 114} r={12} />
        <circle class="nf-key" cx={x} cy={y - 114} r={12} />
      </g>
      <circle class="nf-key" cx={x} cy={y} r={HUB} />
    </g>
  )
}

function Water() {
  const r = mulberry(99)
  const bars: JSX.Element[] = []
  for (let i = 0; i < 52; i++) {
    const y = snap(WATER + 12 + Math.pow(r(), 0.8) * 110, 5)
    const x = snap(r() * 1600)
    const w = snap(20 + r() * (30 + (y - WATER) * 0.6))
    bars.push(
      <rect
        class={r() < 0.35 ? 'nf-ripple-bar nf-ripple-bright' : 'nf-ripple-bar'}
        style={`animation-duration:${f(1.6 + r() * 2.4)}s;animation-delay:${f(-r() * 4)}s`}
        x={x}
        y={y}
        width={w}
        height="2"
      />,
    )
  }
  // Reflection folded about the waterline; quantized noise shears it in horizontal slices.
  const mirror = `matrix(1 0 0 -0.62 0 ${f(WATER * 1.62)})`
  return (
    <svg
      class="nf-layer nf-water"
      data-depth="0.42"
      viewBox={VIEW}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <defs>
        <clipPath id="nf-water-clip">
          <rect x="-200" y={WATER} width="2000" height="400" />
        </clipPath>
        <filter
          id="nf-ripple"
          x="-0.05"
          y="0"
          width="1.1"
          height="1"
          filterUnits="objectBoundingBox"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.0008 0.06"
            numOctaves="1"
            seed="4"
            result="n"
          >
            <animate
              attributeName="seed"
              dur="1.2s"
              values="4;9;15;22"
              calcMode="discrete"
              repeatCount="indefinite"
            />
          </feTurbulence>
          <feComponentTransfer in="n" result="q">
            <feFuncR type="discrete" tableValues="0.2 0.4 0.5 0.6 0.8" />
            <feFuncG type="table" tableValues="0.5 0.5" />
          </feComponentTransfer>
          <feDisplacementMap
            in="SourceGraphic"
            in2="q"
            scale="26"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
        <linearGradient id="nf-water-fade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#fff" stop-opacity="0.6" />
          <stop offset="1" stop-color="#fff" stop-opacity="0" />
        </linearGradient>
        <mask id="nf-water-mask">
          <rect x="-200" y={WATER} width="2000" height="220" fill="url(#nf-water-fade)" />
        </mask>
      </defs>
      <rect class="nf-water-body" x="-200" y={WATER} width="2000" height="400" />
      <g clip-path="url(#nf-water-clip)" mask="url(#nf-water-mask)">
        <g filter="url(#nf-ripple)">
          <rect x="-200" y={WATER} width="2000" height="220" fill="none" />
          <use href="#nf-temple-art" transform={mirror} />
          <use href="#nf-clock-hands" transform={mirror} />
        </g>
      </g>
      <path class="nf-key nf-shore" d={`M-200 ${WATER}H1800`} />
      <g class="nf-ripples">{bars}</g>
      <g class="nf-splashes" />
    </svg>
  )
}

// The fallen eight o'clock stone, rocked onto its back, and a heap of rubble. Both share the
// ruin's depth, so they stay seated under parallax.
function Shore() {
  const lie = -(GATE.r + GATE.ring / 2)
  const stone = voussoir(6, 0, lie, stoneOuter(MISSING))
  const tiers = box(1170, 780, 150, 24) + box(1192.5, 760, 90, 20)
  return (
    <g class="nf-shore-rocks">
      <g transform="translate(366 772) rotate(-7)">
        <path class="nf-fill-stone" d={stone} />
        <path class="nf-key" d={stone} />
        <path class="nf-ink" d={dial(6, 0, lie)} />
      </g>
      <path class="nf-fill-stone" d={tiers} />
      <path class="nf-key" d={tiers} />
    </g>
  )
}

const NotFound: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
  const t = i18n(cfg.locale).pages.error
  return (
    <div
      class="nf-scene"
      data-tablet={`${TABLET.x},${TABLET.y},${TABLET.rot},${TABLET.w}`}
      data-water={WATER}
    >
      <div class="nf-stage">
        <canvas class="nf-layer nf-sky" data-depth="0.1" aria-hidden="true" />
        <canvas class="nf-layer nf-land" data-depth="0.18" aria-hidden="true" />
        <canvas class="nf-layer nf-rift" data-depth="0.42" aria-hidden="true" />
        <canvas class="nf-layer nf-hill" data-depth="0.42" aria-hidden="true" />
        <Ruin />
        <canvas class="nf-layer nf-grain" data-depth="0.42" aria-hidden="true" />
        <Water />
        <canvas class="nf-layer nf-near" data-depth="0.42" aria-hidden="true" />
        <svg
          class="nf-layer nf-fx"
          data-depth="0.42"
          viewBox={VIEW}
          preserveAspectRatio="xMidYMid slice"
          aria-hidden="true"
        >
          <Hands />
          <g
            class="nf-carving"
            transform={`translate(${TABLET.x} ${TABLET.y}) rotate(${TABLET.rot})`}
          />
          <g class="nf-falling" />
          <g class="nf-flies" />
        </svg>
        <canvas class="nf-layer nf-canopy" data-depth="1" aria-hidden="true" />
      </div>
      <aside class="nf-board" aria-label="Tableau des départs">
        <div class="nf-board-head">
          <span>{cfg.baseUrl ?? cfg.pageTitle}</span>
          <time data-nf-clock>--:--:--</time>
        </div>
        <section>
          <h2 class="nf-row nf-board-title">
            Arrivées
            <i class="nf-lead" />
          </h2>
          <p class="nf-row nf-row-you">
            <span>Vous</span>
            <i class="nf-lead" />
            <span data-nf-arrival>arr --:--</span>
          </p>
          <p class="nf-row">
            <code data-nf-path>/</code>
            <i class="nf-lead" />
            <span>{t.title}</span>
          </p>
        </section>
        <section>
          <h2 class="nf-row nf-board-title">
            Départs
            <i class="nf-lead" />
          </h2>
          <ul class="nf-departures" data-nf-departures>
            <li>
              <a class="nf-row" href="/" data-no-popover>
                <span>Accueil</span>
                <i class="nf-lead" />
                <span>quai 0</span>
              </a>
            </li>
          </ul>
        </section>
        <p class="nf-row nf-row-you nf-board-foot">voie 404 · service {t.notFound}</p>
      </aside>
    </div>
  )
}

export default (() => NotFound) satisfies QuartzComponentConstructor
