import { i18n } from '../../i18n'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { GATE, MISSING, STEP_TOP, hourAngle, slipTransform, stoneOuter } from '../scripts/404-gate'

// Every layer shares this viewBox with `slice`, so geometry lines up across the stacked SVGs and the
// pixel canvases. The ruin below is drawn here but printed by the landscape script into its pixel
// grid: flat plates, one-pixel ink keylines, and `nf-void` shapes that bite pieces out of whatever
// was printed before them. The moss, the cat and the people are painted by the script itself. The
// SVG stays hidden, and keeps only the door, the carved letters and the fireflies on screen.
const VIEW = '0 0 1600 1000'
const WATER = 870
const TABLET = { x: 805, y: 786, rot: -3, w: 244, h: 60 }

const f = (n: number) => Math.round(n * 10) / 10
const box = (x: number, y: number, w: number, h: number) =>
  `M${f(x)} ${f(y)}h${f(w)}v${f(h)}h${f(-w)}Z`
// Round holes, several to a path so they merge into one bite.
const circles = (cs: [number, number, number][]) =>
  cs
    .map(
      ([x, y, r]) =>
        `M${f(x - r)} ${f(y)}A${r} ${r} 0 1 0 ${f(x + r)} ${f(y)}A${r} ${r} 0 1 0 ${f(x - r)} ${f(y)}Z`,
    )
    .join('')

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

// A zigzag through a stone, from its outer face inward, as (radius, degrees) pairs.
const crack = (pts: [number, number][]) =>
  pts
    .map(([r, deg], i) => {
      const a = (deg * Math.PI) / 180
      return `${i ? 'L' : 'M'}${f(GATE.x + Math.cos(a) * r)} ${f(GATE.y + Math.sin(a) * r)}`
    })
    .join('')
const polar = (r: number, deg: number, size: number): [number, number, number] => [
  GATE.x + Math.cos((deg * Math.PI) / 180) * r,
  GATE.y + Math.sin((deg * Math.PI) / 180) * r,
  size,
]

const CRACKS: Record<number, string> = {
  0: crack([
    [252, -88],
    [243, -91],
    [236, -88.5],
    [229, -90.5],
    [221, -89],
  ]),
  2: crack([
    [214, -32],
    [204, -29],
    [195, -32.5],
    [184, -29.5],
    [174, -31],
    [168, -30],
  ]),
  9: crack([
    [238, 184],
    [227, 181.5],
    [217, 184],
  ]),
}

// Chips knocked out of the stones, in each stone's own place before it slipped.
const BITES: Record<number, string> = {
  0: circles([polar(250, -103, 11), polar(244, -100.5, 6)]),
  1: circles([polar(170, -50, 8)]),
  4: circles([polar(214, 43, 13), polar(205, 46, 8)]),
  11: circles([polar(214, 240, 12), polar(212, 244, 7)]),
}

// The clock gate: twelve voussoirs, one per hour, around an opening onto the rift. The keystone is
// rose, the mossy five o'clock stone sage, and the eight o'clock stone is gone. Each stone keeps its
// own group so it can slip, and carries its dial marks, cracks and chips along with it.
function Ruin() {
  const { x, y, r, depth } = GATE
  const hours = Array.from({ length: 12 }, (_, h) => h).filter(h => h !== MISSING)
  // The underside of the opening, seen from below its axis: the front edge less the back edge,
  // which sits lower by the ring's depth.
  const half = Math.sqrt(r * r - (depth / 2) ** 2)
  const yi = y + depth / 2
  const soffit = `M${f(x - half)} ${f(yi)}A${r} ${r} 0 1 1 ${f(x + half)} ${f(yi)}A${r} ${r} 0 0 0 ${f(x - half)} ${f(yi)}Z`
  const plinth = box(716, 656, 168, 16) + box(690, 672, 220, STEP_TOP - 672)

  let stepFill = ''
  for (let i = 0; i < 4; i++) stepFill += box(620 - i * 50, STEP_TOP + i * 25, 360 + i * 100, 25)
  const worn = circles([
    [718, 658, 9],
    [711, 665, 5],
    [474, 790, 12],
    [485, 786, 7],
    [1031, 716, 8],
  ])

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
        {hours.map(h => {
          const stone = voussoir(h, x, y)
          const plate = h === 0 ? 'nf-fill-rose' : h === 5 ? 'nf-fill-sage' : 'nf-fill-stone'
          return (
            <g class="nf-stone" data-hour={h} transform={slipTransform(h)}>
              <path class={plate} d={stone} />
              <path class="nf-key" d={stone} />
              <path class="nf-ink" d={dial(h, x, y)} />
              {CRACKS[h] && <path class="nf-key" d={CRACKS[h]} />}
              {BITES[h] && <path class="nf-void" d={BITES[h]} />}
            </g>
          )
        })}
        <path class="nf-fill-stone" d={stepFill} />
        <path class="nf-key" d={stepFill} />
        <path class="nf-fill-stone" d={plinth} />
        <path class="nf-key" d={plinth} />
        <path class="nf-void" d={worn} />
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

// The fallen eight o'clock stone, rocked onto its back on the left bank. It shares the ruin's
// depth, so it stays seated under parallax. The rubble on the right bank is the cat now, and the
// script paints it.
function Shore() {
  const lie = -(GATE.r + GATE.ring / 2)
  const stone = voussoir(6, 0, lie, stoneOuter(MISSING))
  return (
    <g transform="translate(366 772) rotate(-7)">
      <path class="nf-fill-stone" d={stone} />
      <path class="nf-key" d={stone} />
      <path class="nf-ink" d={dial(6, 0, lie)} />
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
        <canvas class="nf-layer nf-near" data-depth="0.42" aria-hidden="true" />
        <svg
          class="nf-layer nf-fx"
          data-depth="0.42"
          viewBox={VIEW}
          preserveAspectRatio="xMidYMid slice"
          aria-hidden="true"
        >
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
