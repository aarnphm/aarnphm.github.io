import type { Analytics } from '../../../../plugins/stores/analytics'
import type { Sport } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { TriathlonContext } from '../../runtime/context'
import { buildLayers as buildLayersNode } from '../../../../util/triathlon-card'
import { clock } from '../../../../util/triathlon-card'
import { triText } from '../../../../util/triathlon-i18n'
import { isRecord } from '../../../../util/type-guards'
import { buildIcon } from '../../activity/primitives'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { monthTicks } from '../shared'
import { polyD } from '../shared'
import { initialAbilitiesModel, updateAbilities } from './abilities-model'

export type SportAbility = Analytics['engine']['abilities']['sports'][number]

export type AbilityAxis = SportAbility['axes'][number]

export const TRI_ABILITIES_SELECTION_KEY = 'tri-abilities-selection'

export const radarUnitText = (presentation: TriathlonPresentation): string =>
  triText(presentation.locale, presentation.distance === 'imperial' ? 'feet' : 'metres')

export const radarDefinition = (presentation: TriathlonPresentation, key: string): string =>
  triText(presentation.locale, key).replace('{unit}', radarUnitText(presentation))

export const radarAxisLabel = (
  presentation: TriathlonPresentation,
  sports: readonly SportAbility[],
  index: number,
): string => {
  const labels = sports
    .map(sport => sport.axes[index]?.label)
    .filter((label): label is string => label != null)
    .map(label => triText(presentation.locale, label))
  return [...new Set(labels)].join(' / ')
}

export const radarAxisDefinition = (
  presentation: TriathlonPresentation,
  sport: Sport,
  axis: AbilityAxis,
): string => {
  const definition = (key: string): string => radarDefinition(presentation, key)
  switch (axis.key) {
    case 'sprint':
      if (sport === 'bike') return definition('radar sprint bike definition')
      if (sport === 'run') return definition('radar sprint run definition')
      return definition('radar sprint swim definition')
    case 'threshold':
      if (sport === 'bike') return definition('radar threshold bike definition')
      if (sport === 'run') return definition('radar threshold run definition')
      return definition('radar threshold swim definition')
    case 'endurance':
      return definition('radar endurance definition')
    case 'climb':
      if (sport === 'swim') return definition('radar pace swim definition')
      if (sport === 'run') return definition('radar climb run definition')
      return definition('radar climb bike definition')
    case 'stride':
      return definition('radar stride run definition')
    case 'cadence':
      if (sport === 'bike') return definition('radar cadence bike definition')
      if (sport === 'run') return definition('radar cadence run definition')
      return definition('radar stroke rate swim definition')
    case 'recovery':
      return definition('radar recovery definition')
    case 'oscillation':
      return definition('radar oscillation run definition')
  }
}

export const radarNotationDefinition = (
  presentation: TriathlonPresentation,
  axis: AbilityAxis,
): string => {
  const definition = (key: string): string => radarDefinition(presentation, key)
  switch (axis.rawUnit) {
    case 'w/kg':
      return definition('radar unit wkg definition')
    case 'ctl':
      return definition('radar unit ctl definition')
    case 'm/h':
      return presentation.distance === 'imperial'
        ? definition('radar unit fth definition')
        : definition('radar unit mh definition')
    case 'm/s':
      return definition('radar unit mspeed definition')
    case 's/100m':
      return definition('radar unit s100m definition')
    case 'rpm':
      return definition('radar unit rpm definition')
    case 'spm':
      return definition('radar unit spm definition')
    case 'str/min':
      return definition('radar unit strmin definition')
    case 'readiness':
      return definition('radar unit readiness definition')
    case 'ms':
      return definition('radar unit ms definition')
    case 'm':
      return definition('radar unit stride definition')
    case 'cm':
      return definition('radar unit oscillation definition')
    default:
      return definition('radar unit default definition')
  }
}

export const radarPaceHint = (
  presentation: TriathlonPresentation,
  sport: Sport,
  axis: AbilityAxis,
): string | null => {
  if (axis.rawUnit !== 'm/s' || axis.rawValue == null || axis.rawValue <= 0) return null
  if (sport === 'swim') return `${clock(100 / axis.rawValue)} /100m`
  return presentation.distance === 'imperial'
    ? `${clock(1609.344 / axis.rawValue)} /mi`
    : `${clock(1000 / axis.rawValue)} /km`
}

export const buildAbilities = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount?: () => () => void } => {
  const text = (key: string): string => context.formatter.text(key)
  const domF = createDomFactory(context.presentation)
  const block = el('div', 'tri-engine-radar')
  block.appendChild(anaTitle(context.formatter, 'abilities', 'radar'))
  const sports = data.engine.abilities.sports.filter(sp => sp.axes.length > 0)
  if (!sports.length || sports.every(sp => sp.axes.every(a => a.score == null))) {
    block.appendChild(el('div', 'tri-ana-empty', text('not enough data')))
    return { element: block }
  }
  let reduced = false
  const availableSports = sports.map(sport => sport.sport)
  let model = initialAbilitiesModel(availableSports)
  const pressed = new Set<Sport>(model.sports)
  let avg = model.average
  const applyModel = (next: typeof model): void => {
    model = next
    avg = model.average
    pressed.clear()
    for (const sport of model.sports) pressed.add(sport)
  }
  const singleOf = (): SportAbility | null =>
    !avg && pressed.size === 1 ? (sports.find(sp => pressed.has(sp.sport)) ?? null) : null

  const tabs = el('div', 'tri-radar-sports', undefined, {
    role: 'group',
    'aria-label': 'radar sports',
  })
  const avgTab = el('button', 'tri-radar-sport tri-radar-sport--avg', undefined, {
    type: 'button',
    'aria-pressed': 'false',
    'aria-label': text('average'),
    title: text('average'),
  })
  avgTab.appendChild(buildLayersNode(domF) as SVGElement)
  tabs.appendChild(avgTab)
  const tabOf = new Map<Sport, HTMLElement>()
  for (const sp of sports) {
    const tab = el('button', `tri-radar-sport tri-radar-sport--${sp.sport}`, undefined, {
      type: 'button',
      'aria-pressed': pressed.has(sp.sport) ? 'true' : 'false',
      'aria-label': sp.sport,
      title: sp.sport,
      'data-sport': sp.sport,
    })
    tab.appendChild(buildIcon(context.presentation, sp.sport))
    tabOf.set(sp.sport, tab)
    tabs.appendChild(tab)
  }
  block.appendChild(tabs)

  const axesRef = sports[0].axes
  const axesN = axesRef.length
  const cx = 50
  const cy = 50
  const R = 36
  const angle = (i: number): number => ((-90 + (360 / axesN) * i) * Math.PI) / 180
  const pt = (i: number, score: number): [number, number] => {
    const th = angle(i)
    const r = (R * score) / 100
    return [cx + r * Math.cos(th), cy + r * Math.sin(th)]
  }
  const zeros = (): number[] => axesRef.map(() => 0)
  const ringD = (vals: number[]): string => `${polyD(vals.map((v, i) => pt(i, v)))} Z`
  const s = svg('svg', { class: 'tri-radar-svg', viewBox: '0 0 100 100' })
  for (const g of [25, 50, 75, 100])
    s.appendChild(svg('path', { d: ringD(axesRef.map(() => g)), class: 'tri-radar-grid' }))
  axesRef.forEach((_, i) => {
    const [px, py] = pt(i, 100)
    s.appendChild(svg('line', { x1: cx, y1: cy, x2: px, y2: py, class: 'tri-radar-spoke' }))
  })
  type RadarKey = Sport | 'avg'
  const solidOf = new Map<RadarKey, SVGElement>()
  const projPathOf = new Map<RadarKey, SVGElement>()
  const radarKeys: RadarKey[] = [...sports.map(sp => sp.sport), 'avg']
  for (const k of radarKeys) {
    const path = svg('path', { d: ringD(zeros()), class: `tri-radar-fill tri-radar-fill--${k}` })
    s.appendChild(path)
    solidOf.set(k, path)
  }
  for (const k of radarKeys) {
    const path = svg('path', { d: ringD(zeros()), class: `tri-radar-proj tri-radar-proj--${k}` })
    s.appendChild(path)
    projPathOf.set(k, path)
  }
  const dots = axesRef.map((_, i) => {
    const [px, py] = pt(i, 0)
    const dot = svg('circle', { cx: px, cy: py, r: 1.4, class: 'tri-radar-dot' })
    s.appendChild(dot)
    return dot
  })
  type RadarAxisLabel = { active: 0 | 1; nodes: [SVGElement, SVGElement] }
  const labels = axesRef.map((a, i): RadarAxisLabel => {
    const th = angle(i)
    const attrs = {
      x: cx + (R + 8) * Math.cos(th),
      y: cy + (R + 8) * Math.sin(th) + 1.6,
      'text-anchor': Math.abs(Math.cos(th)) < 0.3 ? 'middle' : Math.cos(th) > 0 ? 'start' : 'end',
    }
    const current = svg('text', { ...attrs, class: 'tri-radar-ax tri-radar-ax--active' })
    const next = svg('text', { ...attrs, class: 'tri-radar-ax' })
    current.textContent = text(a.label)
    next.textContent = current.textContent
    s.append(current, next)
    return { active: 0, nodes: [current, next] }
  })
  block.appendChild(s)

  const keyCap = el('div', 'tri-radar-key')
  const nowKey = el('span', 'tri-radar-key-item')
  nowKey.append(
    el('span', 'tri-radar-swatch tri-radar-swatch--now'),
    el('span', undefined, text('now')),
  )
  const projKey = el('span', 'tri-radar-key-item')
  projKey.append(
    el('span', 'tri-radar-swatch tri-radar-swatch--proj'),
    el('span', undefined, `${text('projected')} +28d`),
  )
  keyCap.append(nowKey, projKey)
  block.appendChild(keyCap)

  const shown = new Map<RadarKey, { solid: number[]; proj: number[] }>()
  for (const k of radarKeys) shown.set(k, { solid: zeros(), proj: zeros() })
  let raf = 0
  const apply = (): void => {
    for (const [k, st] of shown) {
      solidOf.get(k)!.setAttribute('d', ringD(st.solid))
      projPathOf.get(k)!.setAttribute('d', ringD(st.proj))
    }
    const single = singleOf()
    const focus = avg ? shown.get('avg') : single ? shown.get(single.sport) : null
    if (focus) {
      focus.solid.forEach((v, i) => {
        const [px, py] = pt(i, v)
        dots[i].setAttribute('cx', px.toFixed(2))
        dots[i].setAttribute('cy', py.toFixed(2))
      })
    }
  }
  const avgAxis = (pick: (a: AbilityAxis) => number | null | undefined): number[] =>
    axesRef.map((_, i) => {
      const xs = sports.map(sp => pick(sp.axes[i])).filter((v): v is number => v != null)
      return xs.length ? xs.reduce((acc, v) => acc + v, 0) / xs.length : 0
    })
  const targetOf = (sp: SportAbility): { solid: number[]; proj: number[] } =>
    !avg && pressed.has(sp.sport)
      ? { solid: sp.axes.map(a => a.score ?? 0), proj: sp.axes.map(a => a.proj ?? a.score ?? 0) }
      : { solid: zeros(), proj: zeros() }
  const avgTarget = (): { solid: number[]; proj: number[] } =>
    avg
      ? { solid: avgAxis(a => a.score), proj: avgAxis(a => a.proj ?? a.score) }
      : { solid: zeros(), proj: zeros() }
  const morphAll = (animate: boolean): void => {
    window.cancelAnimationFrame(raf)
    const targets = new Map<RadarKey, { solid: number[]; proj: number[] }>(
      sports.map(sp => [sp.sport, targetOf(sp)] as const),
    )
    targets.set('avg', avgTarget())
    if (!animate || reduced) {
      for (const [k, g] of targets) shown.set(k, g)
      apply()
      return
    }
    const from = new Map(
      [...shown].map(([k, v]) => [k, { solid: [...v.solid], proj: [...v.proj] }] as const),
    )
    const t0 = performance.now()
    const tick = (now: number): void => {
      if (!block.isConnected || context.signal.aborted) return
      const t = Math.min(1, (now - t0) / 450)
      const e = 1 - (1 - t) ** 3
      for (const [k, g] of targets) {
        const f = from.get(k)!
        shown.set(k, {
          solid: f.solid.map((v, i) => v + (g.solid[i] - v) * e),
          proj: f.proj.map((v, i) => v + (g.proj[i] - v) * e),
        })
      }
      apply()
      if (t < 1) raf = window.requestAnimationFrame(tick)
    }
    raf = window.requestAnimationFrame(tick)
  }
  const applyAxisClasses = (): void => {
    const single = singleOf()
    axesRef.forEach((_, i) => {
      const isNull = avg
        ? sports.every(sp => sp.axes[i].score == null)
        : single != null && single.axes[i].score == null
      dots[i].setAttribute('class', isNull ? 'tri-radar-dot tri-radar-dot--null' : 'tri-radar-dot')
      for (const label of labels[i].nodes) label.classList.toggle('tri-radar-ax--null', isNull)
    })
  }

  const devBox = el('div', 'tri-dev-slot')
  const legendOn = new Set<string>(['endurance', 'recovery', 'stride', 'oscillation'])
  let revealDev: (() => void) | null = null
  const DEV_KEYS = [
    'endurance',
    'recovery',
    'stride',
    'oscillation',
    'cadence',
    'sprint',
    'threshold',
    'climb',
  ] as const
  type DevSeries = {
    key: string
    cls: string
    dotCls: string
    label: string
    vals: (number | null)[]
    toggle: boolean
  }
  let devFocus: ((event: MouseEvent) => void) | null = null
  let devLeave: (() => void) | null = null
  let devToggle: ((key: string, item: HTMLElement) => void) | null = null

  const swapDev = (next: HTMLElement | null, animate: boolean): void => {
    if (!animate) {
      devBox.replaceChildren(...(next ? [next] : []))
      return
    }
    const current = devBox.querySelector<HTMLElement>(':scope > .tri-dev:not(.tri-dev--leaving)')
    if (current) {
      current.classList.add('tri-dev--leaving')
    }
    if (!next) return
    next.classList.add('tri-dev--entering')
    devBox.appendChild(next)
    window.requestAnimationFrame(() =>
      window.requestAnimationFrame(() => next.classList.remove('tri-dev--entering')),
    )
  }

  const renderDev = (draw: 'defer' | 'animate' | 'none'): void => {
    revealDev = null
    const single = singleOf()
    const hist = (single ?? sports[0]).history ?? []
    const meanAt = (sp: SportAbility, i: number): number | null => {
      const h = sp.history[i]
      if (!h) return null
      const xs = DEV_KEYS.map(k => h[k]).filter((v): v is number => v != null)
      return xs.length ? xs.reduce((acc, v) => acc + v, 0) / xs.length : null
    }
    let series: DevSeries[]
    if (single) {
      series = DEV_KEYS.filter(k => hist.filter(h => h[k] != null).length >= 2).map(k => ({
        key: k,
        cls: `tri-dev-line--${k}`,
        dotCls: `tri-dev-dot--${k}`,
        label: text(single.axes.find(axis => axis.key === k)?.label ?? k),
        vals: hist.map(h => h[k] ?? null),
        toggle: true,
      }))
    } else if (avg) {
      series = [
        {
          key: 'avg',
          cls: 'tri-line-avg',
          dotCls: 'tri-dev-dot--sp-avg',
          label: text('average'),
          vals: hist.map((_, i) => {
            const xs = sports.map(sp => meanAt(sp, i)).filter((v): v is number => v != null)
            return xs.length ? Math.round(xs.reduce((acc, v) => acc + v, 0) / xs.length) : null
          }),
          toggle: false,
        },
      ].filter(sr => sr.vals.filter(v => v != null).length >= 2)
    } else {
      series = sports
        .filter(sp => pressed.has(sp.sport))
        .map(sp => ({
          key: sp.sport as string,
          cls: `tri-line-${sp.sport}`,
          dotCls: `tri-dev-dot--sp-${sp.sport}`,
          label: text(sp.sport),
          vals: sp.history.map((_, i) => {
            const v = meanAt(sp, i)
            return v == null ? null : Math.round(v)
          }),
          toggle: false,
        }))
        .filter(sr => sr.vals.filter(v => v != null).length >= 2)
    }
    if (hist.length < 2 || !series.length) {
      devFocus = null
      devLeave = null
      devToggle = null
      swapDev(null, draw === 'animate')
      return
    }
    const dev = el('div', 'tri-dev')
    const W = 100
    const H = 30
    const xAt = (i: number): number => (i / (hist.length - 1)) * W
    const yAt = (v: number): number => H - (v / 100) * H
    const frame = el('div', 'tri-dev-frame')
    const yax = el('div', 'tri-dev-yax')
    for (const gv of [100, 75, 50, 25, 0]) {
      const lab = el('span', 'tri-dev-yt', String(gv))
      lab.style.top = `${100 - gv}%`
      yax.appendChild(lab)
    }
    frame.appendChild(yax)
    const plot = el('div', 'tri-dev-plot')
    const sv = svg('svg', {
      class: 'tri-dev-svg',
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'none',
    })
    for (const gv of [100, 75, 50, 25, 0])
      sv.appendChild(
        svg('line', {
          x1: 0,
          y1: yAt(gv),
          x2: W,
          y2: yAt(gv),
          class: gv === 50 ? 'tri-dev-grid tri-dev-grid--mid' : 'tri-dev-grid',
        }),
      )
    const linesG = svg('g', { class: 'tri-dev-lines' }) as SVGGElement
    sv.appendChild(linesG)
    const paths = new Map<string, SVGElement>()
    for (const sr of series) {
      const d = sr.vals
        .map((v, i) => ({ x: xAt(i), v }))
        .filter((p): p is { x: number; v: number } => p.v != null)
        .map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(2)} ${yAt(p.v).toFixed(2)}`)
        .join(' ')
      const off = sr.toggle && !legendOn.has(sr.key)
      const path = svg('path', {
        d,
        class: `tri-dev-line ${sr.cls}${off ? ' tri-dev-line--off' : ''}`,
      })
      linesG.appendChild(path)
      paths.set(sr.key, path)
    }
    const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-chart-cursor' })
    sv.appendChild(cursor)
    plot.appendChild(sv)
    const readoutEl = el('div', 'tri-chart-readout tri-dev-read')
    plot.appendChild(readoutEl)
    const renderRead = (i: number): void => {
      const rows: HTMLElement[] = [
        el('span', 'tri-dev-read-date', context.formatter.shortDate(hist[i].date)),
      ]
      for (const sr of series) {
        const v = sr.vals[i]
        if (paths.get(sr.key)!.classList.contains('tri-dev-line--off') || v == null) continue
        const row = el('div', 'tri-dev-read-row')
        row.append(
          el('span', `tri-dev-dot ${sr.dotCls}`),
          el('span', 'tri-dev-read-k', sr.label),
          el('span', 'tri-dev-read-v', String(v)),
        )
        rows.push(row)
      }
      readoutEl.replaceChildren(...rows)
    }
    const focusIndex = (i: number, hover: boolean): void => {
      const idx = Math.round(clampN(i, 0, hist.length - 1))
      const cxAttr = xAt(idx).toFixed(2)
      cursor.setAttribute('x1', cxAttr)
      cursor.setAttribute('x2', cxAttr)
      readoutEl.style.left = `${clampN((xAt(idx) / W) * 100, 6, 80).toFixed(2)}%`
      readoutEl.style.right = 'auto'
      renderRead(idx)
      dev.classList.toggle('tri-chart--hover', hover)
    }
    const indexAt = (event: MouseEvent): number => {
      const rect = sv.getBoundingClientRect()
      return Math.round(clampN((event.clientX - rect.left) / rect.width, 0, 1) * (hist.length - 1))
    }
    devFocus = event => focusIndex(indexAt(event), true)
    devLeave = () => dev.classList.remove('tri-chart--hover')
    const xax = el('div', 'tri-dev-xax')
    for (const t of monthTicks(
      context.formatter,
      hist.map(h => h.date),
      i => xAt(i),
    )) {
      const lab = el('span', `tri-dev-xt${t.cls ? ' tri-dev-xt--first' : ''}`, t.label)
      lab.style.left = `${t.pct.toFixed(2)}%`
      xax.appendChild(lab)
    }
    frame.appendChild(plot)
    dev.appendChild(frame)
    dev.appendChild(xax)
    const legend = el('div', 'tri-dev-legend')
    for (const sr of series) {
      if (!sr.toggle) {
        const item = el('span', 'tri-dev-leg tri-dev-leg--static')
        item.append(
          el('span', `tri-dev-dot ${sr.dotCls}`),
          el('span', 'tri-dev-leg-name', sr.label),
        )
        legend.appendChild(item)
        continue
      }
      const item = el(
        'button',
        `tri-dev-leg${legendOn.has(sr.key) ? '' : ' tri-dev-leg--off'}`,
        undefined,
        { type: 'button', 'data-dev-key': sr.key },
      )
      item.append(el('span', `tri-dev-dot ${sr.dotCls}`), el('span', 'tri-dev-leg-name', sr.label))
      legend.appendChild(item)
    }
    devToggle = (key, item) => {
      const path = paths.get(key)
      if (!path) return
      const hidden = path.classList.toggle('tri-dev-line--off')
      item.classList.toggle('tri-dev-leg--off', hidden)
      if (hidden) legendOn.delete(key)
      else legendOn.add(key)
    }
    dev.appendChild(legend)
    if (draw !== 'none') {
      linesG.style.clipPath = 'inset(0 100% 0 0)'
      const reveal = (): void => {
        linesG.style.clipPath = 'inset(0 0 0 0)'
      }
      if (draw === 'animate')
        window.requestAnimationFrame(() => window.requestAnimationFrame(reveal))
      else revealDev = reveal
    }
    swapDev(dev, draw === 'animate')
  }
  block.appendChild(devBox)

  let revealed = reduced
  const syncChrome = (): void => {
    block.dataset.sport = avg ? 'avg' : (singleOf()?.sport ?? (pressed.size ? 'all' : 'none'))
    block.dataset.pressed = avg ? sports.map(sp => sp.sport).join(',') : [...pressed].join(',')
    block.classList.toggle('tri-engine-radar--multi', !avg && pressed.size !== 1)
    block.classList.toggle('tri-engine-radar--avg', avg)
    for (const sp of sports)
      block.classList.toggle(
        `tri-engine-radar--${sp.sport}`,
        !avg && pressed.size === 1 && pressed.has(sp.sport),
      )
    avgTab.setAttribute('aria-pressed', avg ? 'true' : 'false')
    for (const [k, tab] of tabOf)
      tab.setAttribute('aria-pressed', !avg && pressed.has(k) ? 'true' : 'false')
    const activeSports =
      avg || pressed.size === 0 ? sports : sports.filter(sport => pressed.has(sport.sport))
    labels.forEach((label, index) => {
      const text = radarAxisLabel(context.presentation, activeSports, index)
      const current = label.nodes[label.active]
      if (current.textContent === text) return
      const nextIndex = label.active === 0 ? 1 : 0
      const next = label.nodes[nextIndex]
      next.textContent = text
      current.classList.remove('tri-radar-ax--active')
      next.classList.add('tri-radar-ax--active')
      label.active = nextIndex
    })
    applyAxisClasses()
  }
  const rerender = (): void => {
    revealed = true
    syncChrome()
    morphAll(!reduced)
    renderDev(reduced ? 'none' : 'animate')
  }
  const persistSelection = (): void => {
    try {
      localStorage.setItem(
        TRI_ABILITIES_SELECTION_KEY,
        JSON.stringify({ average: avg, sports: [...pressed] }),
      )
    } catch {}
  }
  const toggleSport = (sport: Sport): void => {
    applyModel(updateAbilities(model, { type: 'toggle-sport', sport }, availableSports))
    persistSelection()
    rerender()
  }
  const toggleAvg = (): void => {
    applyModel(updateAbilities(model, { type: 'toggle-average' }, availableSports))
    persistSelection()
    rerender()
  }
  const restoreSelection = (): void => {
    try {
      const stored: unknown = JSON.parse(
        localStorage.getItem(TRI_ABILITIES_SELECTION_KEY) ?? 'null',
      )
      if (!isRecord(stored) || typeof stored.average !== 'boolean' || !Array.isArray(stored.sports))
        return
      const storedSports = stored.sports.filter(
        (sport): sport is Sport =>
          (sport === 'swim' || sport === 'bike' || sport === 'run') &&
          sports.some(available => available.sport === sport),
      )
      const restoredSports =
        stored.sports.length === 0 || storedSports.length > 0 ? storedSports : model.sports
      applyModel(
        updateAbilities(
          model,
          { type: 'restore', model: { average: stored.average, sports: restoredSports } },
          availableSports,
        ),
      )
    } catch {}
  }
  const onTabsClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const tab = target.closest<HTMLButtonElement>('.tri-radar-sport')
    if (!tab || !tabs.contains(tab)) return
    const sport = tab.dataset.sport
    if (sport === 'swim' || sport === 'bike' || sport === 'run') toggleSport(sport)
    else if (tab === avgTab) toggleAvg()
  }
  const onDevMove = (event: MouseEvent): void => {
    if (event.target instanceof Element && event.target.closest('.tri-dev-svg')) devFocus?.(event)
  }
  const onDevLeave = (event: MouseEvent): void => {
    if (event.target instanceof Element && event.target.closest('.tri-dev-svg')) devLeave?.()
  }
  const onDevClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const item = target.closest<HTMLElement>('.tri-dev-leg[data-dev-key]')
    const key = item?.dataset.devKey
    if (item && key && devBox.contains(item)) devToggle?.(key, item)
  }
  const onDevTransitionEnd = (event: TransitionEvent): void => {
    if (
      event.propertyName === 'opacity' &&
      event.target instanceof HTMLElement &&
      event.target.classList.contains('tri-dev--leaving')
    )
      event.target.remove()
  }

  syncChrome()
  renderDev('defer')
  apply()

  return {
    element: block,
    mount: () => {
      reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      restoreSelection()
      syncChrome()
      renderDev(reduced ? 'none' : 'defer')
      tabs.addEventListener('click', onTabsClick)
      devBox.addEventListener('mousemove', onDevMove)
      devBox.addEventListener('mouseleave', onDevLeave, true)
      devBox.addEventListener('click', onDevClick)
      devBox.addEventListener('transitionend', onDevTransitionEnd)
      let observer: IntersectionObserver | null = null
      if (reduced) morphAll(false)
      else {
        apply()
        observer = new IntersectionObserver(
          entries => {
            if (!entries.some(entry => entry.isIntersecting)) return
            observer?.disconnect()
            observer = null
            if (revealed) return
            revealed = true
            morphAll(true)
            revealDev?.()
            revealDev = null
          },
          { threshold: 0.15 },
        )
        observer.observe(block)
      }
      return () => {
        tabs.removeEventListener('click', onTabsClick)
        devBox.removeEventListener('mousemove', onDevMove)
        devBox.removeEventListener('mouseleave', onDevLeave, true)
        devBox.removeEventListener('click', onDevClick)
        devBox.removeEventListener('transitionend', onDevTransitionEnd)
        observer?.disconnect()
        window.cancelAnimationFrame(raf)
      }
    },
  }
}
