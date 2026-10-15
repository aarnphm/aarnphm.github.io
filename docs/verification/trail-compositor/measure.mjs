// Measures the homepage trail opening and its motion in the harness page.
// Usage: node measure.mjs <frames|gl|presented|verify> [--runs 3] [--dpr 1] [--port 7373] [--out dir]
//   frames     uncapped requestAnimationFrame throughput through the reveal and after it
//   gl         WebGL calls and draws per frame, 2.5s..5.5s after the click
//   presented  frames the display compositor drew (Display::DrawAndSwap); needs a display (xvfb-run)
//   verify     E2E scenario on a --verify build; every region-painted frame must match a full repaint
// CHROMIUM=<path> selects the browser; SOFTWARE_GL=1 forces SwiftShader for GPU-less machines.
// The homepage opens the trail unless a visitor closed it. Runs start with it closed, so the
// click opens the prepared scene; --initially-open keeps a first visit's open-on-load path.
import { mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const here = path.dirname(fileURLToPath(import.meta.url))
const [mode = 'frames', ...rest] = process.argv.slice(2)
const option = (name, fallback) => {
  const index = rest.indexOf(`--${name}`)
  return index >= 0 ? rest[index + 1] : fallback
}
const runs = Number(option('runs', mode === 'verify' ? 1 : 3))
const dpr = Number(option('dpr', 1))
const url = `http://localhost:${option('port', 7373)}/index.html`
const out = path.resolve(option('out', path.join(here, 'artifacts')))
const label = option('label', 'run')
const initiallyOpen = rest.includes('--initially-open')
const flags = [
  ...(process.env.SOFTWARE_GL
    ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist']
    : []),
  ...(mode === 'frames' ? ['--disable-frame-rate-limit', '--disable-gpu-vsync'] : []),
  ...(mode === 'presented'
    ? ['--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding']
    : []),
]
const browser = await chromium.launch({
  headless: mode !== 'presented',
  executablePath: process.env.CHROMIUM,
  args: flags,
})
const median = values => [...values].sort((a, b) => a - b)[values.length >> 1]
const percentile = (values, share) =>
  [...values].sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(values.length * share))]
const intervals = times => times.slice(1).map((time, index) => time - times[index])

async function open() {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: dpr,
  })
  const page = await context.newPage()
  // The shared room is a live service; the harness measures the scene without it.
  await page.route(/playhtml\.fun/, route => route.abort())
  if (!initiallyOpen)
    await page.addInitScript(() => localStorage.setItem('garden:trail-open', 'false'))
  return { context, page }
}
const ready = page =>
  page.waitForFunction(
    () => {
      const data = document
        .querySelector('.homepage-trail-scene')
        ?.shadowRoot?.querySelector('.trail-scene')?.dataset
      return data?.trailRendered === 'true' && data.trailPaintReady === 'true'
    },
    null,
    { timeout: 120000, polling: 100 },
  )
const toggle = page => page.click('.homepage-trail-toggle')

async function frames() {
  const { context, page } = await open()
  await page.addInitScript(() => {
    const record = (window.__trailFrames = [])
    const loop = time => {
      const data = document
        .querySelector('.homepage-trail-scene')
        ?.shadowRoot?.querySelector('.trail-scene')?.dataset
      record.push([time, data?.trailReveal ?? ''])
      requestAnimationFrame(loop)
    }
    requestAnimationFrame(loop)
    document.addEventListener('DOMContentLoaded', () =>
      document
        .querySelector('.homepage-trail-scene')
        ?.addEventListener('trail:reveal-end', () => (window.__trailRevealEnd = performance.now())),
    )
  })
  await page.goto(url)
  // An open-on-load visit counts from navigation start; otherwise from the click.
  let clicked = 0
  if (!initiallyOpen) {
    await ready(page)
    clicked = await page.evaluate(() => {
      window.__trailFrames.length = 0
      return performance.now()
    })
    await toggle(page)
  }
  await page.waitForFunction(() => window.__trailRevealEnd, null, { timeout: 60000 })
  await page.waitForTimeout(6000)
  const { record, end } = await page.evaluate(() => ({
    record: window.__trailFrames,
    end: window.__trailRevealEnd,
  }))
  await context.close()
  const reveal = intervals(
    record.filter(([, phase]) => phase === 'revealing').map(([time]) => time),
  )
  const settle = intervals(
    record.filter(([time]) => time > end && time < end + 2000).map(([t]) => t),
  )
  const steady = intervals(record.filter(([time]) => time > end + 2000).map(([t]) => t))
  const summary = values => ({
    fps: 1000 / (values.reduce((a, b) => a + b, 0) / values.length),
    p95: percentile(values, 0.95),
    max: Math.max(...values),
  })
  const first = record.find(([, phase]) => phase === 'revealing')?.[0] ?? NaN
  return {
    clickToReveal: first - clicked,
    reveal: summary(reveal),
    settle: summary(settle),
    steady: summary(steady),
  }
}

async function gl() {
  const { context, page } = await open()
  await page.addInitScript(() => {
    const counts = (window.__trailGl = { calls: 0, draws: 0, on: false, frames: [] })
    const prototype = WebGL2RenderingContext.prototype
    for (const name of Object.getOwnPropertyNames(prototype)) {
      const original = Object.getOwnPropertyDescriptor(prototype, name)?.value
      if (typeof original !== 'function' || name === 'constructor') continue
      prototype[name] = function (...args) {
        counts.calls++
        if (name === 'drawArrays') counts.draws++
        return original.apply(this, args)
      }
    }
    const loop = () => {
      if (counts.on && counts.calls) counts.frames.push([counts.calls, counts.draws])
      counts.calls = counts.draws = 0
      requestAnimationFrame(loop)
    }
    requestAnimationFrame(loop)
  })
  await page.goto(url)
  await ready(page)
  if (!initiallyOpen) await toggle(page)
  await page.waitForTimeout(2500)
  await page.evaluate(() => (window.__trailGl.on = true))
  await page.waitForTimeout(3000)
  const counted = await page.evaluate(() => window.__trailGl.frames)
  await context.close()
  const calls = counted.map(([value]) => value)
  const draws = counted.map(([, value]) => value)
  return {
    frames: counted.length,
    calls: median(calls),
    callsP95: percentile(calls, 0.95),
    draws: median(draws),
    drawsP95: percentile(draws, 0.95),
  }
}

async function presented() {
  const { context, page } = await open()
  await page.goto(url)
  await ready(page)
  await page.waitForTimeout(500)
  await browser.startTracing(page, { categories: ['viz', 'blink.user_timing'] })
  await page.evaluate(() => {
    performance.mark('trail-click')
    document
      .querySelector('.homepage-trail-scene')
      .addEventListener('trail:reveal-end', () => performance.mark('trail-reveal-end'))
  })
  await toggle(page)
  await page.waitForTimeout(6500)
  const { traceEvents } = JSON.parse((await browser.stopTracing()).toString())
  await context.close()
  const mark = name => traceEvents.find(event => event.name === name)?.ts
  const click = mark('trail-click')
  const end = mark('trail-reveal-end') ?? click + 600000
  const swaps = traceEvents
    .filter(event => event.name === 'Display::DrawAndSwap' && event.ph === 'X')
    .map(event => event.ts + event.dur)
    .sort((a, b) => a - b)
  const span = (from, to) => {
    const times = swaps.filter(time => time >= from && time < to)
    const gaps = intervals(times).map(gap => gap / 1000)
    return { fps: times.length / ((to - from) / 1e6), p95: percentile(gaps, 0.95) ?? NaN }
  }
  return {
    reveal: span(click, end),
    settle: span(end, end + 2e6),
    steady: span(end + 2e6, end + 5.5e6),
  }
}

async function verify() {
  const { context, page } = await open()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const take = () =>
    page.evaluate(() => {
      const view = document.querySelector('iframe[data-trail-frame]').contentWindow
      const frames = view.__trailVerified ?? []
      view.__trailVerified = []
      return {
        frames: frames.length,
        mismatched: frames.filter(frame => frame.differing).length,
        worst: Math.max(0, ...frames.map(frame => frame.worst)),
        pixels: Math.max(0, ...frames.map(frame => frame.differing)),
      }
    })
  await page.goto(url)
  await ready(page)
  // Let the hidden surface trim and restore its pixels before opening.
  await page.waitForTimeout(1500)
  await page.evaluate(() => {
    document.querySelector('iframe[data-trail-frame]').contentWindow.__trailVerify = true
  })
  const theme = value =>
    page.evaluate(value => {
      document.documentElement.setAttribute('saved-theme', value)
      document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: value } }))
    }, value)
  const steps = [
    ['open', () => (initiallyOpen ? null : toggle(page)), 3500],
    ['scroll', () => page.mouse.wheel(0, 180), 2000],
    ['scroll back', () => page.mouse.wheel(0, -180), 2000],
    ['dark theme', () => theme('dark'), 3500],
    ['light theme', () => theme('light'), 3500],
    ['resize', () => page.setViewportSize({ width: 1200, height: 820 }), 4000],
    ['close', () => toggle(page), 1500],
    ['reopen', () => toggle(page), 3500],
  ]
  const results = []
  for (const [name, action, wait] of steps) {
    await action()
    await page.waitForTimeout(wait)
    results.push({ step: name, ...(await take()) })
  }
  mkdirSync(out, { recursive: true })
  await page.screenshot({ path: path.join(out, `verify-dpr${dpr}.png`) })
  await context.close()
  return { steps: results, errors, pass: !errors.length && results.every(step => !step.mismatched) }
}

const measure = { frames, gl, presented, verify }[mode]
if (!measure) throw new Error(`Unknown mode ${mode}`)
const results = []
for (let run = 0; run < runs; run++) {
  const result = await measure()
  results.push(result)
  console.log(
    JSON.stringify(result, (_, value) =>
      typeof value === 'number' ? Math.round(value * 10) / 10 : value,
    ),
  )
}
await browser.close()
mkdirSync(out, { recursive: true })
const file = path.join(out, `${mode}-${label}-dpr${dpr}.json`)
writeFileSync(
  file,
  JSON.stringify(
    { mode, label, dpr, url, flags, runs: results, recorded: new Date().toISOString() },
    null,
    2,
  ),
)
console.log(`wrote ${file}`)
