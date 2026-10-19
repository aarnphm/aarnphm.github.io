// Replay: node docs/verification/fsm-decoding/replay.mjs
// Renders the five FSMDecoding embeds of the notes, walks every frame of every mode with clicks
// and keys, and checks the drawn mask against an oracle built from the regex language, not from
// the component.
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  checker,
  openFigurePage,
  renderFigurePage,
  repo,
  standardChecks,
} from '../figure-harness.mjs'
import { renderNotes } from './pipeline.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const out = path.join(here, 'artifacts')

// The embeds exactly as the notes write them: four in the thoughts note, one in the post.
const notes = ['content/thoughts/structured outputs.md', 'content/posts/structured decoding.md']
const instances = []
for (const note of notes) {
  const text = await readFile(path.join(repo, note), 'utf8')
  for (const [tag] of text.matchAll(/<FSMDecoding\b[^>]*\/>/g)) {
    const props = {}
    for (const [, k, v] of tag.matchAll(/(\w+)="([^"]*)"/g)) props[k] = v
    instances.push(props)
  }
}

// Oracles. The JSON language is finite; the number language is prefix-closed, so a prefix is
// viable exactly when it matches the whole regex.
const JSON_LANG = ['Paul', 'John'].flatMap(n => ['20', '30'].map(a => `{"name":"${n}","age":${a}}`))
const NUMBER_RE = /^([0-9]*)?\.?[0-9]*$/
const oracle = {
  json: { viable: s => JSON_LANG.some(w => w.startsWith(s)), complete: s => JSON_LANG.includes(s) },
  number: { viable: s => NUMBER_RE.test(s), complete: s => NUMBER_RE.test(s) },
}

const pagePath = await renderFigurePage({ component: 'FSMDecoding', instances, out })
const { browser, page, errors } = await openFigurePage(pagePath)
const c = checker(out)
// The same embeds through Quartz's Codeblock transformer, the note entry point.
const viaPipeline = (await renderNotes(notes))
  .flatMap(h => [...h.matchAll(/<figure class="fsm-decoding"[^>]*>/g)])
  .map(([t]) => `${t.match(/data-fsm-mode="(\w+)"/)[1]}:${t.match(/data-fsm-stepper="(\w+)"/)[1]}`)
const expected = instances.map(
  p => `${p.mode === 'compressed' ? 'compressed' : 'char'}:${p.view !== 'graph'}`,
)
c.check(
  'Codeblock transformer renders every embed with its props',
  viaPipeline.join(',') === expected.join(','),
  { viaPipeline, expected },
)

const figs = page.locator('figure.fsm-decoding')
c.check('five embeds rendered', (await figs.count()) === 5, instances)

const ids = await figs.evaluateAll(fs =>
  fs.map(f => f.querySelector('input[type=radio]')?.getAttribute('name') ?? null),
)
c.check(
  'radio names are unique per instance',
  new Set(ids.filter(Boolean)).size === ids.filter(Boolean).length,
  ids,
)

// Reads the visible state of one figure.
const snapshot = fig =>
  fig.evaluate(root => {
    const mode = root.getAttribute('data-fsm-mode')
    const visible = el => el.getClientRects().length > 0
    const graphs = [...root.querySelectorAll(`[data-fsm-graph="${mode}"] svg`)]
    const tape = root.querySelector(`[data-fsm-tape-mode="${mode}"]`)
    return {
      mode,
      graphs: graphs.map(svg => ({
        shown: visible(svg),
        n: [...svg.querySelectorAll('[data-fsm-node]')]
          .map(n => n.getAttribute('data-st'))
          .join(''),
        e: [...svg.querySelectorAll('[data-fsm-edge]')]
          .map(n => n.getAttribute('data-st'))
          .join(''),
        current: [...svg.querySelectorAll('[data-fsm-node][data-st="c"]')].map(n =>
          n.textContent.trim(),
        ),
      })),
      mask: [...root.querySelectorAll('[data-fsm-tok]')].map(t => ({
        tok: t.querySelector('.fsm-tok-text').textContent,
        st: t.getAttribute('data-st'),
        to: t.querySelector('.fsm-tok-to').textContent,
      })),
      tape: tape
        ? [...tape.querySelectorAll('[data-fsm-tape]')].map(t => ({
            tok: t.querySelector('code').textContent,
            tag: t.querySelector('span').textContent,
            st: t.getAttribute('data-st'),
          }))
        : [],
      action: [...root.querySelectorAll('[data-fsm-act]')]
        .filter(p => !p.hidden)
        .map(p => p.textContent),
      out: Object.fromEntries(
        [...root.querySelectorAll('[data-fsm-out]')].map(d => [
          d.getAttribute('data-fsm-out'),
          d.textContent,
        ]),
      ),
      buttons: Object.fromEntries(
        ['back', 'step', 'reset'].map(b => [b, root.querySelector(`[data-fsm-${b}]`)?.disabled]),
      ),
    }
  })

const framesOf = fig => fig.evaluate(r => JSON.parse(r.getAttribute('data-fsm-frames') ?? 'null'))

async function walk(i, preset, mode, useKeys) {
  const fig = figs.nth(i)
  const frames = (await framesOf(fig))[mode]
  const tag = `#${i} ${mode}`
  const o = oracle[preset]
  let passes = 0
  for (let k = 0; k < frames.length; k++) {
    const s = await snapshot(fig)
    const f = frames[k]
    const label = `${tag} frame ${k}`
    c.check(`${label}: mode attribute`, s.mode === mode, s.mode)
    const shownGraphs = s.graphs.filter(g => g.shown)
    c.check(`${label}: one graph layout visible`, shownGraphs.length === 1, s.graphs.length)
    for (const g of s.graphs) {
      c.check(`${label}: nodes drawn as model`, g.n === f.n, { drawn: g.n, model: f.n })
      c.check(`${label}: edges drawn as model`, g.e === f.e, { drawn: g.e, model: f.e })
      c.check(`${label}: one current state`, g.current.length === 1, g.current)
    }
    c.check(
      `${label}: action names the current state`,
      s.action.length === 1 &&
        s.action[0].toLowerCase().includes(`state ${shownGraphs[0].current[0]}`),
      { action: s.action, current: shownGraphs[0].current },
    )
    // The prefix before this frame's action is the text of the settled tape cells.
    const prefix = s.tape
      .filter(t => t.st === 's')
      .map(t => (t.tok === 'EOS' ? '' : t.tok))
      .join('')
    const lit = s.tape.filter(t => t.st === 'l')
    let maskOk = true
    const bad = []
    for (const m of s.mask) {
      const allowed = m.tok === 'EOS' ? o.complete(prefix) : o.viable(prefix + m.tok)
      const drawnAllowed = m.st === 'a' || m.st === 'p'
      if (allowed !== drawnAllowed) {
        maskOk = false
        bad.push(m)
      }
      if (drawnAllowed !== (m.to !== '')) bad.push({ ...m, why: 'target label' })
    }
    c.check(`${label}: mask equals the regex oracle at "${prefix}"`, maskOk && !bad.length, bad)
    const picks = s.mask.filter(m => m.st === 'p')
    if (f.lit && lit.every(t => t.tag !== 'ff')) {
      passes += 1
      c.check(
        `${label}: sampled token is lit in mask and tape`,
        picks.length === 1 && lit.length === 1 && picks[0].tok === lit[0].tok,
        { picks, lit },
      )
      c.check(`${label}: pass tag`, lit[0].tag === `p${passes}`, lit[0])
    } else {
      c.check(`${label}: no sampled token on a forced or done frame`, picks.length === 0, picks)
    }
    c.check(`${label}: readout passes`, s.out.passes === String(passes), s.out)
    c.check(
      `${label}: readout allowed count`,
      s.out.allowed === `${s.mask.filter(m => m.st !== 'x').length} / ${s.mask.length}`,
      s.out,
    )
    c.check(
      `${label}: back and reset disabled only at start`,
      s.buttons.back === (k === 0) && s.buttons.reset === (k === 0),
      s.buttons,
    )
    c.check(
      `${label}: step disabled only at the end`,
      s.buttons.step === (k === frames.length - 1),
      s.buttons,
    )
    if (k === frames.length - 1) {
      const text = s.tape
        .filter(t => t.st !== 'h')
        .map(t => (t.tok === 'EOS' ? '' : t.tok))
        .join('')
      c.check(`${tag}: final output is in the language`, o.complete(text), text)
      c.check(`${tag}: tape ends with EOS`, s.tape.at(-1).tok === 'EOS', s.tape.at(-1))
      c.check(`${tag}: done readout`, s.out.step === 'done', s.out)
      break
    }
    if (useKeys) {
      await fig.locator('[data-fsm-step]').focus()
      await page.keyboard.press(k % 2 ? 'Space' : 'Enter')
    } else await fig.locator('[data-fsm-step]').click()
  }
  // Focus stays on the stepper when step becomes disabled.
  const focused = await page.evaluate(() => document.activeElement?.hasAttribute('data-fsm-back'))
  c.check(`${tag}: focus moves to back when step disables`, focused, focused)
  await fig.locator('[data-fsm-back]').click()
  const back = await snapshot(fig)
  c.check(
    `${tag}: back returns one frame`,
    back.out.step === `${frames.length - 1} / ${frames.length - 1}`,
    back.out,
  )
  await fig.locator('[data-fsm-reset]').click()
  const reset = await snapshot(fig)
  c.check(
    `${tag}: reset returns to frame 0`,
    reset.out.step === `1 / ${frames.length - 1}`,
    reset.out,
  )
  return passes
}

// Instance 0: click through character mode, then switch with the keyboard and walk jump-forward.
const charPasses = await walk(0, 'json', 'char', false)
await figs.nth(0).locator('input[value="char"]').focus()
await page.keyboard.press('ArrowRight')
c.check(
  '#0 arrow key selects jump-forward',
  (await figs.nth(0).getAttribute('data-fsm-mode')) === 'compressed',
  await figs.nth(0).getAttribute('data-fsm-mode'),
)
const outline = await figs
  .nth(0)
  .locator('.fsm-segment:has(input:focus-visible)')
  .evaluate(el => getComputedStyle(el).outlineStyle)
c.check('#0 focused segment shows an outline', outline === 'solid', outline)
const jfPasses = await walk(0, 'json', 'compressed', true)
c.check('json: 10 passes character, 2 jump-forward', charPasses === 10 && jfPasses === 2, {
  charPasses,
  jfPasses,
})
const totals = await figs.nth(0).locator('.fsm-readout dd:not([data-fsm-out])').allTextContents()
c.check(
  'json: static totals equal walked passes',
  totals.join(',') === `${charPasses},${jfPasses}`,
  totals,
)

// Instance 1 starts in jump-forward mode; walk it with keys, and snap a mid-step state.
await walk(1, 'json', 'compressed', true)
await figs.nth(1).locator('[data-fsm-step]').click()
await c.shoot(figs.nth(1), 'jf-step-2.png')
await figs.nth(1).locator('[data-fsm-reset]').click()
await figs.nth(0).locator('.fsm-segment:has(input[value="char"])').click()
for (let k = 0; k < 3; k++) await figs.nth(0).locator('[data-fsm-step]').click()
await c.shoot(figs.nth(0), 'char-step-4.png')
await figs.nth(0).locator('[data-fsm-reset]').click()

// Instance 2: no stepper; the compressed edges must spell exactly the language.
const rows = await figs
  .nth(2)
  .locator('.fsm-edge-row:not(.fsm-edge-row--head)')
  .evaluateAll(rs => rs.map(r => [...r.children].map(x => x.textContent)))
const expand = l =>
  l.startsWith('[')
    ? l
        .slice(1, l.indexOf(']'))
        .split('')
        .map(ch => ch + l.slice(l.indexOf(']') + 1))
    : [l]
const words = []
const visit = (q, acc) => {
  const outs = rows.filter(r => r[0] === q)
  if (!outs.length) words.push(acc)
  for (const [, l, to] of outs) for (const piece of expand(l)) visit(to, acc + piece)
}
visit('0', '')
c.check(
  'compressed edges spell the language',
  words.sort().join('|') === [...JSON_LANG].sort().join('|'),
  words,
)
c.check(
  'static instance has no stepper',
  (await figs.nth(2).locator('[data-fsm-step]').count()) === 0,
)
await figs.nth(2).locator('.fsm-segment:has(input[value="char"])').click()
c.check(
  'static toggle shows the character FSM',
  await figs.nth(2).locator('[data-fsm-graph="char"] svg').first().isVisible(),
)
await figs.nth(2).locator('.fsm-segment:has(input[value="compressed"])').click()

// Instance 3: number regex, no mode toggle because no state has a single forced path.
c.check(
  'number: no automaton toggle',
  (await figs.nth(3).locator('input[type=radio]').count()) === 0,
)
const numPasses = await walk(3, 'number', 'char', false)
c.check('number: one pass per token', numPasses === 4, numPasses)

// Instance 4: the post's embed, walked with keys.
c.check(
  'post embed is the character FSM',
  (await figs.nth(4).getAttribute('data-fsm-mode')) === 'char',
)
await walk(4, 'json', 'char', true)

await standardChecks(page, errors, c, '.zoomable')
await browser.close()
process.exit((await c.finish()) ? 0 : 1)
