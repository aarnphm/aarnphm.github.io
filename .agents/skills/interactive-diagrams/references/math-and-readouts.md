## metrics, readouts & sliders

a stats panel is a `<dl>` inside one of the bordered cards (`.cf-card`, `.kvg-readout`): same `1px solid var(--fig-frame-border)`, square, `box-shadow: none`. `dt` is the label in `--fig-note`; `dd` is the value in `--dark`. always `font-variant-numeric: tabular-nums` so digits don't reflow as the number ticks. heading is `figure-card-heading` (sentence-case). two shapes exist — match one:

- **CascadeFilter `.cf-stats`** — `grid-template-columns: auto 1fr`, dense rows, values re-rendered as KaTeX every frame.
- **GQA `.kvg-readout-row`** — `grid-template-columns: minmax(0,1fr) auto`, one row per stat.

**every number is KaTeX, including the slider readout.** this supersedes any earlier "the slider chip stays plain text" guidance — CascadeFilter renders `\tau`'s value through KaTeX (`cf-slider-value` → `renderMath(tau.toFixed(2))`), and the stats (`kept`, `speedup`, `recall`) likewise. a row that pairs a KaTeX label with a plain bold number reads as two fonts; render both. escape `%` as `\\%`, `×` as `\\times`, `∞` as `\\infty`. mechanics live in **LaTeX text rendering** below.

**keeping KaTeX values live** — two routes, no `window.katex` needed at runtime:

1. **re-render in the inline script** (continuous values). the script imports its own `katex` and writes `innerHTML` (see CascadeFilter's `cfRenderMath`):
   ```ts
   import katex from 'katex'
   const cfRenderMath = (tex: string): string =>
     katex.renderToString(tex, {
       displayMode: false,
       output: 'html',
       strict: false,
       throwOnError: false,
     })
   // …
   state.statRecall.innerHTML = cfRenderMath(`${Math.round(r * 100)}\\%`)
   ```
2. **pre-render every discrete value in TSX, toggle visibility** (small finite value sets). KVHeadGrouping renders each regime's KaTeX once and shows the active one with `[data-kvg-regime-r][data-kvg-active='false'] { display: none }` — no JS KaTeX at all. use this when the values are a handful of integers/ratios; use route 1 for a continuous slider.

**sliders** use `@include stableRangeControl(...)` (square track + thumb, `:active` thumb `scale(.96)`). drive the track fill from JS by setting a CSS var the gradient reads — these `--<prefix>-*` runtime vars are the only ones JS touches:

```scss
.cf-slider {
  @include stableRangeControl(
    $track-color: linear-gradient(
        to right,
        var(--cf-keep) 0%,
        var(--cf-keep) calc(var(--cf-tau, 0.5) * 100%),
        var(--cf-divider) calc(var(--cf-tau, 0.5) * 100%),
        var(--cf-divider) 100%
      )
  );
}
```

```ts
state.root.style.setProperty('--cf-tau', threshold.toFixed(3))
```

mirror the value into `aria-valuenow`/`aria-valuetext` on every input (CascadeFilter does), and give the readout chip `min-width` so the row doesn't jitter as the value width changes.

## LaTeX text rendering

math is KaTeX everywhere, **including the values in readouts/stats**, not only the formula labels. a row that pairs a KaTeX formula on the left with a plain bold number on the right reads as two different fonts; render both.

- **readout / stat values.** render the initial value through `renderMath` in the TSX, not as a text child:
  ```tsx
  <span
    data-x-mem
    class="x-readout-val"
    dangerouslySetInnerHTML={{ __html: renderMath(`L/${p}\\,d`) }}
  />
  ```
  the inline script updates it with a katex helper writing `innerHTML` (NOT `textContent`):
  ```ts
  import katex from 'katex'
  const xTex = (t: string): string => {
    try {
      return katex.renderToString(t, {
        displayMode: false,
        output: 'html',
        throwOnError: false,
        strict: false,
      })
    } catch {
      return t
    }
  }
  // …in the readout updater:
  el.innerHTML = xTex(`${p - 1}\\,L/${p}\\,d`)
  ```
  then `.x-readout-val .katex { font-size: 1em; color: inherit }` so the numerals sit at the row's size and color. KaTeX ignores `font-weight`, so the value renders CM-serif instead of bold sans — that IS the intended math look. escape `%` as `\\%` (bare `%` is a KaTeX comment), `\\times` for `×`, `\\infty` for `∞`. RingRotation, RazorEvictor, and CascadeFilter readouts do this. **the slider's own value chip is KaTeX too** — CascadeFilter renders `\tau`'s number through `renderMath` and updates it via `el.innerHTML = cfRenderMath(threshold.toFixed(2))`, with `.cf-slider-value .katex { font-size: .95em; color: inherit }` and a `min-width` so the chip doesn't jitter. (an earlier version of this skill said the chip stays plain text — that's superseded; render it as math like every other readout.)
- **non-math labels stay plain text.** request strings and code-ish token names (`#1 chat_a → turn1 + "explain"`, `branch chat_a`) are not math — render them as plain `{label}` children, not `tex('\\text{…}')`. two stacked KaTeX `\text{}` spans that must share a left edge drift by the leading glyph's side-bearing (`#` vs `b`) — a visible alignment bug. keep KaTeX only where it IS math and has no neighbour to align to (a single-line readout echo of the label). RadixPrefixTree's prompt list is plain text for exactly this.
- **`<foreignObject>` clips KaTeX even when the inner div is `overflow:visible`.** the foreignObject _element_ defaults to `overflow:hidden` (UA sheet), so a label wider than its box is shaved at the edge regardless of `.x-fo { overflow: visible }`. KaTeX text can't wrap, so a snug fo always clips. fix once per component: `.x-graph foreignObject { overflow: visible }` (RazorCompression edge labels), and size fos generously.
- **display math clips under the frame.** `figure-frame`'s `content-visibility: auto` implies `contain: paint`, so display math wider than its card is painted-clipped at the figure box. give it `.x-math--display .katex-display { overflow-x: auto; overflow-y: hidden }` (RazorEvictor eviction rule).
- **two labels under adjacent SVG elements that collide:** KaTeX won't wrap, so don't shrink them to fit — anchor them to opposite edges so they diverge into the free space. `justify-content: flex-end` on the left one, `flex-start` on the right one (a `--end` / `--start` fo modifier), each computed to the box's near edge (RazorHeadTaxonomy echo/induction labels).
