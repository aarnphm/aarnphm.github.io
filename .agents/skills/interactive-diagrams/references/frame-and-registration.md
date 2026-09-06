## mdx authoring: zoom + the 1px boundary

a figure is consumed from a note through a fenced `jsx imports={…}` block. wrap every interactive or large figure in `<Zoomable label="…">` so it gets the expand/collapse trigger:

````md
```jsx imports={Zoomable,KVHeadGrouping}
<Zoomable label="KV head grouping">
  <KVHeadGrouping
    caption="Slide the group ratio to watch KV boxes merge and decode cache reads shrink."
    heads={8}
  />
</Zoomable>
```
````

the `imports={A,B}` list is the comma-separated set of registered component names the fence may reference; `Zoomable` goes first, the figure second. props are passed inline (`caption`, plus whatever the component declares — `heads`, `tiles`, …). a static one-off SVG can skip `Zoomable`, but anything with sliders/tabs/many cells wants the zoom panel.

**the boundary is one 1px line, never a shadow.** `figure-frame` paints the resting frame: `border: 1px solid var(--fig-frame-border)`, `border-radius: var(--radius-none)` (square corners), no `box-shadow`. every card / readout / decomp panel _inside_ the figure reuses the **same** `1px solid var(--fig-frame-border)` + `border-radius: var(--radius-none)` + `box-shadow: none`, so nested boxes read as one family (CascadeFilter's `.cf-card`/`.cf-decomp`, GQA's `.kvg-readout`). this is also the repo rule — no `box-shadow`, no `border-left`. when `Zoomable` enters the zoomed state, `zoomable.scss` strips the figure's own border and the panel supplies the frame, so the zoom-restore re-asserts **padding + background only** (see **never paint a border on zoom**).

## the shared grammar (figure.scss)

`figure.scss` is a mixin-only partial (mirrors `quartz/styles/mixin.scss`: no underscore, emits nothing on its own, never imported by a `.tsx`). every figure stylesheet does:

```scss
@use '../../styles/variables.scss' as *;
@use '../../styles/mixin.scss' as *;
@use './figure.scss' as *;
```

### mixins

| mixin                                              | use on                                                            | emits                                                                                                                                                                                                                                                                                                                                         |
| -------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `figure-frame($container, $intrinsic, $bg, $zoom)` | the root `.my-figure` block                                       | container-type/name, margin, `content-visibility:auto`, contain-intrinsic-size, box-sizing, padding, radius, 1px `--fig-frame-border`, bg, color; the `--fig-*` palette; the dark `--fig-frame-border` override; and (when `$zoom: true`, default) a **borderless** `.zoomable.is-zoomed > .zoomable-content > &` restore (padding + bg only) |
| `figure-caption`                                   | `.my-figure-caption` (the `<figcaption>`)                         | margin-top, `--fig-caption-fg` color, font-size .78rem, line-height, text-align left, text-wrap pretty                                                                                                                                                                                                                                        |
| `figure-caption-math-gap`                          | `.my-figure-math` **nested inside** the caption                   | `margin: 0 0.22em; vertical-align: -0.03em` (inline-math spacing in caption prose)                                                                                                                                                                                                                                                            |
| `figure-inline-math($katex)`                       | standalone `.my-figure-math`                                      | `display:inline-block; color:inherit; .katex{color:inherit; font-size:$katex}` (default `.95em`)                                                                                                                                                                                                                                              |
| `figure-fo`                                        | `.my-figure-fo` (a `foreignObject > div` holding KaTeX in an SVG) | width/height 100%, flex, `align-items:center`, line-height, `color:var(--dark)`, `.katex{color:inherit;font-size:1em}`, `.katex-display{margin:0}`                                                                                                                                                                                            |
| `figure-card-heading($size)`                       | an `h4` inside a sidebar card                                     | margin, font-size `$size` (default .72rem), 600, `--fig-note` color. **sentence-case, not uppercase** — older components that hard-coded `text-transform:uppercase` on their headings were converted to match GQA/MLA; don't re-add it                                                                                                        |

### the parameters that vary per component

- `$container` — the `container-name`, always `'<prefix>-figure'` (e.g. `'mla-figure'`). the `@container <prefix>-figure (…)` queries you write for responsive reshaping must use the same name.
- `$intrinsic` — `contain-intrinsic-size` height hint (`22rem`–`42rem`). eyeball the rendered height.
- `$bg` — defaults to `color-mix(in srgb, var(--light) 94%, transparent)`. pass `transparent` for a frameless figure (KVCacheVariants does).
- `$zoom` — defaults `true`. pass `false` when the component needs a **custom** zoom-restore (CascadeFilter regrids to `min(100%,1100px)`; MLALatentPath also restyles `.mla-stage`/`.mla-graph`) and write the custom `.zoomable.is-zoomed > .zoomable-content > .my-figure { … }` yourself.

### never paint a border on zoom

when a `<figure>` is zoomed, `zoomable.scss` strips it (`> figure { border:0; padding:0; background:light }`) and the `.zoomable.is-zoomed` panel itself supplies the frame (an inset box-shadow). so the zoom-restore re-asserts **padding + background only, never a border** — a border there double-frames the figure inside the panel. `figure-frame`'s restore is already borderless; any custom zoom block you write (cf/mla) must be too, and only override what the layout needs (a grid root, an inner stage, a width clamp).

### important: `figure-fo` omits `justify-content`

the generic `Fo` helper components render a bare `class="my-figure-fo"` when no modifier is passed, so a base `justify-content` would silently shift them. the mixin only sets `align-items: center`. add `justify-content: center;` at the call site for centered labels, or let `.my-figure-fo--tau` / `--start` / `--end` modifiers set it.

## canonical new component (SCSS)

```scss
@use '../../styles/variables.scss' as *;
@use '../../styles/mixin.scss' as *;
@use './figure.scss' as *;

.sliding-window {
  @include figure-frame('swm-figure', 30rem);
  --swm-mask: var(--fig-salmon);
  --swm-keep-bg: color-mix(in srgb, var(--fig-salmon) 28%, var(--light));
}

:root[saved-theme='dark'] .sliding-window {
  --swm-keep-bg: color-mix(in srgb, var(--fig-salmon) 22%, var(--lightgray));
}

.swm-graph .swm-fo {
  @include figure-fo;
  justify-content: center;
  overflow: visible;
  pointer-events: none;
}

.swm-math {
  @include figure-inline-math;
}

.swm-caption {
  @include figure-caption;
  .swm-math {
    @include figure-caption-math-gap;
  }
}
```

Use `@container swm-figure (max-width: …)` to reshape descendant grids. Apply narrow-layout padding changes to a descendant; the query container cannot query its own size.

**don't put the responsive grid on the figure root.** the root _is_ the `swm-figure` container, and an element can't `@container`-query its own size — `@container swm-figure { .swm-figure { grid-template-columns: 1fr } }` silently never matches (it stays 2-col at every width, squeezing a column to nothing). put the 2-col grid on an inner `.swm-stage` wrapper and reshape _that_ (KVHeadGrouping, MLA); or, when the content genuinely doesn't balance in two columns, make the figure `display: flex; flex-direction: column` and let panels stack/wrap (RazorEvictor). the same trap silently drops a `.swm-figure { padding: 0 }` self-override at a breakpoint — move padding overrides onto a descendant, or accept the frame padding.

## tokens & palette

- accents are Flexoki: `--fig-salmon: #fdb2a2`, `--fig-sage: #cdd597`. reuse them via `var(--fig-salmon)`; derive fills with `color-mix(in srgb, var(--fig-salmon) 28%, var(--light))` and edges with `… 90%, var(--dark)`.
- neutrals the frame already defines on the root (inherited by descendants): `--fig-stroke`, `--fig-stroke-soft`, `--fig-div`, `--fig-note`, `--fig-line`, `--fig-panel-shadow`, `--fig-surface`, `--fig-caption-fg`.
- component-unique colors get a short `--<prefix>-…` token. only add a `:root[saved-theme='dark']` block for the tokens that actually change in dark — **never** redefine `--fig-frame-border`, the frame mixin handles it.

## TSX wiring

```tsx
import style from '../styles/slidingWindowMask.scss'
import { MathText } from '../../util/math-text'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

const Impl: QuartzMdxComponent<{ caption?: string }> = ({ caption }) => (
  <figure class="sliding-window" data-sliding-window>
    <svg class="swm-graph" viewBox="0 0 W H" role="img" aria-label="…">
      <foreignObject …>
        <div class="swm-fo swm-fo--label" dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
      </foreignObject>
    </svg>
    {caption ? (
      <figcaption class="swm-caption">
        <MathText text={caption} mathClass="swm-math" />
      </figcaption>
    ) : null}
  </figure>
)

const Component = Impl as QuartzMdxComponent<{ caption?: string }>
Component.css = style
export const SlidingWindowMask = registerMdxComponent('SlidingWindowMask', Component)
```

then add the `import './SlidingWindowMask'` + `export { … }` lines to `quartz/components/mdx/index.ts`.

- KaTeX inside SVG must go through `<foreignObject>` wrapping a `.<prefix>-fo` div (SVG `<text>` can't render KaTeX). render with `katex.renderToString(tex, { …katexOptions, output: 'html', macros: customMacros, throwOnError: false })`.
- caption math uses `<MathText text={caption} mathClass="<prefix>-math" />` — the class is a literal, which is why restyling never needs a TSX edit.
