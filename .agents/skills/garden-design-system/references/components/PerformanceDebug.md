# PerformanceDebug

A frame-timing panel that the trail and the triathlon pages share. It is for reading cost while a scene runs: frame rate, frame interval, main-thread work per frame, and where the worst frame went.

**Source:** `quartz/components/scripts/performance-debug.ts` (`setupPerformanceDebug(profile)`) and `scripts/performance-debug/` (`panel.ts` the DOM and chart, `sampling.ts` the frame and event probes, `profile.ts` the page profile contract, `presentation.ts` number formats) · styles: `quartz/components/styles/performance-debug.scss` · trail profile: `quartz/components/scripts/homepage-trail.inline.ts` (`trailPerformanceProfile`) and `quartz/components/visitor-trail/performance-probe.ts`

## Opening

- `⌥⇧D` (`Alt`+`Shift`+`KeyD`, matched by `event.code` so the layout does not matter) toggles the panel on any page that sets it up.
- `?trail-debug=performance` on `/` opens it at load.
- The close button and the shortcut both close it. The panel is an `aside`; it does not take focus or trap it.

## Preview

A copy of the live panel from `/` with the trail open (selector `.site-perf-debug`). The preview drops the fixed position. The chart is a `canvas` on the site; the preview shows a still of it for each theme (`img[data-ds-theme]`, swapped on `saved-theme`), read from the canvas in the in-app browser. The stills come from a run with synthetic main-thread work (6 to 20 ms every 290 ms) so that the spread band and the long-frame ticks show. The dark still comes from `/triathlon`, because the home page follows the sky and stays light in the day. The chart and the numbers therefore do not describe the same six seconds.

## Parts

| Part        | Content                                                                                                                                                              |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Head        | `performance`, the shortcut `⌥⇧D`, a 20px close button                                                                                                               |
| Primary     | frames per second at 1.15rem, and the profile's state (`trail`, `closed`, a triathlon view)                                                                          |
| Chart       | 5rem high. Orange line: frame interval, with a salmon band for its spread. Sage area: main-thread work. Salmon ticks: long frames. Dashed rules at 15, 30 and 60 fps |
| Legend      | interval, work, long frame, drawn with the same marks as the chart                                                                                                   |
| Metrics     | two columns: frame p95, dropped, work p95, rAF p95, render p95, long frames, worst frame, blocked, shift, scroll, cursor p95, popover p95, heap, DOM nodes           |
| Readings    | the profile's own probes. The trail reports scene nodes, animations, canvases and scene writes                                                                       |
| Attribution | worst script and worst event, cut with an ellipsis                                                                                                                   |

## Rules

- The panel uses the figure palette (`--fig-*`) and one accent, `--perf-accent: #fc4c02`. Work is `--fig-sage`, long frames `--fig-salmon`.
- Every number is tabular and right-aligned in its row, so values do not move while they update.
- The shortcut is set in `system-ui` with no border. The mono face draws `⌥` and `⇧` from a fallback font at a different size; one system face keeps the three glyphs even. The global `kbd` rule in `base.scss` has the same problem.
- Labels that do not fit end in an ellipsis; values never wrap.
- A page adds readings through its profile. The panel owns the frame metrics; a profile does not restate them.
- The chart reads its colours from computed style on `themechange`, and again on `transitionend` of any ancestor, because the home trail fades its body tokens over 1.2 s after a theme change.
- With `prefers-reduced-motion: reduce` the panel has no transitions.
