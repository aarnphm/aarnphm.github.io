# Trail compositor: retained canvas with dirty regions

Task: make the homepage trail opening and the motion after it as smooth as possible (target 120 fps).

## Change

- `scene-compositor.ts` keeps the WebGL canvas as a retained frame. Each frame records what every draw showed (texture and its upload count, placement, alpha, mask, clip, and the draw below it). Only the regions that changed are cleared and repainted. Each layer that crosses a changed region is drawn once, with geometry limited to those regions.
- Draws that stay unchanged for 500 ms join cached segments: framebuffer textures that hold a run of consecutive draws composited once. Premultiplied source-over is associative, so the cached run gives the same pixels as the separate draws. A built segment stays while its draws are unchanged. Draws that settle next to it form new segments. A draw that moves again after it joins a segment waits twice as long next time (up to 8 s). Segment textures are limited to 3 canvas areas in total, and to 1 canvas area of rebuilds per frame.
- The fragment shader gets texture, mask and clip coordinates from `gl_FragCoord` and the inverse placement. Interpolated coordinates selected different nearest texels when a region draw used different geometry than the full quad (about 0.3% of pixels flipped). Now both paths sample the same texels.
- The clock corner is repainted only when the scene drew over it or the corner moved.
- When hidden, the surface still releases its segment textures, but it keeps the retained frame state, because the restored pixels are identical. Reopening does not repaint everything.
- `actor-spacing.ts`: `readActors` matches `[data-trail-collision]` once per frame instead of once per journey call.
- `crown-wind.ts`: the gust step uses the scroll offset instead of `getBoundingClientRect()`, so it does not force layout after other writes.

## Replay

From this directory, with the repository dependencies installed and `playwright-core` available (`pnpm dlx` or a local install):

```sh
node build.mjs site                                  # current checkout
git worktree add /tmp/trail-base <baseline-commit>
node build.mjs site-base --repo /tmp/trail-base      # baseline with the same harness
node serve.mjs site 7375 & node serve.mjs site-base 7376 &
node measure.mjs gl --port 7376 --label base
node measure.mjs gl --port 7375 --label current
node measure.mjs frames --port 7375 --label current
xvfb-run -a -s "-screen 0 1920x1080x24" node measure.mjs presented --port 7375 --label current
node build.mjs site-verify --verify && node serve.mjs site-verify 7377 &
node measure.mjs verify --port 7377 --label current
```

`CHROMIUM=<path>` selects a browser. `SOFTWARE_GL=1` forces SwiftShader on machines with no GPU. Results are written to `artifacts/`. On a machine with a GPU, leave `SOFTWARE_GL` unset to measure the real presentation path.

The harness page (`index.html`) contains the homepage trail markup with a short landing. `build.mjs` bundles `frame.ts` and the homepage script in the same way as `shared-trail-assets.ts`. The live playhtml room is blocked so that runs do not depend on other visitors.

## Results

Recorded on 2026-10-15 in a 4-core cloud VM with no GPU (Chromium 141, SwiftShader), 1440×900 at DPR 1, baseline `170eafb`. Medians of three runs unless noted.

| Steady state, 2.5–5.5 s after opening                              | Baseline          | Current                                                         |
| ------------------------------------------------------------------ | ----------------- | --------------------------------------------------------------- |
| WebGL calls per frame (median / p95)                               | 3304–4023 / ~4040 | 1596–2156 / 2839–3054                                           |
| Draws per frame (median)                                           | 929–1150          | 328–461                                                         |
| Composited fill per frame, canvas areas                            | 3.9–5.6           | 0.25–0.7 for changed regions (plus segment rebuilds, at most 1) |
| Presented frames per second, Xvfb + SwiftShader                    | 5                 | 8                                                               |
| Pixel verification (open, scroll, theme ×2, resize, close, reopen) | n/a               | 0 mismatched frames; worst channel difference 2/255             |

In this VM, software GL is the bottleneck. Display swaps take 200–400 ms, and uncapped `requestAnimationFrame` runs show multi-second backpressure stalls in both builds. Use these numbers as relative GPU and call load, not as expected frame rates on hardware. At DPR 2, SwiftShader presents about 1 frame per second, so changes accumulate between frames and most frames are near-full repaints. Repeat the DPR 2 runs on hardware.

Not changed:

- Chrome 141 does not composite the opening `clip-path: circle()` animation (`compositeFailed`, `unsupportedProperties: ["clip-path"]`). Each opening frame is driven by the main thread and needs a masked pass of the scene layer. Journeys start after the opening, so the main thread is mostly idle during it.
- Page load still has long tasks while the hidden scene is generated: about 230 ms to evaluate the frame script, generation slices of 130–180 ms (terrain, backdrop, wayside, repoussoir), and one forced layout of the generated SVG (about 300 ms in this VM).
