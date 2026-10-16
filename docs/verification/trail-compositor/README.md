# Trail compositor: retained canvas with dirty regions

Task: make the homepage trail opening and the motion after it as smooth as possible (target 120 fps).

## Change

- `scene-compositor.ts` keeps the WebGL canvas as a retained frame. Each frame records what every draw showed: the texture and its upload count, placement, alpha, mask, clip, and the draw below it. Only the regions that changed are cleared and repainted. Each layer that crosses a changed region is drawn once, with its geometry limited to those regions. A frame with nothing to repaint makes no WebGL calls.
- Draws that stay unchanged for 500 ms join cached segments. A segment is a framebuffer texture that holds a run of consecutive draws, composited once. Premultiplied source-over is associative, so the cached run gives the same pixels as the separate draws.
  - A built segment stays while its draws are unchanged. Draws that settle next to it form new segments.
  - A draw that moves again after it joins a segment waits twice as long next time, up to 8 s.
  - Segment textures are limited to 3 canvas areas in total. Rebuilds are limited to 1 canvas area per frame.
  - A resized segment does not count its old texture against the limit.
- Segments hold untinted pixels and receive the daylight tint when they are drawn to the canvas. A tint change repaints the whole canvas from the cached segments, keeps the segments, and keeps the retained draw positions, so the next frame again repaints only changed regions.
  - With a gain above 1, the per-draw `rgb ≤ a` clamp no longer applies inside a segment. With a forced dusk tint (1.05, 0.98, 0.91), at most 58 pixels per frame differ, by at most 7/255. With a neutral tint the result is exact.
- A canvas surface (`<foreignObject>`) can name the rects it repainted in `trail:canopy-paint` (`detail.rects`, in canvas texels). Only those rects are dirtied. Without `detail.rects`, the whole surface is dirtied. Moss sends its clump rects. A moss step dirties about 57k px on average, instead of the whole 1440 × 459 meadow surface, which otherwise covered more than half the canvas and caused a whole repaint on every step.
- The fragment shader gets texture, mask and clip coordinates from `gl_FragCoord` and the inverse placement. With interpolated coordinates, a region draw selected different nearest texels than the full quad when the geometry differed (about 0.3% of pixels flipped). Now both paths sample the same texels.
- While the trail is hidden, the surface releases its segment textures but keeps the retained frame state, because the restored pixels are identical. Reopening does not repaint everything.
- `actor-spacing.ts`: `readActors` matches `[data-trail-collision]` once per frame instead of once per journey call.
- `crown-wind.ts`: the gust step uses the scroll offset instead of `getBoundingClientRect()`, so it does not force layout after other writes.
- `scenery-worker.ts` paints the terraces, hinterland, terrain (ravine, puddles, fords, meadow, wayside) and the mountain backdrop off the main thread. It is emitted as `trail-scenery.worker.js`. It sends each part as soon as it is ready, while the main thread samples routes and plans the forest. If the worker cannot start or fails, `scenery-workers.ts` paints the remaining parts on the main thread when the scene needs them. The foreground beeches stay with the tree workers.

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

How the runs open the trail:

- The homepage keeps the trail closed and generates the scene when it opens, unless the visitor left it open (`garden:trail-open`).
- Runs start with the trail closed and click it open, so click-to-reveal includes generation.
- `--initially-open` stores the open preference and measures from navigation start.

Options and output:

- The verifier checks every frame painted from retained regions, and every whole repaint drawn from cached segments, against a full repaint of the same frame.
- `CHROMIUM=<path>` selects a browser.
- `SOFTWARE_GL=1` forces SwiftShader on machines with no GPU. On a machine with a GPU, leave it unset to measure the real presentation path.
- Results are written to `artifacts/`.

The harness page (`index.html`) contains the homepage trail markup and a short landing. `build.mjs` bundles `frame.ts`, its workers and the homepage script in the same way as `shared-trail-assets.ts`. Workers are served from `static/scripts`; checkouts that embedded worker source still build. The live playhtml room is blocked, so runs do not depend on other visitors.

## Results

Recorded on 2026-10-16 in a 4-core cloud VM with no GPU (Chromium 141, SwiftShader), at 1440×900 and DPR 1. The baseline is `main` at `4ae6bba`.

Steady state, 2.5–5.5 s after the reveal (2 runs):

|                                   | `main`    | Current |
| --------------------------------- | --------- | ------- |
| WebGL calls per frame (median)    | 1250–1482 | 728–809 |
| Draws per frame (median)          | 335–412   | 159–173 |
| Frames rendered in the 3 s window | 12        | 24–31   |

From the click that opens the trail to the end of its reveal (5 runs, median, vsync paced):

|                          | `main`  | Current |
| ------------------------ | ------- | ------- |
| Click → scene prepared   | 3.10 s  | 2.96 s  |
| Long-frame blocking time | 1048 ms | 884 ms  |
| Long frames              | 12      | 9       |

Pixel verification covered open, scroll, theme ×2, resize, close and reopen. The result was 50 checked frames with 0 mismatches and a worst channel difference of 2/255. A run with a continuously changing tint checked 43 frames with 0 mismatches.

In this VM, software GL is the bottleneck. Treat the numbers as relative call load and main-thread blocking, not as expected frame rates on hardware. Repeat the DPR 2 runs on hardware.

To check that the worker paints the same scenery, seed `Math.random`, `crypto.randomUUID` and `Date.now` identically in both builds. With live attributes removed (`data-motion-time`, `style`, `transform`, `data-phase`), the ground base, terrain, hinterland and backdrop markup matches `main`. The only differences are in live attributes, and they also appear between two runs of `main`. With `Worker` blocked, the fallback renders the scene without errors.

Not changed:

- Between the click and the reveal, one frame of 550–700 ms remains in both builds. It contains the first rasterization of every scene texture and the first full layout of the generated SVG, about 270 ms of forced layout. The layout is paid by whichever geometry read comes first, currently `shared-grove`'s `getBBox`, so deferring that read would only move it.
- Chrome 141 does not composite the opening `clip-path: circle()` animation (`compositeFailed`, `unsupportedProperties: ["clip-path"]`). Each opening frame is driven by the main thread and needs a masked pass of the scene layer.
