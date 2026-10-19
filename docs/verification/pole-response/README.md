# PoleResponse verification

## Change

`PoleResponse` (`quartz/components/mdx/PoleResponse.tsx`, `quartz/components/styles/poleResponse.scss`, `quartz/components/scripts/pole-response.inline.ts`) replaces 14 control-systems images with an s-plane and a unit-step response that share one model, `quartz/components/scripts/pole-response-model.ts`. The server render and the inline script both call `timeView` or `locusView`, so the marks, the readout and the first paint come from the same functions.

- First order: $c(t)=1-e^{-at}$; $\tau$, $T_r=\ln 9/a$, $T_s=-\ln 0.02/a$ and the $4/a$ rule.
- Second order: closed-form step responses for a complex pair, a double pole and two real poles. $T_p=\pi/\omega_d$ and $\%OS=100e^{-\zeta\pi/\sqrt{1-\zeta^2}}$ are closed form; $T_r$ (10% to 90%) and the 2% settling time are measured on $c(t)$, and the $4/\sigma_d$ rule is shown beside them.
- Locks: same envelope (fixed $\sigma_d$), same frequency (fixed $\omega_d$), same overshoot (fixed $\zeta$).
- Root locus: Nise Example 9.1, $G(s)=1/((s+1)(s+2)(s+10))$, with gain only, an integrator, or PI $K(s+0.1)/s$. The branches are roots of the characteristic polynomial (Durand-Kerner, warm-started per gain); the step response uses residues at the closed-loop poles. The model reproduces the textbook values: $K=164.5$ at $\zeta=0.174$ with poles $-0.69\pm j3.93$, $-11.61$, $e(\infty)=0.108$; PI $K=158.1$ with $-0.68\pm j3.84$, $-11.55$, $-0.09$.

## Replay

```sh
ln -sfn <deps>/node_modules node_modules   # once, if node_modules is missing
node docs/verification/pole-response/replay.mjs
node docs/verification/pole-response/notes.mjs
```

`replay.mjs` uses the shared harness `docs/verification/figure-harness.mjs` (Chromium at `/opt/pw-browsers/chromium`, or `CHROMIUM`). It renders the props each note embeds, one page per note, and writes `artifacts/<note>/page.html`, screenshots and `results.json`. `notes.mjs` runs every PoleResponse fence of the four notes through Quartz's `Codeblock` transformer and writes `artifacts/notes.json`.

## Results (2026-10-19)

- `replay.mjs`: system response 103/103, time response 42/42, root locus control 22/22, Z-transform 13/13.
- `notes.mjs`: 78/78.

The replay checks that the handle, the conjugate, the peak dot and the settling dot sit where the readout says, for the SSR state and after every pointer drag, arrow key and radio change; that each lock keeps its quantity; that the course presets give their poles; that the locus dots match the listed closed-loop poles, the design-gain button lands on $\zeta=0.174$, the slider steps by 0.1 and a drag to the $j\omega$ crossing gives $K\approx396$; that instances are independent; that cleanup removes the listeners; and the standard light, dark and 320px screenshots with no page scroll or errors.

## Not checked

- No dev watcher ran in this environment, so the figure was not checked inside a full Quartz page, under SPA navigation, or at 200% and 400% browser zoom.
- Touch input was not exercised; the plane sets `touch-action: none`, so a vertical swipe on it moves the pole instead of the page.
