# Isometric figures with the anatomy kit

Use the `anatomy` skill (`~/.claude/skills/anatomy`) when a concept maps one-to-one onto a physical object: a cache check as a pin-tumbler lock, render targets as a chest of drawers, a separable blur as a bed of nails beside two combs. Read its `SKILL.md` and `references/craft.md` first, and size the work as that skill says (a Figure is 40 to 80 parts). Most notes are better served by a flat figure of the mechanism itself; choose the object only when it explains more than the flat drawing would.

No figure on the site uses the kit yet. The integration below is a starting contract, not a proven one: check every value in the rendered page in both themes before relying on it.

## Placement

- Build the static art at build time in a Preact component under `quartz/components/mdx`, from geometry in a plain `.ts` module next to it. Copy the kit files the figure needs into the component's folder rather than importing from `~/.claude`. Keep live motion in the figure's `.inline.ts`, registered in the `nav` handler.
- Wrap the drawing in the normal figure frame (`figure-frame`). The anatomy card corners (Fig n, title, hint, readout) map to the Garden frame: controls on top, readout card beside or below, caption under the frame.

## Theme mapping

Put `class="iso" data-theme="light"` on the drawing wrapper and override the kit tokens from Garden variables, so the figure follows `saved-theme` without a second theme switch:

```scss
.my-figure .iso {
  --anatomy-page: var(--light);
  --anatomy-card: var(--light);
  --anatomy-paper: var(--light);
  --anatomy-face: var(--light);
  --anatomy-top: var(--light);
  --anatomy-text: var(--darkgray);
  --anatomy-muted: var(--darkgray);
  --anatomy-strong: var(--dark);
  --anatomy-hi: var(--dark);
  --anatomy-mid: var(--darkgray);
  --anatomy-lo: var(--gray);
  --anatomy-faint: color-mix(in srgb, var(--gray) 45%, var(--light));
  --anatomy-dot: var(--dark);
  --anatomy-lit: var(--dark);
  --anatomy-lit-top: color-mix(in srgb, var(--fig-salmon) 40%, var(--light));
  --anatomy-lit-shade: color-mix(in srgb, var(--fig-salmon) 55%, var(--light));
  --anatomy-red: var(--fig-salmon);
  --anatomy-green: var(--fig-sage);
  --anatomy-lift: none;
  --anatomy-mono: var(--codeFont);
}
```

- **Paper equals the frame ground.** Fills hide what is behind them only when `--anatomy-paper` matches the figure background.
- **Shades stay close and get darker away from the light** in both themes. The kit's light ramp (`#dcdcd9` to `#f4f4f1`) is neutral grey; on Garden's cream paper, derive it from `--gray` mixed into `--light` (about 45%, 32%, 20% and 9%) and set a separate dark-theme ramp under `:root[saved-theme='dark']`, where mixing toward `--lightgray` would lighten the far faces.
- **The lit accent** is the salmon top and shade with a `--dark` edge, the same "one thing lit" rule as flat figures. `red` and `green` map to salmon and sage; use them only when colour is the subject.
- **`--anatomy-lift` is a box-shadow** and stays `none`. The halo under the base is part of the drawing's volume; keep its opacity at the kit's light value or lower.

## What carries over unchanged

The kit's painter's order, lines that end on something, textures that end on an edge or fade, true-shape audit (`node build.mjs --audit`) and the line-end check (`scripts/lines.mjs`) all apply. Garden adds its own verification on top: the rendered page from the running dev watcher, both themes, 320px.
