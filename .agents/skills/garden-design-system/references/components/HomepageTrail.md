# HomepageTrail

The shared walking trail on the home page. A row at the top right holds the local time, the visitor count, the moon calendar and the trail toggle. The toggle opens a pixel landscape of three paths to a campfire around the home text.

**Source:** `quartz/components/HomepageTrail.tsx` · script: `quartz/components/scripts/homepage-trail.inline.ts` · scene: `quartz/components/visitor-trail/` (`frame.ts` and `frame/`, `scene-compositor.ts` and `scene-compositor/`, `visitor-register.ts`, `season-toggle.ts`, `scene-controls.ts`) · styles: `quartz/components/styles/homepage-trail.scss`, `shared-trail-frame.scss` and the `trail-*.scss` modules (weather, lighting, moss, fauna, holidays, scene controls and the rest) · notes: `docs/visitor-weather.md`, `docs/visitor-trail-performance.md`

**Debt:** the accessible names are capitalised ("Walking trail", "Shared trail to the campfire", "Moon calendar"). UI names are lowercase elsewhere.

## Preview

A copy of the server-rendered markup from `/` (selector `.homepage-trail-toggle`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static: no landscape, no walking visitors and no fire animation, because the scene script and its playhtml connection are left out. For the panel that opens from the scene, see TrailSettings.

## Rules

- The toggle is a stroked trail icon with a 44px hit target, fixed at the top right. It sets `aria-expanded` and swaps its title between show and hide. The scene opens as a circle that grows from the toggle and sets `data-trail-open` on the body.
- The landscape is a retained SVG model painted as pixel art. Season, holiday, weather, light and the moon phase come from the visitor's place and clock.
- Visitors walk the paths. The count opens a visitor register (arrivals and departures, in a popover). The moon button opens a month calendar.
- The boulder in the scene opens the settings panel (see TrailSettings). Escape closes a popover first, then hides the scene and returns focus to the toggle.
- Text and links under the scene stay selectable; the overlay clips its own hit area around them.
- Under reduced motion the flame is one static frame and the embers, smoke and toast stop.

## Holidays

The visitor's calendar picks a holiday (`quartz/components/visitor-trail/holidays.ts`); the scene carries it as `data-holiday`. Props are pixel sprites from `holiday-sprites.ts` and `holiday-sprites/` in the holiday palette (one class per tone, `trail-holiday-ink` to `trail-holiday-iron`, each with a fill fallback), placed once per layout by the prop placer in `holidays/props.ts`: each prop names an anchor and a clearance radius, and the placer searches outward from the anchor and skips a prop that finds no room. Nothing placed moves.

- Christmas hangs ornaments and tapers on the ten largest meadow crowns (`holidays/tree-ornaments.ts`).
- Halloween (`holidays/halloween.ts`) is a Victorian penny-dreadful register, drawn in 2px cells with four tones of its own: `h` mourning black, `n` wrought iron, `k` blood and `l` lamp glass, which is dark by day and lit under `data-lighting="moonlit"` with a glow underneath. The ten largest crowns take cobwebs; bats leave the first four at dusk and ravens the gallows. Camps get a pair of jack-o'-lanterns and a ghost or skeleton past the fire. Gallows trees stand in the upper meadow at 3px cells, the size of the living trees. One wayside piece stands per stretch of the lower trail: a gibbet, a pumpkin patch, a graveyard, and on scenes 700px or wider a guillotine and a hearse. The graveyard grows a plot: a mausoleum, iron railings, a mourner, candles, the resurrection men at work and two more ghosts. Gaslamps line the lower trail (four on wide scenes, two otherwise), a scarecrow stands in the pasture with a line of crows beside it, and a coffin leans on every other webbed trunk.
- A sprite's size is fixed in cells, so the scene gains props with width instead of scaling them. The ghosts are the only Halloween pieces that move (a 7s float), and only under `prefers-reduced-motion: no-preference`; the bats, the lamp glow and everything else are still frames. The float pauses with the weather motion.
