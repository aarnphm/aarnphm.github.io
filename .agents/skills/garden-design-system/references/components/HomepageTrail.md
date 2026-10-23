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
