# HomepageTrail

The shared walking trail on the home page: a toggle in the header and a scene of three paths to a campfire around the home text.

**Source:** `quartz/components/HomepageTrail.tsx` · scene: `quartz/components/visitor-trail/` (`frame.ts`, `layout.ts`, `scenery.ts`, `campsite.ts`) · styles: `quartz/components/styles/homepage-trail.scss`, `quartz/components/styles/shared-trail-frame.scss` · notes: `docs/visitor-weather.md`

**Debt:** the accessible names are capitalised ("Walking trail", "Shared trail to the campfire"). UI names are lowercase elsewhere.

## Preview

A copy of the server-rendered markup from `/` (selector `.homepage-trail-toggle`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. and `.homepage-trail-scene`, with the scene shown. The preview is static: no stones, no walking dots and no fire animation, because the scene script and its playhtml connection are left out.

## Rules

- The toggle is a stroked trail icon with a 44px hit target, fixed at the top right. It sets `aria-expanded` and swaps its title between show and hide.
- The scene is SVG: dotted paths in `gray`, pixel trees, and a campsite with two seated campers. Hover or focus on the campsite shows the control hints along three smoke-shaped text paths.
- Each browser owns one stone, a named slider with a `you` label. Arrows, Home/End and PageUp/PageDown move it; Shift with left or right switches paths. Escape hides the scene and returns focus to the toggle.
- Text and links under the scene stay selectable; the overlay clips its own hit area around them.
- Under reduced motion the flame is one static frame and the embers, smoke and toast stop.
- The stone count is accumulated stones, not visitors online.
