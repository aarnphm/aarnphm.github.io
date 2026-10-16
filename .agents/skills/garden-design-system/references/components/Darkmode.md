# Darkmode

The theme toggle: one button that cycles system, dark and light, shown as three icon states.

**Source:** `quartz/components/Darkmode.tsx` · styles: `quartz/components/styles/darkmode.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `#light-toggle`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is static because its script sets the theme when it loads, which would override the theme switch on this page.

## Rules

- The choice writes `saved-theme` on the root element and persists across visits.
- Icons come from the Tabler icon font; only the current state's icon is visible.
