# VisitorWeather

The home page sun or moon button and its weather panel.

**Source:** `quartz/components/VisitorWeather.tsx` · scripts: `quartz/components/scripts/visitor-weather.inline.ts`, `visitor-weather-map.ts` · worker: `worker/weather.ts` · styles: `quartz/components/styles/visitor-weather.scss` · notes: `docs/visitor-weather.md`

**Debt:** the accessible name is capitalised ("Weather near you").

## Preview

A copy of the server-rendered markup from `/` (selector `.visitor-weather`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The panel is shown open (`data-ds-open` in place of `:popover-open`) and the Mapbox map is a grey placeholder. The preview is static because the readings come from `/api/weather`.

## Rules

- The icon is 16px with a 44px hit target. It shows the sun by local day and the moon by night. The rays turn and the body floats unless reduced motion is set.
- The panel is a native `popover="auto"`: condition and country, a 90px map crop, temperature and wind, and the time to the next sunrise or sunset. City and source text are left out.
- Location comes from Cloudflare, rounded to a tenth of a degree. The site never asks for browser geolocation.
- The map loads on first open and is removed on navigation. A map failure shows a retry button.
