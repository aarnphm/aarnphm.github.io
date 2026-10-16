# Home page widgets

The home page (`/`) carries two widgets that no other page renders, the weather and the shared trail. Their buttons sit fixed at the top right of the viewport, out of the reading text. The press kit opens from any page; it is listed here because the harvest took it from the home page.

| Widget | Trigger | Surface |
| --- | --- | --- |
| VisitorWeather | 16px sun or moon, 44px hit target | Native `popover="auto"` with a 90px Mapbox crop and the next sunrise or sunset |
| HomepageTrail | Trail icon, 44px hit target | A full-page SVG scene: three dotted paths, pixel trees, a campfire, one shared stone per browser |
| TrailSettings | the settings sign in the trail scene | A fixed 220px panel: sliders, selects and one action as rows of one table, each with a value cell |
| PerformanceDebug | `⌥⇧D`, or `?trail-debug=performance` | A fixed 22rem panel at the bottom right: fps, a six-second frame chart, metrics, scene readings and the worst script and event |
| PressKit | `g` then `s`, on any page | Native modal `<dialog>`, in French, with an image card, a citation and links |
| PressKitCard | the `note` tab of PressKit, on a page or a stream entry | A canvas card: breadcrumb and date, title, rule, tags and reading time |

## Rules

- The weather map loads on first open and the playhtml frame loads when the trail shows. Navigation removes both.
- No widget asks for browser permissions. Weather location comes from Cloudflare, rounded to a tenth of a degree. The shared trail stores only a position and a path name per browser.
- Motion stops under `prefers-reduced-motion`: the sun's rays and float, the flame, embers, smoke and toast.
- The trail's pixel art (trees, campers, fire) and the 404 page are the two places the site draws in pixels.
