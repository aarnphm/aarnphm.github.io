The garden is a reading site that also holds working tools: notes, collections, readers and a triathlon analytics workspace. Pages use warm paper, dark ink, square edges and thin structural borders. Type, spacing, alignment and fill carry the hierarchy. Use this system for any page, tool or mock that belongs on aarnphm.xyz.

"Paper" in this system means the `light` role: the page background in either theme.

## Content

- Write in the first person, lowercase by default. Note titles keep their author's casing ("Aaron's notes"). UI labels, legends and accessible names stay lowercase: "monthly peak averages", "previous week", "execution colors".
- Name a control by what it does ("list", "week", "copy"), never by how it is built.
- Keep labels short and literal. No emoji as markers, no exclamation marks, no marketing tone.
- A number keeps its unit, date and source: "4:27/km", "42.2 km", "2026-10-07". Label a computed estimate or a model output as such. Show missing data as "–", never as zero.
- An error says what failed, at the place it failed, with a recovery button beside it: "could not load this block" and "retry" (Arena block view).
- The site root is `~`. Breadcrumbs separate with `/`.

## Colour

- Paint the page `light`. Body text is `darkgray`; headings and key values are `dark`. In the dark theme `light` holds the dark background, so read `light` as "background" and `dark` as "ink".
- `surface` is the raised working ground (row hover, toolbar, panel). `selected` marks a selected or matched row or segment; its label is `dark`.
- `line` draws hairlines between rows and around quiet panels. It reads 1.43:1, so it only groups. A border that marks where a control or list starts and stops uses `dark` or `darkgray`.
- `gray` is for structure: gridlines, tag borders, inactive icons. It reads 2.00:1 and never carries meaningful text.
- `secondary` (sage) and `tertiary` (peach) are fills and series. They are never ink, and never a focus ring in the light theme. Text on any solid accent fill uses `on-accent`.
- `salmon` and `secondary` are the signature accents. `rose`, `love`, `lime`, `gold`, `pine`, `foam` and `iris` are the Flexoki set for series, status and figure marks. Pair every status colour with a word or an icon.
- Tool surfaces (Arena, triathlon, courses, stream, the PDF reader, Base views, tag and folder lists, Curius) alias the shared roles under their own prefix (`arena-*`, `course-*`, `pdf-*`, `calendar-*`) and add their own sets (`tri-*`, `exec-*`, `race-*`). Read the Tool surfaces, Arena, Triathlon and Courses sections before building in one of them.
- Three themes: Light, Dark, and Dark (accessible). Dark (accessible) is Dark with `darkgray` raised to Flexoki base-400 (#9f9d96), so secondary text clears 4.5:1 on `selected` too (4.58:1, against 3.37:1 in Dark). The site ships Light and Dark only.
- Text pairs reach 4.5:1 in every theme; text at 24px or more, borders, focus rings and icons reach 3:1. Each colour token's note gives its ratio on paper.

## Type

- Four families. `title` (Space Grotesk, registered as `Space Groteskque`) sets the page title and headings. `body` (PP Neue Montreal) sets prose and general UI. `code` (Berkeley Mono) sets code, dates, units, keyboard hints and numeric readouts. `math` (KaTeX_Main) sets LaTeX.
- Prose starts at `body` (16px / 1.4). Long-form readers use `reader` (1rem / 1.65). Headings descend from `h1` (25.5px) by a ratio of 1.0699, so they stay close to body size and rank by weight and family. The page title uses `article-title` (2.6rem).
- Chrome uses two sizes: `ui` (14px) for controls, menus and collection titles, and `meta` (12px) for dates, counts and captions. Nothing a reader needs goes below `meta`.
- Set `font-variant-numeric: tabular-nums` on every number that changes or aligns in a column. Readouts use `meta-mono`.
- Weight 400 is the default; 500 to 600 marks a value; headings use 700.
- Headings wrap with `text-wrap: balance` and prose with `pretty`. Long URLs and identifiers wrap with `overflow-wrap: anywhere`.

## Math and numbers

- Math sets in `math` at 1.1em of the surrounding text, in `math-ink`. Wide display math scrolls inside its own box.
- A number inside an argument goes in inline math (`500{,}000`, `28\,\mathrm{mm}`, `2.3\times`). A number in a tool, table or readout goes in `meta-mono` with tabular figures (`500,000`, `28 mm`).
- Definitions, theorems, lemmas and proofs use the math callout: a 1px `math-frame` box with an italic title. The Math section has the details.

## Space, shape and depth

- Use the `space-*` scale for padding and gaps. A control group's internal gap starts at `space-2`; the gap between groups is at least twice that. Collection rows pad `0.4rem space-3`. Figures pad `space-6 1.75rem`.
- `kern` (12px) belongs to the shell grid only.
- Surfaces, controls, inputs, tags and figures are square (`radius-none`). Only true handles are round (`radius-full`, `radius-circle`).
- Group with a 1px border, a fill change or space. No `box-shadow`. No decorative left border; the blockquote's `3px solid secondary` rule is debt. A border that does structural work (table edge, joined segment, chart marker) is fine.

## Layout

Choose the page mode from the task. The Layout section has widths and grid tracks.

- Reading: one column with title and byline, plus table of contents and sidenotes when space allows.
- Collection: a bounded list or grid, with search and view controls in one fixed place. Rows are links.
- Workspace: a fixed toolbar over panes that scroll on their own. Charts and comparisons get the width they need.

The site has two registers. Prose pages use the reading defaults in this README. Tool surfaces (Arena, triathlon, courses, stream, `/read`, Base views, tag and folder pages, Curius) use mono lowercase controls, the `line` / `surface` / `selected` recipe and their own link hover. The Tool surfaces section describes them.

## States and interaction

- Use a native `<button>` for an action and an `<a href>` for navigation.
- Focus: `outline: 2px solid` `focus` with `outline-offset: 2px`. Inside a framed list, use `-2px` so the frame does not clip it. Focus stays visible on selected, pressed and expanded controls. In forced colours, use `Highlight`. The Focus section says when a page may remove the outline and what must replace it.
- Links on prose pages invert on hover: `link-hover-bg` ground, `link-hover-ink` text. This is the site's default `a:hover`. Tool surfaces replace it inside their own scope with an underline, a `surface` row fill or a `surface` block fill (see Link).
- Control hover changes fill or ink slightly, only under `(hover: hover) and (pointer: fine)`.
- Selected: `selected` fill, `dark` label and the matching ARIA attribute (`aria-pressed`, `aria-selected` or `aria-current`).
- Disabled: native `disabled` and `opacity: 0.5`. A loading control keeps its size and shows a status word.
- Control heights are compact: 1.5rem in readers, 1.75rem in collections, 2.125rem for Base view controls. Under `(pointer: coarse)` every control is 44px. No hit target is smaller than 24 × 24px.
- Text inputs use 16px or more on mobile so iOS does not zoom.

## Motion

- Frequent feedback: 100 to 150ms on named properties (`background-color`, `border-color`, `color`). Panels: 220ms `cubic-bezier(0.23, 1, 0.32, 1)`. Never `transition: all`.
- Under `prefers-reduced-motion: reduce`, stop movement, smooth scroll and spinners. The state still changes, with a static cue.

## Overlays

- A blocking modal is a native `<dialog>` opened with `showModal()`. It has a name and initial focus, Escape closes it, and focus returns to its trigger.
- Popovers and drawers sit on `light` with a 1px `line` or `dark` border, square corners and a close button that does not move. On narrow screens an edge panel becomes a bottom drawer. Stack them with the `z-*` tokens.

## Figures and data

- A technical figure sits in a square 1px `fig-frame` on paper, padded `1.5rem 1.75rem`, with a caption under it in `fig-caption` at the `meta` size.
- Accents inside a figure are `salmon` and `secondary`. Salmon marks the one element to look at; structure stays in `dark`, `darkgray` and `line`; a second series is `secondary`. Symbol labels are KaTeX, not SVG text.
- Charts take their colours from tokens: gridlines `line`, axes and tick labels `darkgray`, values and data strokes `dark`. Each value keeps its unit and reference date.

## Icons and imagery

- Icons are inline SVG in `currentColor`, 16px, stroked. An icon-only button has an accessible name; a decorative icon has `aria-hidden`. The site has no icon font and no emoji.
- External link previews show the source's favicon (Wikipedia, arXiv, LessWrong, SEP).
- The site image is the mascot: a hand-drawn circle with two dash eyes shifted left, drawn as pressure-width pen ribbons in ink #1c1b1a (Flexoki base-950, a step above `dark`) on paper #fffcf0 (`assets/Imagery/mascot.svg`, `icon.webp`). It comes from `quartz/util/mascot.ts`; `quartz/scripts/brand-art.ts` rasterises it into the favicon, touch icons and avatar. Its eyes look left, so a scene around it rises or moves to the left.
- The default social card (`assets/Imagery/og-image.webp`, 1200 × 630) shows the mascot climbing a peak toward a flag, a `salmon` pack and sun against a `secondary` range.
- Draw new brand art with the mascot's pen (`stroke`, `pressure`, `hand` in `mascot.ts`), never with filled pixels or a typeface. The site has no logotype; set the name in `title` type.

## Not in this system

- PP Neue Montreal and Berkeley Mono are licensed and served from the site's own font store, so they are not copied here. Previews fall back to the system stack until the woff2 files are added. Space Grotesk and the 20 KaTeX faces are open-licensed and included.
- Left out on purpose: the 8 `--shadow-*` tokens (the site bans shadows), 16 legacy numeric radius steps, the 1,500 imported Obsidian theme variables in `main.scss`, and the display faces used by single authored pages (GT Sectra, Playfair, Parclo Serif, Departure Mono, ITC Garamond).
- `surface`, `selected`, `line` and `fig-frame` are stored as resolved hex. The code writes them with `color-mix()`.
- Components come in two kinds. The pattern components (Button, Readout, the Arena and triathlon parts, Math) are renditions written from the SCSS files named in each README, on `bundle.css`. The Controls run the site's own code through `Garden.ui`. The 45 Quartz components (Navigation, Article, Prose, Collections, Stream, Study, Courses, Home, Diagrams) are copies of the site's server-rendered markup, styled by `site/garden.css`, which is the compiled site CSS pruned to the rules those copies use. Copy a pattern component; read a Quartz component to see what the site renders today.
- `components/bundle.js` is one classic script that defines `window.Garden`. It holds 17 of the site's own scripts from `quartz/components/scripts`; `Garden.mount(slug)` fires the `nav` event they listen on, so 15 Quartz previews (callouts, code copy, TikZ, sidenotes, slides, table of contents, page-list filter, keybind panel, breadcrumbs, Evergreen, Base toolbar, and three diagrams) respond to clicks and input as they do on the site. Scripts that fetch from the site (search, palette, stream, transclusion, stacked notes, reading time, flashcard review) and the theme toggle, which would override the frame's theme, are left out, so those previews stay static.
- `Garden.ui` runs the site's own controls: `dropdown`, `select`, `datePicker` and `calendar` wrap `setupDropdown`, `buildSelect`, `buildDatePicker`, `MonthGrid` and `useAnchoredPreview` from `quartz/components/controls/`, and `mountAll(scope)` binds every `[data-g-dropdown]`, `[data-g-select]`, `[data-g-datepicker]` and `[data-g-calendar]` in the scope. Each binder returns a cleanup function. The bundle carries Preact and floating-ui for them, and `bundle.css` carries the site's `controls.scss`, so a Controls preview shows the classes, ARIA and keys the site ships. Each Controls README lists the site API and the declarative inputs.
- Diagrams keep three references: KVHeadGrouping for the figure grammar (frame, sliders, KaTeX labels, readout), AttributionGraph for a node-and-edge graph with a hairline readout, ablation and pruning, and MetaEthicsTree for a tree figure. The other attention diagrams and MethodologyTree were removed; read their source in `quartz/components/mdx` when you need one.
- `Recommendations` is placed in no layout, so no page renders it; its preview is last-known markup.
- Quartz components with no preview: `CascadeFilter` and `MFAFactorBases` (no page uses them), and the layout wrappers (`Head`, `Flex`, `Grid`, `Spacer`, `DesktopOnly`, `MobileOnly`, `ConditionalRender`, `NotebookRuntimeLoader`), which render no UI of their own. `Explorer` and `Graph` are not in the current layout.
- Where a source rule fails contrast (sage focus rings, `gray` labels, paper text on `tri-accent` tabs, `gray` proof reasons), the rendition uses the corrected role, and its README names the source value.
- Triathlon has about 800 local custom properties. This system carries the colours and sizes that repeat across panels. Chart geometry (plot insets, per-chart heights) stays in the code.
