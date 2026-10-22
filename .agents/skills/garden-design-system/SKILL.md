---
name: garden-design-system
description: 'Build, revise or review Garden UI with its design system: tokens, the reading and tool registers, layout modes, component contracts and the review checklist.'
---

# Garden design system

Use this skill for any page, component, chart, figure or mock that belongs on aarnphm.xyz, and for reviewing one. It is a local copy of the Garden Design System artifact in the Claude app (`https://claude.ai/artifact/6QVXBrg6Dw3dtUvSF27ZjK`), so every agent can read it without that app.

The repo is the source of truth. When this skill and the code disagree, the code wins for what the site renders today; report the difference instead of "fixing" one side silently. The exception is contrast: where a source value fails contrast, this system names a corrected role on purpose (see Contrast corrections).

## The rules that cover most work

- Paint pages `light` (paper). Body text `darkgray`, headings and key values `dark`. In the dark theme `light` is the dark background, so read `light` as "background" and `dark` as "ink".
- Working surfaces use the recipe `line` (hairline, groups only), `surface` (raised ground: hover, toolbar, panel) and `selected` (selected or matched row, label `dark`). A border that marks where a control starts and stops uses `dark` or `darkgray`.
- `gray` is structure only (gridlines, tag borders, inactive icons). It never carries text. `secondary` (sage) and `tertiary` (peach) are fills and series, never ink. Text on a solid accent uses `on-accent`.
- Square corners everywhere (`radius-none`); only true handles are round. No `box-shadow`. No decorative left border. Group with a 1px border, a fill change or space.
- Four families: `title` (Space Grotesk) for titles and headings, `body` (PP Neue Montreal) for prose and UI, `code` (Berkeley Mono) for code, dates, units, keys and readouts, `math` (KaTeX_Main) for LaTeX. Chrome uses `ui` (14px) and `meta` (12px); nothing a reader needs goes below `meta`.
- Numbers: inside an argument, inline LaTeX (`28\,\mathrm{mm}`, `500{,}000`); in a tool, table or readout, `meta-mono` with `tabular-nums` (`28 mm`). Keep units, dates and sources. Show missing data as `–`. Label estimates as estimates.
- Copy is first person, lowercase, literal. Controls are named by what they do ("list", "copy"). No emoji, no exclamation marks.
- Two registers. Prose pages invert links on hover (`link-hover-bg` / `link-hover-ink`). Tool surfaces (Arena, triathlon, courses, stream, `/read`, Base views, tag and folder pages, Curius) use mono lowercase controls, the `line` / `surface` / `selected` recipe and scope their own hover (underline or `surface` fill).
- Native `<button>` for actions, `<a href>` for navigation, native `<dialog>` with `showModal()` for blocking modals. Focus is `outline: 2px solid` `focus` with `outline-offset: 2px` (`-2px` inside a framed list) and stays visible on selected and expanded controls. Remove the outline only with a replacement from `references/focus.md`.
- Hover only under `(hover: hover) and (pointer: fine)`. Controls are 1.5rem in readers, 1.75rem in collections, 2.125rem in Base views, 44px under `(pointer: coarse)`. Text inputs are 16px or more on mobile.
- Motion: 100 to 150ms on named properties for feedback, 220ms `cubic-bezier(0.23, 1, 0.32, 1)` for panels, never `transition: all`. Reduced motion stops movement and keeps a static cue.
- Pick the page mode first (reading, collection, workspace); see `references/layout.md`. Text-holding grid and flex children get `min-width: 0`. The page never scrolls sideways down to 320px.

## Read only what the task needs

| Task                                                                      | Read                                                                     |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Anything new, or unsure                                                   | `references/brand-book.md` (the full brand book)                         |
| A colour, size, space, radius or z-index value                            | `references/tokens.md`                                                   |
| Page width, grid tracks, page mode                                        | `references/layout.md`                                                   |
| A tool surface (Base views, stream, `/read`, tag or folder lists, Curius) | `references/tool-surfaces.md`                                            |
| Arena                                                                     | `references/arena.md`                                                    |
| Triathlon analytics, calendar, charts                                     | `references/triathlon.md`                                                |
| Courses and flashcards                                                    | `references/courses.md`                                                  |
| Home page widgets, trail, press kit                                       | `references/home.md`                                                     |
| Math, numbers in prose, chart labels, TikZ                                | `references/math.md`                                                     |
| Focus rings, `outline: none`, focus on tabs or charts | `references/focus.md` |
| Brand art, mascot, social images                                          | `references/imagery.md`                                                  |
| A specific component                                                      | `references/components/index.md`, then `references/components/<Name>.md` |
| Calling a surface done                                                    | `references/review.md`                                                   |

A component README states what the component is, its tokens, its states and the source files in the repo. The Controls (Dropdown, Select, DatePicker, Calendar) wrap `quartz/components/controls/`; reuse those builders, do not write a new select or calendar. A CodeMirror completion list takes `completionTheme` from `quartz/util/codemirror-completion-theme.ts` (see Completion); tint it with `--control-accent`, never restyle it from page CSS. Quartz component READMEs describe what the site renders today; read the named source before changing it.

## Where the values live in the repo

- Theme colours: `theme.colors.lightMode` / `darkMode` in `quartz.config.ts`, exposed as `--light`, `--lightgray`, `--gray`, `--darkgray`, `--dark`, `--secondary`, `--tertiary`, `--highlight`.
- Spacing, radii, z-index, transitions: `quartz/styles/tokens/`. Shared control skin (`g-*` classes): `quartz/components/styles/controls.scss`.
- Surface prefixes: `--arena-*`, `--course-*`, `--pdf-*`, `--calendar-*`, `--tri-*`, `--exec-*`, `--race-*` in each surface's SCSS under `quartz/components/styles/`.
- `surface`, `selected`, `line` and `fig-frame` are written with `color-mix()` in code; `references/tokens.md` gives each source mix next to the resolved hex. Reuse the existing variable for the surface you are in; do not paste the hex.
- Figures: `quartz/components/styles/figure.scss` mixins and the `interactive-diagrams` skill.

## Contrast corrections

These system values depart from the source on purpose, because the source fails contrast. Keep the corrected role in new work and do not revert a rendition to the source:

| Source                                      | Ratio  | Use instead                |
| ------------------------------------------- | ------ | -------------------------- |
| sage focus rings                            | 1.51:1 | `focus` (`dark`)           |
| `gray` labels and proof reasons             | 2.00:1 | `darkgray`                 |
| paper text on `tri-accent` tabs             | 3.31:1 | `on-accent`                |
| figure caption `gray` 70% mixed with `dark` | 3.68:1 | `fig-caption` (`darkgray`) |

Text reaches 4.5:1 in every theme; text at 24px or more, borders, focus rings and icons reach 3:1. Colours that must be told apart differ in lightness, not hue alone. Every status colour also has a word or an icon.

## Checking the work

Run `references/review.md` on the rendered page, in both themes, at 320px and at desktop width. Follow the repo's AGENTS.md for how to get a rendered page (the running `quartz/scripts/dev.ts` watcher) and where to save evidence (`docs/verification/<task>/artifacts/`).

## Keeping this copy in sync

`references/` is generated; edit the design system artifact, then regenerate. Only an agent with the Claude app's Artifact tool can read the artifact:

1. `Artifact read` the artifact with `paths` for `project/README.md`, every `project/*.md`, `project/tokens.json`, `project/assets/Imagery/README.md` and every `project/components/<Name>/README.md`, saving into one folder.
2. `python3 .agents/skills/garden-design-system/scripts/build_references.py <that folder>`.
3. Review the diff of `references/`. Edit this `SKILL.md` by hand only when a rule above changes.

Previews, `bundle.js`, `bundle.css`, `site/garden.css` and fonts stay in the artifact. The build kit for those is in `.claude/design-system/` (git-ignored).
