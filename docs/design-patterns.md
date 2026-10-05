# Garden design patterns

The garden uses a warm paper background, an editorial reading layout, and compact tools organized with type, alignment, and thin borders. Collections and training workspaces share this language while changing the amount of space given to content and controls.

This document records the patterns found in the source on **2026-10-04** and gives rules for extending them. **Existing** describes the implementation. **Adopt** describes the design contract for future changes, including corrections identified in the [styling audit](design-audit-2026-10-04.md). Proposed recipes below are documentation examples; they have not been added to the runtime.

## 1. Choose the layout from the task

| Mode       | Use it for                                                   | Layout contract                                                                                                                      | Existing owners                                                                                                                                                                                                                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reading    | Essays, technical notes, reference material                  | One reading column, title and byline, supporting notes alongside it when space permits. Let the document own scrolling.              | [base.scss](../quartz/styles/base.scss), [quartz.layout.ts](../quartz.layout.ts), [renderPage.tsx](../quartz/components/renderPage.tsx)                                                                                                                             |
| Collection | Arena channels, folder/tag lists, Evergreen, stream archives | A bounded list or grid, compact metadata, a stable place for search and view controls. Rows lead to a full item.                     | [arena.scss](../quartz/components/styles/arena.scss), [listPage.scss](../quartz/components/styles/listPage.scss), [evergreen.scss](../quartz/components/styles/evergreen.scss), [stream.scss](../quartz/components/styles/stream.scss)                              |
| Workspace  | Arena feed/reader, PDF reader, Curius, triathlon analysis    | Stable toolbar and independently scrollable content where needed. Give charts, media, and comparisons the width their task requires. | [arena-feed.scss](../quartz/components/styles/arena-feed.scss), [pdf-reader.scss](../quartz/components/styles/pdf-reader.scss), [curius.scss](../quartz/styles/pages/curius.scss), [triathlon-workspace.scss](../quartz/components/styles/triathlon-workspace.scss) |

Start with the reading layout for prose, a collection for finding an item, and a workspace for manipulating or comparing information. A preview, menu, or chart embedded in another page belongs to its host's available width.

**Existing geometry.** The shared shell has nine content tracks with gutters and outside tracks. At desktop width, reading content uses `--layout-reading-column: 5 / span 3`; tablet and mobile use `3 / -3`. The shared breakpoints are `800px` and `1400px`. Mobile gutters derive from `--kern: 12px`, giving `18px` per side. Arena collections and its reader use a `52rem` maximum width. Curius combines a `15rem` rail with a `minmax(0, 52rem)` content column. Triathlon assigns `3 / -3` to its reading and wide columns. These are different task layouts within the same site.

**Adopt.** Use the `--layout-*` properties when changing the shared shell. Use a container query for a component's internal split, following triathlon's comparison and workspace containers. Keep `min-width: 0` on grid/flex children and `min-height: 0` on children of a bounded scrolling workspace. Limit overflow to the surface that owns it. Reserve global `body { overflow: hidden }` for an explicitly bounded workspace that keeps all content and actions reachable.

For new reading surfaces, aim for a measure around `60–75ch` and inspect actual text. The existing shell uses grid tracks, so this is a reading target, rather than a new fixed width for every page. Preserve room for equations, tables, figures, and sidenotes.

## 2. Use the palette by role

[quartz.config.ts](../quartz.config.ts) defines the site palette. [main.scss](../quartz/styles/main.scss) supplies additional Flexoki and imported theme variables. The names below describe roles even though some retain Quartz's original light/dark naming.

| Existing token                    | Role                               | Light theme           | Dark theme            | Usage rule                                                                       |
| --------------------------------- | ---------------------------------- | --------------------- | --------------------- | -------------------------------------------------------------------------------- |
| `--light`                         | Page background                    | `#fffcf0`             | `#100f0f`             | Page, panel, and reading background                                              |
| `--lightgray`                     | Recessed surface / structural base | `#e6e4d9`             | `#282726`             | Surface mixes and quiet dividers                                                 |
| `--gray`                          | Faint structural color             | `#b7b5ac`             | `#575653`             | Decorative lines and nonessential marks; meaningful text needs a stronger role   |
| `--darkgray`                      | Body and secondary text            | `#6f6e69`             | `#878580`             | Body copy, explanations, metadata, captions                                      |
| `--dark`                          | Strong foreground                  | `#100f0f`             | `#cecdc3`             | Headings, key values, emphasized text                                            |
| `--secondary`                     | Sage accent                        | `#cdd597`             | `#cdd597`             | Selected fills, annotations, and series; measure before using it as ink or focus |
| `--tertiary`                      | Peach accent                       | `#fcc192`             | `#fcc192`             | Secondary accent and interaction hints                                           |
| `--highlight` / `--textHighlight` | Highlight treatments               | Theme-specific values | Theme-specific values | Measure text against the resulting fill                                          |

`--light` remains the background in dark mode. A component that exchanges `--light` and `--dark` creates an inverted surface. Make that inversion an explicit content requirement, such as a contrasting data readout, and check the entire surface.

**Existing shared surface recipe.** Arena, Curius, headings navigation, and triathlon's best efforts repeat these relationships under their own prefixes:

```scss
--local-line: color-mix(in srgb, var(--gray) 55%, transparent);
--local-surface: color-mix(in srgb, var(--light) 86%, var(--lightgray));
--local-selected: color-mix(in srgb, var(--secondary) 18%, var(--local-surface));
```

The PDF reader uses a slightly quieter `90% --light` surface. Figure borders use their own palette mixin. Preserve those differences when the content calls for them.

**Adopt.** Keep surface, text, selected state, focus, and chart-series roles distinct. Use `--darkgray` or `--dark` for meaningful labels. Use sage and peach primarily as fills or series, with a readable foreground. During a future consolidation, promote the repeated line/surface/selection recipe at the shared style boundary; retain component prefixes for component-specific geometry and data. Such shared aliases do not exist yet.

Text needs at least `4.5:1` contrast at ordinary sizes, or `3:1` when it meets the large-text definition. Required control and graphical indicators need `3:1` against their adjacent colors. Measure the actual rendered surface, including transparency, selection fills, and embedded content. The audit records declared-token measurements and their limitations. Sources: [W3C text contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html), [W3C non-text contrast](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html).

## 3. Give type a job

The default families are configured in [quartz.config.ts](../quartz.config.ts) and loaded through [fonts.scss](../quartz/styles/fonts.scss):

| Role                                                   | Existing family / size                             | Adopt                                                                                            |
| ------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Titles and headings                                    | Space Grotesk, registered as `Space Groteskque`    | Use `--titleFont` / `--headerFont`; preserve the registered name until the font owner changes it |
| Prose and general UI                                   | PP Neue Montreal                                   | Use `--bodyFont`; default prose starts at `1rem`                                                 |
| Code, dates, units, keyboard hints, technical controls | Berkeley Mono                                      | Use `--codeFont`; apply `font-variant-numeric: tabular-nums` to changing or aligned values       |
| General UI text                                        | `--text-ui: 0.875rem` (`14px` at the default root) | Shared controls, collection titles, menus                                                        |
| Metadata                                               | `--text-meta: 0.75rem` (`12px`)                    | Dates, secondary descriptions, captions                                                          |
| Dense workspace text                                   | Many local values between `0.48rem` and `0.78rem`  | Keep these as existing exceptions to review; new meaningful labels start at the metadata step    |

**Existing heading scale.** Body headings descend through `25.5`, `23.8337`, `22.2763`, `20.8207`, `19.4601`, and `18.1885px` for `h1–h6`, an adjacent ratio of approximately `1.0699`. The article title has a separate `2.6rem` (`41.6px`) rule in [custom.scss](../quartz/styles/custom.scss). Page owners can override that title size. The root line height is `1.4`. Collections commonly use `1.45`; workspace body copy uses values around `1.5–1.7`.

**Adopt.** Use the shared heading scale for ordinary notes, and the UI/meta steps for chrome. Use semantic heading levels independently of their visual size. Long-form reader text can use `1.5–1.6` line height within its owner. Keep labels that wrap to three lines at `1.4` or higher. Weight `400` is the default for small meaningful text; use `500–600` to emphasize values. Verify that the font face actually supplies the requested weight and style.

Use balanced wrapping on headings, natural wrapping on prose, and `overflow-wrap: anywhere` for long URLs and identifiers. A short unit/value may remain unbroken. Truncated titles and identifiers need a reachable full value through the item page, expanded state, or keyboard-accessible disclosure. Set mobile text inputs to at least `16px` to avoid automatic zoom in iOS Safari.

Express density through compact spacing, alignment, and disclosure before reducing type. Technical values should remain selectable and retain their unit, date, and source when needed to interpret them.

## 4. Build hierarchy with spacing and edges

[tokens/\_spacing.scss](../quartz/styles/tokens/_spacing.scss) provides a quarter-rem scale: `--space-1` through `--space-6`, then `--space-8`, `10`, `12`, `16`, `20`, and `24`. `--space-px` and `--space-0` handle hairline and zero values. The shell also uses `--kern: 12px` for its grid relationships.

**Adopt.** Use the spacing scale for component padding and gaps. Preserve `--kern` where it defines the existing shell. Start compact control groups around `--space-2`; give separate groups at least twice their internal gap when space is the grouping signal. Collection rows commonly use `0.4–0.5rem` block padding and `0.75rem` inline padding. Figures deliberately have more breathing room through the shared frame mixin.

Align title, metadata, content, and controls to the same local edges. Use logical padding, margins, and insets for reading-direction layout. Keep physical coordinates for maps, chart axes, and pointer positions.

**Shape.** The default surface and control radius is `--radius-none`. Contiguous segmented controls share edges, and selected segments retain the same geometry. Use a `1px` border for structural grouping. Pills and larger radii require a specific established owner or function, such as circular handles, rather than a new generic card treatment.

**Depth.** Follow [AGENTS.md](../AGENTS.md): avoid `box-shadow` and decorative `border-left`. Establish hierarchy through spacing, backgrounds, borders, type, and position. Existing shadow tokens and legacy shadow declarations are audit debt. A table edge, chart marker, or joined segment border has a structural function and should be assessed on that function.

## 5. Controls share a state contract

Use a native button for an action and a real link for navigation. Links retain open-in-new-tab behavior. Keep an icon in `currentColor`, provide an accessible name for an icon-only control, and hide decorative SVGs from the accessibility tree.

| State              | Visual treatment                                                                      | Behavior contract                                                               |
| ------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Default            | Neutral foreground, clear placement, thin border or established text-action underline | Label identifies the action or destination                                      |
| Hover              | Slight surface/foreground change                                                      | Gate hover-specific treatments with `(hover: hover) and (pointer: fine)`        |
| Focus              | Visible outline that works on the adjacent surface                                    | Focus remains visible on selected, expanded, and pressed controls               |
| Selected / pressed | Static fill plus readable label and, where useful, checkmark or underline             | Expose `aria-selected`, `aria-pressed`, or `aria-current` according to behavior |
| Expanded           | Stable trigger with a disclosure cue                                                  | Expose `aria-expanded`; Escape dismisses and restores focus where appropriate   |
| Disabled           | Distinct inactive state                                                               | Native `disabled` when unavailable; explain why if the reason matters           |
| Loading            | Stable control geometry and a visible label/status                                    | Expose busy state and prevent duplicate requests where necessary                |
| Error              | Readable message at the failed operation, with a recovery action                      | Preserve useful content and offer retry, clear filters, or the source link      |
| Empty              | State what is empty and how to proceed                                                | Search/filter emptiness offers a way to broaden or clear the selection          |

**Existing density.** Arena collections and Curius use `1.75rem` controls; Arena feed and the PDF reader use `1.5rem`. Coarse-pointer media queries raise those controls to `44px`. Triathlon controls vary by tool.

**Adopt.** Preserve compact visual chrome and ensure a target of at least `24×24` CSS pixels unless a documented WCAG exception applies. Aim for `44×44` on touch, following the existing coarse-pointer treatment. Adjacent expanded hit targets must remain distinct. The `44px` target is a site preference; the Level AA minimum and exceptions are defined in [W3C target size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html).

Use `--dark` or a measured component focus color for an outline outside the control. `--secondary` fails against the light paper surface. Leave enough space around scroll containers to avoid clipping the outline. In forced-colors mode, preserve a system-color indicator.

## 6. Reuse these component patterns

### Bordered collection row

Use one outer frame and shared separators between rows. Put the title first, aligned metadata alongside it, and the description beneath when needed. Make the row's navigation a link. Use a modest surface change on hover and an inset, measured focus outline when the border frame constrains outside space.

The model exists in Arena's channel list and the folder/tag list. Folder metadata can be visually hidden while retaining its slot when alignment depends on it. Filtered-out rows must not leave a frame with no visible content. Keep a full-title destination available.

Owners: [arena.scss](../quartz/components/styles/arena.scss), [listPage.scss](../quartz/components/styles/listPage.scss), [PageList.tsx](../quartz/components/PageList.tsx).

### Compact toolbar and segmented choice

Group search, sort/filter, view selection, and secondary actions in a stable toolbar. Let search take flexible width. A segmented choice retains equal edge treatment and a static selected cue. Use roving tab focus and arrow-key movement for a true tablist; use `aria-pressed` for independent toggles.

Keep the selected tab's focus visible. Category controls may scroll within a bounded row when their function requires it, with a cue and a keyboard route to the remaining items. Use a native select or the existing accessible listbox implementation when a long choice list is involved.

Owners: [arena-feed/filter.tsx](../quartz/components/arena-feed/filter.tsx), [baseViewSelector.scss](../quartz/components/styles/baseViewSelector.scss), [triathlon-best-efforts.scss](../quartz/components/styles/triathlon-best-efforts.scss), [comparison-chart-tabs.ts](../quartz/components/triathlon/activity/comparison-chart-tabs.ts).

### Reading workspace

Keep toolbar and navigation stable while the primary content pane scrolls. Give notes, source information, and filters a rail that can collapse at the width where content stops fitting. Use `100dvh`, safe-area-aware gutters, and bounded inner scrolling for full-viewport readers. Let long source text wrap; put code overflow inside its own focusable surface.

The PDF raster remains white in both themes. Its toolbar, rails, and annotation chrome follow the site theme. That is a content-specific exception documented by the reader owner.

Owners: [arena-feed.scss](../quartz/components/styles/arena-feed.scss), [pdf-reader.scss](../quartz/components/styles/pdf-reader.scss), [curius.scss](../quartz/styles/pages/curius.scss).

### Figures and data panels

Use the mixins in [figure.scss](../quartz/components/styles/figure.scss) for a technical figure's palette, frame, caption, and math alignment. The frame owns its container query and theme-sensitive border. Local cards, readouts, and controls inherit that palette. Register technical figures through the existing figure machinery.

Triathlon data panels use the same flat frame language with denser charts, tables, and aligned readouts. Use `tabular-nums`; retain units, source, time range, and the date of a reference measurement. Distinguish native measurements, calculated estimates, and model outputs in the label or legend. Missing data stays visibly missing.

Triathlon's existing category colors include swim teal `#3aa99f`, bike sage `#cdd597`, run salmon `#fdb2a2`, and Strava orange `#fc4c02`. These are category/provider colors. Encode a series with a label and a second cue where differentiation matters, such as a dashed reference line or marker. Evaluate chart contrast against the plot background; muted gridlines can remain decorative.

The best-efforts panel already demonstrates an Arena-style framed list within an analytics panel. The comparison workspace uses container queries and gives multiple series one inspectable plot surface. Copy those owners' composition when adding a similar analysis.

### Overlay, drawer, and popover

Choose behavior before appearance. A blocking modal uses a native `<dialog>` opened with `showModal()`, a name, initial focus, and focus restoration. A nonblocking popover or utility panel permits background interaction and uses the corresponding platform behavior. Keep background scroll, focus, and dismissal consistent with that choice.

Use a paper surface, thin boundary, square corners, stable close control, and bounded scrolling. An overlay must fit within the viewport and preserve safe areas. On narrow screens, an edge-attached panel may become a bottom drawer. Headings navigation demonstrates this geometry. Context menus anchor to their trigger and flip/shift when the viewport requires it, as the existing Floating UI owners do.

Owners: [headings.scss](../quartz/components/styles/headings.scss), [headings-modal.inline.ts](../quartz/components/scripts/headings-modal.inline.ts), [speech.inline.scss](../quartz/components/styles/speech.inline.scss), [arena-feed/delete-note.tsx](../quartz/components/arena-feed/delete-note.tsx). Arena's older block modal needs the accessibility correction described in the audit before serving as a behavior template.

## 7. Motion communicates a state change

**Existing.** Shared transition tokens range from `100–300ms`. The overlay variables in [variables.scss](../quartz/styles/variables.scss) use `220ms` with `cubic-bezier(0.23, 1, 0.32, 1)`. Triathlon uses short property-specific transitions and guards several interactions with reduced-motion preferences. SPA navigation and the activity workspace check the preference in JavaScript.

**Adopt.** Frequent controls give immediate feedback or a brief color/opacity transition around `100–150ms`. Use the established panel timing for occasional panel movement. Specify the properties that change. Keep a static label, selected state, or icon when motion is disabled. Guard movement, smooth scrolling, and view transitions with `prefers-reduced-motion`; essential state updates still happen immediately. Check the owning JavaScript as well as its stylesheet.

Theme changes update semantic colors together. Any future theme-switch transition suppression belongs at the theme owner, after observing that transitions cause a visible smear. Retain the existing `saved-theme` and system-preference mechanism.

## 8. Keep style ownership explicit

| Boundary                                               | Owns                                                                  | Extension rule                                                                                        |
| ------------------------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `quartz.config.ts` and `quartz/styles/main.scss`       | Theme values, typography configuration, theme compatibility variables | Change global roles here and verify every consuming mode                                              |
| `quartz/styles/tokens/` and `variables.scss`           | Spacing, radii, layers, shared timings, shell breakpoints             | Reuse consumed roles; introduce a shared role when multiple owners need it                            |
| `quartz/styles/base.scss`                              | Element defaults and shared grid layout                               | Fix shared behavior at this boundary                                                                  |
| `quartz/styles/custom.scss` and `quartz/styles/pages/` | Site-wide additions and route/layout composition                      | Route overrides belong with the route; avoid broad substring matching when a precise attribute exists |
| `quartz/components/styles/`                            | Component geometry and state                                          | Keep component rules scoped and pair them with their markup/runtime owner                             |
| `quartz/components/styles/figure.scss`                 | Technical figure mixins                                               | Import the mixins; let each figure own its local data/layout                                          |
| Browser scripts and Preact components                  | Interaction, ARIA state, focus, hydration                             | Mount at the `nav` lifecycle and register cleanup in the same lifecycle                               |

For new shared control work, consolidate the repeated surface/state recipe before adding another copy. Keep data-dependent chart layout local. Split the large triathlon stylesheet along its existing shell/calendar/activity/analytics/tools responsibilities when touching those boundaries. Preserve source order while moving rules; imported best-efforts styles already document order-sensitive overrides.

Use the existing layer tokens for page-level overlays. Numeric stacking within a chart's local stacking context can remain local. A page overlay needs a documented relationship to sticky headers, popovers, and top-layer dialogs.

## 9. A documentation recipe for a new tool surface

This example uses existing tokens and the shared surface relationships. The `.site-tool` names are illustrative. It intentionally places a strong focus outline outside the button; the layout must leave it room.

```scss
.site-tool {
  --tool-line: color-mix(in srgb, var(--gray) 55%, transparent);
  --tool-surface: color-mix(in srgb, var(--light) 86%, var(--lightgray));
  --tool-selected: color-mix(in srgb, var(--secondary) 18%, var(--tool-surface));
  --tool-control-height: 2rem;

  min-width: 0;
  color: var(--darkgray);
  background: var(--light);
  border: 1px solid var(--tool-line);
  border-radius: var(--radius-none);
  font: var(--text-ui) / 1.45 var(--bodyFont);

  .site-tool-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    padding: var(--space-3);
  }

  button {
    min-block-size: var(--tool-control-height);
    padding-inline: var(--space-3);
    color: var(--dark);
    background: transparent;
    border: 1px solid var(--tool-line);
    border-radius: var(--radius-none);
    font: inherit;
    cursor: pointer;
    touch-action: manipulation;
  }

  button[aria-pressed='true'] {
    background: var(--tool-selected);
    text-decoration: underline;
    text-underline-offset: 0.2em;
  }

  button:focus-visible {
    outline: 2px solid var(--dark);
    outline-offset: 2px;
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  @media (hover: hover) and (pointer: fine) {
    button:hover:enabled {
      background: var(--tool-surface);
    }
  }

  @media (pointer: coarse) {
    --tool-control-height: 44px;
  }

  @media (forced-colors: active) {
    button:focus-visible {
      outline-color: Highlight;
    }
  }
}
```

## 10. Review a new surface through its real entry point

Check the rendered implementation through the running Quartz watcher after its matching `build:ready` and a successful HTTP response. Preserve existing processes and unrelated work. Follow [agent-development.md](agent-development.md) for command effects.

Inspect the default, hover, keyboard focus, selected, expanded, disabled, loading, empty, and error states that the surface actually supports. Test both themes, a coarse-pointer viewport, `320px` reflow, and `200%` zoom. Check breakpoint boundaries when changing the shell, and container boundaries when changing a component. Two-dimensional plots and tables may scroll within their own surface when the information requires it; surrounding controls and prose must reflow. See [W3C reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html).

Complete the flow using the keyboard, including Escape and focus restoration. Check reduced motion, long titles, missing values, realistic source content, and the site's supported locales. Save the exact route, inputs, theme/viewport, watcher epoch, result, and a screenshot or response artifact. A source inventory, a loaded stylesheet, and a screenshot each prove different parts of the result.
