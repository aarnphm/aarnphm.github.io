# TrailSettings

The settings panel of the home page trail. It opens from the settings sign in the trail scene and holds the forest, holidays, animals, campsites and people controls.

**Source:** `quartz/components/visitor-trail/forest-controls.ts` (`forestControlsMarkup`, `mountForestControls`) · styles: `quartz/components/styles/shared-trail-frame.scss` (panel, rows, legends), `quartz/components/styles/trail-community.scss` (select and action rows)

## Preview

A copy of the live panel from `/` (selector `.trail-tree-panel`), cut to two forest sliders, the holiday scene, one campsite slider and the light, visit and pack-up rows. The selects run the site's `buildSelect` through `Garden.ui.mountAll()`. The preview drops the fixed position and the rise animation, and restates the value cell because the frame's `bundle.css` is unlayered and would otherwise win over the site's layered select skin.

## Rows

Every control is a row of one table. Rows are separated by a 1px `lightgray` rule; a section is a `fieldset` whose `legend` is a full-width 9px `gray` band.

| Row | Left | Value cell | Below |
| --- | --- | --- | --- |
| Slider | label | `output`, mono, 1px `lightgray` box, right-aligned | range input, 4px track filled to the value |
| Select | label | `.g-select` at `11em`, trigger 18px high in the same box as `output` | nothing |
| Action | label | a native button in the same box (`pack`) | nothing |

## Rules

- The value cell is one shape for every row: 1px `lightgray` border, no radius, 0 3px padding, `darkgray` text, the row's font. A select or button is never taller than an `output`.
- No row has a description line. State that a reader needs goes in the value cell; status for assistive technology goes in a visually hidden `role="status"` region (the camp connection).
- The first legend has no top border, because the panel head's bottom border already sits above it.
- A select menu takes at least the trigger's width and grows leftward to fit its longest option.
- An action that cannot be undone from the panel (`pack`) is a button. It is not a switch, because there is no "unpack".
- Escape closes the panel and returns focus to the sign. Arrow, Home, End and Page keys stay inside sliders and selects.
