# Select

A single-choice list behind a trigger, with optional groups and a meta value on each option.

**Source:** `quartz/components/controls/select.ts` (`buildSelect`), skinned by `quartz/components/styles/controls.scss`, which `quartz/styles/base.scss` loads on every page. Used by the activity range, the wind graph picker, the DEXA lab date, the best-efforts year, sort and month pickers, and the press kit.

**Skin:** the preview loads the site's `controls.scss` as shipped. Labels are `darkgray`, options show the `focus` ring inside the menu frame, and the menu takes the width of the root. A host sizes the root; the preview gives the range select `inline-size: 15rem` through `data-class`.

## Site API

- `buildSelect({ id, label, groups, selected, onSelect, className? })` returns `{ element, trigger, setSelected, mount }`. `mount()` attaches the listeners and returns the cleanup; call it inside the `nav` handler and pass the cleanup to `window.addCleanup`.
- `label` names the listbox. The trigger has no visible label; its accessible name is "label: chosen option" ("wind graph: CdA Profile"), so a screen reader hears the value as well as the purpose. `setSelected` updates that name.
- A group with a `label` renders a `.g-select-group` (`role="group"`) with an `aria-hidden` `.g-select-heading`. An unlabelled group renders its options in place.
- An option is a `.g-select-option` button with a `.g-select-check` and a `.g-select-label`. The trigger holds `.g-select-value` and a `.g-select-chevron`.
- Hosts tint the control through `--control-height`, `--control-font-size`, `--control-surface`, `--control-accent`, `--control-accent-ink` and `--control-focus`. Triathlon sets them at its roots. A host's own width or position rule wins, because the root box is styled through `:where(.g-select)`.

## Menu variant

The triathlon shift settings (`buildShiftingSettings` in `quartz/util/triathlon-card.ts`) wear the same skin as a `role="menu"` of `menuitemradio` items. The menu stays open while you choose in several groups, so its behaviour lives in its host. The skin marks a choice from `aria-selected` (listbox) or `aria-checked` (menu).

## Preview

The preview runs the site's `buildSelect` through `Garden.ui.select(root)`. The root is a `div` with `data-g-select`, an optional `data-id` and `data-class`, and a child `<script type="application/json">` that holds `{ label, selected, groups }`. The binder writes the chosen value to `root.dataset.value`, dispatches a bubbling `g:change` event with `detail.value`, and returns a cleanup. `Garden.buildSelect` is the builder itself.

An option has no meta slot. Put a meta value in its label ("lap 1 · 12:00"); typeahead matches the start of the label.

## Rules

- Opening moves focus to the selected option. ArrowUp and ArrowDown move by one, Home and End go to the ends, a typed letter jumps to the next matching label (the buffer resets after 700ms), Escape closes and returns focus to the trigger, Tab closes. Focus leaving the control or a pointer press outside it closes it.
- Choosing an option marks it, calls `onSelect`, closes the menu and returns focus to the trigger.
- The selected option has a `selected` ground, `dark` text and a check mark. Other options are `darkgray`.
- Under `pointer: coarse` the trigger and options are 44px tall.
- A Select has no modified state. A Base view marks a changed query on its filter and sort tabs instead; see BaseToolbar.
