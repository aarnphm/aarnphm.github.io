# Dropdown

A text button that shows or hides one panel of details, such as gear or pace zones.

**Source:** `quartz/components/controls/dropdown.ts` (`setupDropdown`). The triathlon shell (`triathlon/shell/dropdown.ts`) calls it for the gear and pace panels with its own classes.

**Skin:** the site helper has no skin, so the `g-dropdown-*` classes in `bundle.css` belong to the system, for a page without its own. The close button is `darkgray`, and every trigger and close button shows the `focus` ring. The triathlon source close button is `gray` with no focus style.

## Site API

- `setupDropdown({ wrap, trigger, panel, openClass, closeButton? })` returns the cleanup. The site version carries behaviour only: the host owns the markup and the classes, and its styles show the panel while `openClass` is on `wrap`.
- `closeButton: { className, container }` adds a "×" button as the first child of `container` and moves focus to it on open. Use it when the panel covers the page.
- Escape that closes the panel calls `preventDefault()`, so later Escape handlers on the page skip it instead of moving focus again.

## Preview

The preview runs the site's `setupDropdown` through `Garden.ui.dropdown(root)`. The root has `class="g-dropdown"` and `data-g-dropdown`, and holds a `.g-dropdown-trigger` and a `.g-dropdown-panel`. The open class is `g-dropdown-open`; the panel is hidden while the root lacks it. `data-close` on the root asks the helper for its close button (`g-icon-button g-dropdown-close`, named "close" plus the trigger text, such as "close pace"). To render a panel open, write `g-dropdown-open` on the root, `aria-expanded="true"` on the trigger and `aria-hidden="false"` on the panel. `Garden.ui.mountAll(scope)` binds every `[data-g-dropdown]` in the scope.

## Rules

- A click on the trigger toggles the panel. A click outside the root closes it. Escape closes it and moves focus back to the trigger.
- The trigger is `darkgray` over a `gray` underline; open, both turn `dark`. The panel has a 1px `darkgray` frame on paper, square, at `z-dropdown`.
- Labels in `.g-kv` are `darkgray`, lowercase. Values are `dark`, tabular. Long values wrap.
