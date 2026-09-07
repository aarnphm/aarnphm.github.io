# Figure controls

Keep the existing square segments, figure border token, and visible focus styling. Choose semantics by behavior before copying a component's visual classes.

| Behavior                                            | Control and state                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Switch among related content panels                 | Native buttons with `role="tab"`, `aria-selected`, and panel relationships inside a named tablist |
| Select exactly one parameter value                  | A named native radio group, styled as segments if needed                                          |
| Toggle an independent feature or optional highlight | A native button with `aria-pressed`, or a checkbox for a form value                               |

## Tabs

Give each tab a unique ID and `aria-controls` pointing to its panel. Give each panel `role="tabpanel"` and `aria-labelledby` pointing back. Use a component-instance prefix so multiple figures do not share IDs. The selected tab has `aria-selected="true"` and `tabindex="0"`; the others have `aria-selected="false"` and `tabindex="-1"`. Hide inactive panels from both layout and accessibility APIs.

A horizontal tablist supports Left and Right to move focus with wraparound. A vertical list uses Up and Down and declares its orientation. Home and End may move to the first and last tab. Prevent the handled navigation key's default scrolling, and leave unrelated keys alone.

For immediately available local panels, automatic activation on focus is appropriate. For costly loading, use manual activation with Enter or Space. Native button activation provides the click event for those keys. Keep the selected state, focus position, root mode attribute, and visible panel consistent with the chosen activation model.

Register click and keydown listeners during page setup. Remove both on cleanup, with setup and `window.addCleanup` registration inside the `nav` handler. Check initial SSR state before attaching listeners. CSS may select the panel through the root mode attribute, or the controller may use `hidden`; maintain one consistent visibility owner.

Test keyboard entry into the tablist, wraparound, activation, visible focus, and exit with Tab. Repeat after SPA navigation and with two instances on the same page. See the [W3C tabs pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/) for the complete contract.

## Toggle buttons and radio groups

Keep a toggle button's accessible name stable and update `aria-pressed` with its state. `role="tab"` does not belong on an optional highlight button. For an exclusive choice with a valid "none" state, include a "none" radio option or preserve the existing documented selection model.

Use `aria-checked` for a switch or radio role; prefer native controls when practical. Preserve labels and keyboard interaction while applying the figure's visual tokens. Pointer hit areas must remain usable and must not overlap neighboring controls.
