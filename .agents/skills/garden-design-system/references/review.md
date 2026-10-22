# Review checklist

Run this before calling a new or changed surface done. Check the rendered page, not only the source.

1. **Mode.** The page uses the reading, collection or workspace contract from Layout, and the right width for it.
2. **Tokens.** Colours, spacing, radii and layers come from tokens. No new literal hex outside an authored page; no new radius above `radius-none` except a handle.
3. **Text contrast.** Meaningful text, including captions and proof reasons, is `darkgray` or `dark`, never `gray`. Text on an accent fill is `on-accent`. Check both themes, including selected and hover fills.
4. **Focus.** Tab through every control. Each shows a 2px `focus` ring, also when selected, pressed or expanded. Every `outline: none` has one of the replacements in the Focus section, and the reset does not win over the ring by specificity. On tabs, focus on the selected tab looks different from selection alone.
5. **States.** Default, hover, focus, selected, expanded, disabled, loading, empty and error all exist where the control can reach them. Empty and error states offer a way forward.
6. **Type.** Nothing a reader needs is below `meta` (12px). Numbers use tabular figures and keep their units.
7. **Math.** Numbers in an argument are inline math; numbers in a tool or table are `meta-mono`. Units are `\,\mathrm{unit}`. A wide equation scrolls inside its own box at 320px. Figure symbols are KaTeX labels.
8. **Depth.** No `box-shadow`, no decorative left border, no rounded cards.
9. **Targets.** At least 24 × 24px; 44px under coarse pointers. Mobile inputs at 16px or more.
10. **Motion.** Named properties, 100 to 150ms for feedback, 220ms for panels. Reduced motion stops movement and smooth scroll.
11. **Overlays.** Blocking modals are `<dialog>` with focus in, Escape, and focus restored.
12. **Reflow.** 320px wide and 200% zoom: no page-level horizontal scroll, every action reachable.
13. **Evidence.** Record the route, viewport, theme and a screenshot of each state you checked.
