# Hover and focus details

Expose explanatory detail through both pointer hover and keyboard focus. Keep essential information in an accessible description, including when a visual tooltip is unavailable. Use a wide invisible SVG hit-path for a thin arc, with a focusable, named target and visible focus treatment.

Choose the interaction from the layout:

- If details fit in reserved space without obscuring content, a CSS `:has()` treatment can respond to target `:hover` and `:focus-visible`. Keep the detail visible while the pointer is over it, and provide a continuous pointer path from the target. Hidden details must not intercept pointer events.
- If a popup obscures meaningful content, use a controller that supports Escape dismissal while focus remains on the trigger. Suppress reopening until the current hover/focus interaction ends. Register listeners and cleanup within the page's `nav` handler.
- If the explanation contains actions or needs deliberate opening, use a native disclosure or button-controlled popover with the appropriate focus behavior.

Check target focus, movement from target onto the detail, dismissal where needed, and touch access to essential information. A tooltip with permanent `pointer-events: none` cannot satisfy pointer transfer to its own content. An opacity-only hover selector also needs explicit focus and hidden-state behavior.

The [W3C hover/focus criterion](https://www.w3.org/WAI/WCAG22/Understanding/content-on-hover-or-focus.html) defines persistence, hoverability, and when dismissal is required. Existing figure implementations are useful layout references; check their interaction contract before copying them.
