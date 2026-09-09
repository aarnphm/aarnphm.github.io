# Figure frame and registration

Register a Preact component in `quartz/components/mdx/registry.ts` through `registerMdxComponent`, and add its import/export to `quartz/components/mdx/index.ts`. Attach its stylesheet to `.css` and, when needed, its inline script to `.afterDOMLoaded`. Give the component an explicit `QuartzMdxComponent<Props>` type; a second type assertion is unnecessary.

Embed it in a note with the registered names in a JSX fence:

````md
```jsx imports={Zoomable,KVHeadGrouping}
<Zoomable label="KV head grouping">
  <KVHeadGrouping caption="Adjust the group ratio to inspect KV sharing." heads={8} />
</Zoomable>
```
````

Use `Zoomable` for interactive or large figures. A small static SVG can stand alone. Read `quartz/components/mdx/KVHeadGrouping.tsx` when a complete registration example is useful, and use its structure with the current component's props and IDs.

## Shared SCSS

`quartz/components/styles/figure.scss` emits mixins only. Import it from the figure stylesheet with `@use './figure.scss' as *;`. Use the source for current defaults and tokens.

| Mixin                                              | Purpose                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| `figure-frame($container, $intrinsic, $bg, $zoom)` | Root container, square 1px frame, palette, padding, and zoom restoration |
| `figure-caption`                                   | Caption typography and spacing                                           |
| `figure-caption-math-gap`                          | Inline math spacing inside a caption                                     |
| `figure-inline-math($katex)`                       | Math sizing and inherited color                                          |
| `figure-fo`                                        | HTML container for KaTeX inside SVG `foreignObject`                      |
| `figure-card-heading($size)`                       | Sentence-case headings in figure cards                                   |

Reuse `--fig-frame-border` for internal cards and readouts, and Flexoki accents such as `--fig-salmon` and `--fig-sage`. Derive fills through `color-mix`. Keep component-specific tokens under their own prefix and let the frame mixin own its theme-dependent border. Existing shadow-named tokens do not require adding a shadow.

The root is the named query container. Place responsive grids and breakpoint-dependent padding on descendants: a container cannot apply its own size query to itself. Select an intrinsic height from the rendered figure rather than copying another component's value.

`zoomable.scss` removes the zoomed figure's border, padding, and background. The frame mixin restores padding and background. Custom zoom rules should restore only the layout they need; adding a border frames the figure again inside the zoom container. Use `$zoom: false` only when supplying that custom restoration.

## SVG math and IDs

Use `foreignObject` with an HTML div for KaTeX. `figure-fo` sets vertical alignment and deliberately leaves `justify-content` to the caller. Size the foreignObject for the rendered label and set its own overflow when needed; overflow on the child div cannot prevent parent clipping.

Use `MathText` from `quartz/util/math-text.tsx` for caption math. The [math reference](math-and-readouts.md) covers render options and live readouts. Give SVG markers, clips, labels, and controls instance-specific IDs when the figure can appear more than once on a page.
