# Wikilinks and embed settings

Garden parses wikilinks as Markdown syntax. A link or embed must fit on one line. The general order is:

```markdown
[[target#heading#{settings}|label]]
![[target#heading#{settings}|caption|widthxheight]]
```

The anchor, settings, and label are optional. Settings come before the first alias pipe. They are removed from the destination URL. Their meaning depends on the destination: image embeds accept CSS declarations; links, note transclusions, and PDFs use object settings.

## Images: raw CSS

```markdown
![[thoughts/images/bike-trainer.webp#{float:right}]]
![[thoughts/images/bike-trainer.webp#{float:right; width:min(40%, 14rem); height:auto; margin:0 0 1rem 1.5rem}]]
![[thoughts/images/bike-trainer.webp#{object-fit:cover; aspect-ratio:3 / 4}|Indoor training|300x400]]
```

The text inside `#{...}` becomes the image's `style` attribute. Use CSS property names and semicolon-separated declarations. Values such as percentages, `calc()`, `min()`, `var()`, custom properties, quoted URLs, and `!important` pass through unchanged. The HTML serializer escapes attribute delimiters. The browser validates the CSS and applies its normal cascade. See the [HTML style attribute](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/style).

This is a declaration list. Selectors, stylesheet rules, and `@media` blocks require a stylesheet. Keep the wikilink on one line. The wikilink tokenizer balances braces; escape a literal unmatched brace in a CSS string with a backslash, or use a CSS hexadecimal escape such as `\7d ` for `}`.

Image settings use CSS syntax: `float:right; width:40%`. Object syntax such as `float:'right', width:'40%'` is for other embed types and is not converted into CSS. Empty settings add no style. The raw settings also remain available as `data-metadata` for existing consumers.

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, and `.webp` (case-insensitive). A regular link such as `[[photo.webp]]` stays a link.

### Captions and dimensions

| Syntax                                                         | Result                                                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `![[photo.webp]]`                                              | Image with empty alt text and automatic dimensions.                             |
| `![[photo.webp\|300]]`                                         | Width attribute of 300 pixels; automatic height.                                |
| `![[photo.webp\|300x200]]`                                     | Width and height attributes in pixels.                                          |
| `![[photo.webp\|A caption]]`                                   | Alt text; the OFM HTML pass also creates a figure and Markdown caption.         |
| `![[photo.webp\|A caption\|300x200]]`                          | Caption and dimensions together.                                                |
| `![[photo.webp#{width:40%; height:auto}\|A caption\|300x200]]` | CSS on the image, with the existing caption and dimension attributes preserved. |

The final pipe segment counts as dimensions only when it is digits, optionally followed by `x` and more digits. Earlier segments form the caption. CSS width and height take precedence over the dimension attributes under the normal cascade.

Styles belong to the `<img>`. A generated caption belongs to its enclosing `<figure>` and does not inherit the image's layout properties. Use an authored HTML figure when the entire image-and-caption group needs a float or other wrapper layout.

On newspaper pages, an image with a nonempty inline style keeps its authored layout and dimensions. It bypasses the automatic profile frame and responsive photo dressing. Captioned images retain their ordinary OFM figure. A gallery containing a styled image retains its ordinary image grid; other galleries and unstyled photos keep the newspaper presentation.

### Flow and grids

Place a floated image before the text that should wrap beside it. `float:right` positions the image within its current container; it cannot move it above preceding blocks. Add a width and margin when the surrounding text needs room. A later block can use `clear:both` to end wrapping. See [CSS float](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/float).

Consecutive images in one Markdown paragraph form an image grid. A blank line gives each image its own paragraph. Grid and flex items follow their container's layout rules, so a float will not rearrange grid cells.

## Links and object settings

```markdown
[[thoughts/craft|craft]]
[[thoughts/craft#practice|practice]]
[[thoughts/craft#{popover:false}|craft without a preview]]
```

For non-image destinations, use a JSON object body. The parser also accepts common relaxed forms with unquoted word keys, single-quoted strings, and trailing commas. This is a small compatibility parser, not a complete JSON5 implementation. Strict JSON is the dependable form for quoted strings with punctuation:

```markdown
[[thoughts/craft#{"popover":false}|craft]]
```

`popover:false` disables supported hover previews. False-like values also include `0`, `"false"`, `"0"`, `"no"`, and `"off"`. The legacy keys `noPopover` and `no-popover` use the same false-like interpretation despite their names; prefer `popover`. Other object fields remain metadata unless the destination's renderer consumes them. CSS image settings do not turn ordinary links into styled links.

## Note transclusions

```markdown
![[thoughts/craft]]
![[thoughts/craft#practice]]
![[thoughts/craft#^block-id]]
![[thoughts/craft#practice#{collapsed:true}|Practice notes]]
![[thoughts/craft#{collapsed:false}|Open notes]]
```

The presence of `collapsed` requests a collapsible wrapper. Boolean `true` starts it closed; boolean `false` starts it open. An alias supplies its title. Omit the setting for the normal transclusion. Use actual booleans: a nonempty string such as `"false"` is truthy here.

The source note's `transclude` frontmatter can override title and heading presentation. That is a separate page-level option, not an arbitrary wikilink setting.

## PDF embeds

```markdown
![[paper.pdf#page=3]]
![[paper.pdf#{page:3, fit:'page', scale:'125%', height:720, overscan:4}|Slides]]
```

| Setting           | Meaning                                                                               |
| ----------------- | ------------------------------------------------------------------------------------- |
| `page`            | Positive page number. Takes precedence over `#page=N`.                                |
| `fit` (or `mode`) | `width`, `page`, or `actual`. Aliases: `fit-width`, `fit-page`, `native`, and `100%`. |
| `scale`           | Positive numeric scale, or a percentage string such as `'125%'`.                      |
| `height`          | Positive viewport height in pixels.                                                   |
| `overscan`        | Positive number of extra pages used by the viewer.                                    |

The alias supplies the viewer title. A final dimension-like alias segment is removed from that title; use `height` to size the viewer. Regular PDF links preserve literal `#page=N` for the reader.

## Other destinations and fragments

| Destination                                                       | Result                                                                                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `[[note#Parent#Child]]`                                           | Keeps the heading path for nested heading resolution; the fallback fragment uses the final heading.                     |
| `[[#heading]], [[#^block]]`                                       | Heading or block in the current note.                                                                                   |
| `[[note label]]`                                                  | Implicit alias only when `note` exists and the full path `note label` does not. An explicit pipe avoids ambiguity.      |
| `[[catalogue.base#Table View]]`                                   | Base view route, with the view name as the default label. Add `!` to transclude it.                                     |
| `![[deck.fc]]`                                                    | Flashcard deck's flashcards view.                                                                                       |
| `![[thoughts/Jax#code-cell-1\|Example]]`                          | Notebook cell transclusion; `collapsed` uses the same wrapper behavior.                                                 |
| `![[movie.mp4]]`                                                  | Video with controls and looping. Also `.webm`, `.ogv`, `.mov`, `.mkv`.                                                  |
| `![[audio.mp3\|Caption]]`                                         | Audio player with caption. Also `.wav`, `.m4a`, `.ogg`, `.3gp`, `.flac`.                                                |
| `![[https://www.youtube.com/watch?v=ID#transcripts]]`             | YouTube embed with transcript opt-in. Raw timestamp fragments such as `#t=35` are preserved.                            |
| `![[https://example.com\|600x400]]`                               | External webpage embed; external embed dimensions also accept percentages. Provider embedding restrictions still apply. |
| `![[triathlon#calendar#2026#display=year]]`                       | Calendar view; `list` and `year` displays are supported. The year is optional.                                          |
| `![[triathlon#2026-10-31#analytics#settings=display:simplified]]` | Activity trace settings in anchor segments, separate from `#{...}` object/CSS settings.                                 |

Triathlon embeds also use `filter=` and boolean trace switches in `settings=`, joined by `&`. Their grammar is owned by [the day-card parser](../../../quartz/components/triathlon-day-card.tsx) and [trace settings](../../../quartz/util/triathlon-trace-settings.ts). Calendar handling lives in [transclude-triathlon](../../../quartz/components/renderPage/transclude-triathlon.tsx).

## Tables and escaping

Pipes inside wikilinks must be escaped in Markdown tables. Garden's OFM preprocessing does this automatically; explicit escaping also works, including the pipe after settings:

```markdown
| picture                                                     | note           |
| ----------------------------------------------------------- | -------------- |
| ![[photo.webp#{width:10rem; max-width:100%}\|Caption\|160]] | a styled image |
```

Backslashes escape Markdown punctuation in targets and aliases. An escaped pipe still acts as the alias separator so table cells can contain wikilinks. `\#` in a target is literal rather than an anchor marker. Code spans and fenced code blocks preserve examples as text. Incomplete settings such as a missing closing brace leave the wikilink literal.

## Owners and verification

- [Wikilink tokenizer and Markdown annotations](../../../quartz/extensions/micromark-extension-ofm-wikilinks/): syntax, captions, dimensions, object settings, image CSS, media routing.
- [String parser](../../../quartz/util/wikilinks.ts): extraction and target resolution outside the Markdown compiler.
- [OFM transformer](../../../quartz/plugins/transformers/ofm.ts): image grids and rendered captions.
- [Profile transformer](../../../quartz/plugins/transformers/profile.ts): newspaper framing and explicit-style precedence.
- [Link transformer](../../../quartz/plugins/transformers/links.ts): preview opt-outs and URL rewriting.
- [Verification replay](../../../docs/verification/wikilink-image-styles/README.md): exact commands and saved fixtures.
