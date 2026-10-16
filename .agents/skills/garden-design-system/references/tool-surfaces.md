# Tool surfaces

Pages for finding, reading or working through many items use the tool register: Arena, the triathlon pages, the course pages, stream, the PDF reader (`/read`), Base views, tag and folder lists, Curius, Evergreen and link popovers. They share one colour recipe. Each page sets its own hover, density and accents inside its own scope.

## The shared recipe

Every tool surface declares the same three roles under its own prefix:

| Role | System token | Local names in the code |
| --- | --- | --- |
| Hairline | `line` | `--arena-line`, `--course-line`, `--pdf-line`, `--base-table-line`, `--base-list-line`, `--calendar-line`, `--tri-be-line` |
| Raised ground | `surface` | `--arena-surface`, `--course-surface`, `--base-table-surface`, `--base-list-surface`, `--tri-be-surface`, `--tri-src-surface`, `--tri-header-tab-surface` |
| Selected ground | `selected` | `--arena-selected`, `--course-selected`, `--pdf-selected` |

The PDF reader uses a quieter 90% mix (`pdf-surface`, `pdf-selected`). Tag and folder lists frame in `dark` where Arena frames in `arena-line`, because a list of notes reads as a document and Arena reads as a tool. Keep that difference.

## Links and hover

- **Prose pages** keep the inverted hover: `link-hover-bg` behind `link-hover-ink`. It is the site's default `a:hover`.
- **Tool surfaces** override it inside their own scope with one of three forms (see Link): a text link turns `dark` and underlines, a row link fills with `surface`, and a block link fills with `surface` and raises its border to `gray`.
- Scope every override to the page (`[data-slug*='arena']`, `.course-home`, `body[data-slug^='read/']`, `.triathlon`, `.stream`, `.bases-*`, `ul.section-ul`). Never change the global `a:hover` to suit one tool.

## Density

| Page | Control height | Type for controls | Frame |
| --- | --- | --- | --- |
| Tag and folder lists | rows padded `0.4rem 0.75rem` | `ui` | `1px dark` |
| Base views | `control-base` 2.125rem | `ui`, 500 | `lightgray` on buttons, `line` on tables |
| Arena index and channel | `control-collection` 1.75rem | `arena-control` (mono 0.66rem) | `arena-line` |
| Arena reader, PDF reader | `control-reader` / `control-pdf` 1.5rem | mono 0.66rem | `arena-line` / `pdf-line` |
| Course pages | `--course-control-height` 1.75rem | mono `meta` | `course-line` |
| Triathlon calendar | `control-calendar` 2rem | mono `meta` | `calendar-line` |
| Triathlon panels | about 1.55rem | `tri-tab` (mono 0.62rem) | `line` |

Under `pointer: coarse` every control rises to `control-touch` (44px).

## Rules

- Start a new tool surface from the shared recipe. Use the system names (`line`, `surface`, `selected`) and add a local alias only when the page needs its own value.
- Controls are mono and lowercase. Labels at 0.62 to 0.66rem are existing dense exceptions; new labels start at `meta`.
- Focus is the `focus` ring, inset (`-2px`) inside frames. Triathlon panels may use `tri-be-focus`. Sage rings in the current code are debt.
- `gray` labels in the current code (stream title actions, the Arena count, best-efforts captions, triathlon block titles) are debt. Use `darkgray`.
- A full-viewport tool (`/read`, the Arena reader) owns `100dvh`, hides the page footer and lets each pane scroll on its own.
