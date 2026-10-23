# Layout

The shell is one grid with nine content tracks between gutters and outer tracks: `0 | kern × 5 | 9 × 1fr | kern × 5 | 0` on desktop, `kern × 3` gutters on tablet, `kern × 1.5` (18px a side) below `bp-mobile`. Pages place content by named column ranges; components split themselves with container queries.

| Mode       | Use it for                                                          | Contract                                                                                                                 | Width                                                                                                                   |
| ---------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Reading    | Essays, technical notes, letters, reference                         | One column, title and byline, table of contents and sidenotes beside it when space permits. The document owns scrolling. | Desktop column `5 / span 3`; tablet and mobile `3 / -3`. Aim for 60 to 75 characters a line.                            |
| Collection | Folder and tag lists, Evergreen, stream, Base views, Arena channels | A bounded list or grid, compact metadata, one stable place for search and view controls. Each row links to a full item.  | Arena frames at `52rem`; lists fill the reading column.                                                                 |
| Workspace  | Arena reader, PDF reader, Curius, triathlon analysis                | Stable toolbar, panes that scroll on their own, rails that collapse when content stops fitting.                          | Arena reader `52rem`; Curius `15rem` rail + `minmax(0, 52rem)`; triathlon `3 / -3`. Full-viewport readers use `100dvh`. |

## Rules

- Choose the mode before styling. A preview, menu or chart embedded in another page takes its host's width.
- Give every grid or flex child that holds text `min-width: 0`, and every child of a bounded scrolling workspace `min-height: 0`.
- Overflow belongs to the surface that needs it: a wide table, chart or code block scrolls inside its own container. The page body never scrolls sideways, down to 320px.
- `body { overflow: hidden }` is reserved for a bounded workspace that keeps every action reachable.
- Use logical properties (`padding-inline`, `margin-block`, `inset-inline`) for reading-direction layout; keep physical coordinates for maps, chart axes and pointer positions.
- Align title, metadata, content and controls to the same local edges.
- Authored pages (letters, the 404 scene, quotes, menu) may replace the palette, faces and layout, scoped by a precise route or frontmatter attribute. They still keep visible focus, readable contrast, reduced-motion guards and no page-level horizontal scroll.
