# Vault rendering

The `vault` view shows a film collection as 35 mm cans standing on edge in a steel archive
rack. It is the cinema counterpart of the bookcase. The design brief is
`/mnt/project-files/film-collection/DIRECTIONS.md`, section 4.

| Owner                                              | Contract                                                                                                                                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layout.ts`                                        | Can sizes by era, the packing of the run into shelves and bays, dividers, card holders, and the can geometry (band, label strip, lid disc, label arcs, pull offset). Uses `project` and `box` from the bookcase geometry. |
| `styles.ts`                                        | Flat tone ladders per theme and fixed label colours. Emits the scene `<style>` (a fill class per tone) and the shared `<defs>` (handwheel finishes, label arcs).                                                          |
| `scene.ts`                                         | Builds the SVG in painter's order: rails and deck, then per bay its posts, its shelves from the bottom up (board and card holders, then cans and dividers by x), its cap; last the end panel and handwheel.             |
| `render.ts`                                        | `buildVault(entries, view)`: takes resolved plain entries and returns the stage, the index and the vault card dialog. `../render.ts` resolves the Base properties into these entries.                                     |
| `../../../components/scripts/base-vault.inline.ts` | Selection, pull animation and card navigation. Its private `base-vault/` modules own the camera, gestures, depth masks, lid detail, poster loading and scene preparation. |

Every can is moulded grey polypropylene with a ribbed lid. Era sets its size: before the first
`eraCuts` year a 1000 ft can, after it a 2000 ft can. Faces are flat fills: gradient fills on every
can cost about 30 times the raster time while the camera moves. Status sets the label: `to watch` is a vinyl strip,
`recurring` is skewed camera tape, `finished` is vinyl with a red `VIEWED` date. The rating dot
(salmon at 8.5 and up, sage from 7.5) always has the number beside it on the label.

Without `groupBy`, a divider and a card holder mark each decade, so a year sort reads as a
timeline. With `groupBy`, group changes do. Entry indices identify placed copies, because a film
in several list-valued groups has one can per group.

Each moving can has a separate SVG sized to its swept bounds. A fixed mask copies the silhouettes
of the cans, dividers and posts that paint after its original slot. Those silhouettes conceal
the lid until it emerges along the projected pull axis. Only band and lid contours enter the
mask, so it does not repeat label shaping. When a foreground can also moves, the mask follows
its live copy until it returns. Card holders sit below the shelf top and paint before its cans.

The original stays transparent and keeps its hit area; it returns when the moving copy seats.
Pointer progress lives in memory, without changing a data attribute every frame. An interrupted
animation resumes from its actual eased progress. At most one active copy and three returning
copies remain mounted. The rack itself remains one SVG to avoid extra compositing cost.

The server emits the scene in an inert HTML template. The client fits the camera around the
rack before it lays out the can labels. After the first paint, it prepares labels for at most
eight cans per animation frame, with a four-millisecond script budget. Visible cans come first.
The script budget excludes browser layout and paint, which can still produce long frames.

Pointer holds pause preparation. Wheel and keyboard input postpone it until 180 ms after the
last input. Selecting a can prepares its labels immediately. Navigation cancels pending frames
and timers and restores the inert template. The film index, search and camera are available
during preparation. Scroll anchoring is disabled in the viewport because the camera owns its offsets.

The scene holds no images. The cover emitter swaps each index row's
`data-cover` for the 320 px thumbnail; the card loads the original poster after it.
