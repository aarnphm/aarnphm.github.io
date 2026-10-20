# Vault rendering

The `vault` view shows a film collection as 35 mm cans standing on edge in a steel archive
rack. It is the cinema counterpart of the bookcase. The design brief is
`/mnt/project-files/film-collection/DIRECTIONS.md`, section 4.

| Owner                                              | Contract                                                                                                                                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layout.ts`                                        | Can sizes by era, the packing of the run into shelves and bays, dividers, card holders, and the can geometry (band, label strip, lid disc, label arcs, pull offset). Uses `project` and `box` from the bookcase geometry. |
| `styles.ts`                                        | Flat tone ladders per theme and fixed label colours. Emits the scene `<style>` (a fill class per tone) and the shared `<defs>` (handwheel finishes, label arcs).                                                          |
| `scene.ts`                                         | Builds the SVG in painter's order: rails and deck, then per bay its posts, its shelves from the bottom up (board, then cans and dividers by x, then card holders), its cap; last the end panel and handwheel.             |
| `render.ts`                                        | `buildVault(entries, view)`: takes resolved plain entries and returns the stage, the index and the vault card dialog. `../render.ts` resolves the Base properties into these entries.                                     |
| `../../../components/scripts/base-vault.inline.ts` | Camera (native scroll, drag to pan, zoom, fit), selection, the pull and lid detail, drag along a can's pull to slide it out or back in, and the vault card.                                                               |

Every can is moulded grey polypropylene with a ribbed lid. Era sets its size: before the first
`eraCuts` year a 1000 ft can, after it a 2000 ft can. Faces are flat fills: gradient fills on every
can cost about 30 times the raster time while the camera moves. Status sets the label: `to watch` is a vinyl strip,
`recurring` is skewed camera tape, `finished` is vinyl with a red `VIEWED` date. The rating dot
(salmon at 8.5 and up, sage from 7.5) always has the number beside it on the label.

Without `groupBy`, a divider and a card holder mark each decade, so a year sort reads as a
timeline. With `groupBy`, group changes do. Entry indices identify placed copies, because a film
in several list-valued groups has one can per group.

Moving cans stay in `.vault-lift` throughout a pull, a held drag and the return to the slot.
The rack and its tags paint behind the complete moving solid. Its lid relief stays visible
during the slide; the original can returns when the moving copy seats and is removed.

The scene holds no images. The cover emitter swaps each index row's
`data-cover` for the 320 px thumbnail; the card loads the original poster after it.
