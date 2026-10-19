# Bookcase rendering

The Base query supplies the books, groups and order. The cabinet consumes that
selection through the shared geometry and scene modules.

| Owner                                                 | Contract                                                                                                                                                                                                                                     |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geometry.ts`                                         | Packs `ShelfGroup<T>[]` into runs (`BookcaseLayout<T>`) and holds the projection, box solids and case dimensions. This module is also imported by the Obsidian plugin; keep it free of imports and side effects.                             |
| `styles.ts`                                           | Places each run in world space for the shelf style and material (`placeShelves` gives a `StyledLayout<T>` in painter's order) and builds the book faces in each orientation. Imports only `geometry.ts`; the Obsidian plugin imports it too. |
| `scene.ts`                                            | Builds the SVG HAST from a styled layout and `SceneBook` metadata. Quartz and browser search use the same scene builder.                                                                                                                     |
| `../render.ts`                                        | Resolves Base properties, grouping, sorting and the sidebar. Serializes book paths, titles, author names and size options for browser repacking.                                                                                             |
| `../../../components/scripts/base-bookcase.inline.ts` | Repacks matching sidebar entries, handles the camera and selection, and opens the hardcover preview.                                                                                                                                         |

Search uses the sidebar's current reading order, including semantic ranking within
groups and group ranking. Empty groups leave the drawing. Clearing search restores
the current Base query's books. An empty selection produces three empty shelves;
workspace height is independent of the cabinet's projected bounds.

Entry indices identify placed copies. A book can occur in several tag groups, so
its slug alone cannot identify a placed copy. Repacking keeps each entry's original
index for hit testing, selection and preview navigation.

The browser caches the matching sidebar order between search and topic refreshes.
Each refresh reads the live DOM order again, including semantic ranking; root
cleanup releases the cached entries.

Every shelf book has a jacket surface, cover-board rims, recessed pages and spine
folds. The saved artwork belongs on the front board. Verified edition spine artwork
comes from `jackets.json`, which pins the publisher source, ISBN, crop and hashes of
the local front and spine images. The emitter attaches a spine only when both local
hashes match. The spine otherwise uses sampled colour and readable title/author
labels. Spine crops are local WebP assets, at most 96 × 320 px; sourcing is an offline
step, with no publisher requests during a build or browsing. The cover emitter
generates 320 px front images. The camera keeps up to 32 shelf images total across
fronts and spines, using
cached bounds, a 120 px viewport margin and a 72 px minimum projected book height.
Exposed jackets take priority over nearby spines and covers hidden between books,
so fitting a tall cabinet cannot starve its top or bottom cover faces.
Residency updates when selection, layout or a coarse camera window changes. Spine
art projects through the shared book frame, including flat and rotated books. Its
lettering replaces the fallback only after load; failure or eviction restores the
fallback. The selected lift and hardcover preview use the same edition texture. A lift
may reveal one temporary neighbour image; the camera can adopt that image without
the lift deleting it when its transition ends. Offscreen images leave the scene.
Live Base queries reuse known thumbnails and sampled spine colours through a cache
scoped to the current page navigation. Original covers remain available to the
selected hardcover preview. Browsing the shelf does not attach original scans.

The view keys `shelfStyle` (straight, corner, spine, wall, zigzag, ladder) and
`shelfMaterial` (paper, walnut, birch-ply, black-lacquer, steel) select a cabinet
variant. The server render writes them to `data-shelf-style` and
`data-shelf-material`, and browser repacking reads them back. A material sets the
tone tokens through an inline style in the scene. Books on the corner wing turn 90
degrees and show the back cover, so they get no cover image. Keep filtering,
grouping, entry identities and preview navigation in their existing owners.

Replay scripts and measured limits live in
`docs/verification/bookcase-memory/README.md`.
