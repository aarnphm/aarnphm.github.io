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
folds. Each pair of board rims and jacket rims shares a compound SVG path; the
closed faces and their paint order stay separate. The saved artwork belongs on the front board. Verified edition spine artwork
comes from `jackets.json`, which pins the publisher source, ISBN, crop and hashes of
the local front and spine images. The emitter attaches a spine only when both local
hashes match. The spine otherwise uses sampled colour and readable title/author
labels. Spine crops are local WebP assets, at most 96 × 320 px; sourcing is an offline
step, with no publisher requests during a build or browsing.

The offline source workflow can opt into `sampling: 'independent'` for a complete
photographed spine plane. It retains native samples across the spine while limiting
its length to 320 px. Each axis stays within the photographed edge lengths and the
96 × 320 px cap; the renderer already maps artwork to the book's physical face.
This improves narrow lettering without increasing resident image or detail limits.

For an interior crop of a photographed plane, `lowPerspective` pins the exact
retained crop's corners in the original image. Its local targets cover the entire
cropped plane. The offline recipe proves that this plane reproduces the prior crop
before changing its sampling; the original `perspective` and `crop` still own the
detail derivative.

An SVG source can retain lettering outlines from a verified designer wrap, or use
`vectorSource.kind: 'reference-reconstruction'` when a complete matching physical
spine supports the layout. Its `vectorSource` record pins the local file, hash and
glyph provenance alongside the original source and front-cover hashes. Retained
bitmap marks remain identified as bitmap artwork. The offline workflow renders the
SVG into the same bounded WebP tiers; the browser keeps one image per face and does
not inline the vector nodes. A mixed raster/vector source can cap its detail width
with `vectorSource.detailWidth` (at most 256 px) to preserve native raster samples.

The cover emitter generates 320 px front images. The camera keeps up to 32 covers
and 64 spine images in separate pools, with at most four detailed spine images.
It uses cached bounds, a 120 px viewport margin and a 72 px minimum projected book
height. Prefetch stays within four requests in flight and 48 cached images.
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

The shelf viewport uses native two-axis scrolling for wheel input and momentum.
Wheel distance stays in screen pixels and preserves zoom and horizontal position
without lifting books from the shelf. Hovering a book on the stage lifts its copy
and reveals the matching sidebar entry without moving the camera. Scrolling the
sidebar list interpolates selection continuously from the top visible entry to the
bottom entry and follows it with the camera. The lifted copy waits until that
scroll has been still, so a fast pass does not clone a book on every row. A padded,
clipped plane keeps SVG overflow within the camera's existing bounds; fit, zoom,
drag and keyboard jumps write the same native scroll offsets. A settled scale is
stored in the world and scene size, so the scrolled layer matches the drawing.
Pinch zoom and a scale glide use a temporary transform from the last baked size,
then bake when the scale is still.
Detailed spine upgrades wait during wheel input, drag and camera glides. Replay and
limits live in `docs/verification/bookcase-shelf-scroll/README.md`.

Keyboard focus uses the selected topic's expanded label and fill, and an inverted
fill on other controls. The viewport title uses that same fill while the drawing
has focus. Controls and the preview dialog do not draw focus outlines. A category's
initial selects and focuses its next matching topic from anywhere inside the
bookcase, except editable fields and an open preview. Repeated initials cycle the
matching topics; arrows and Home/End still operate within the topic list.

The preview keeps its own entry index while the modal is open. Wheel, button and
keyboard browsing update the preview and its bounded thumbnail preloads. The shelf
camera, index scroll and lifted copy update once on close. Shelf painting and
camera prefetch pause under the modal. The hardcover entrance runs on opening;
changing books inside it keeps the model at rest. The full cover, detail spine,
drag rotation and reduced-motion handling stay in the hardcover preview module.
Derivative artwork loads immediately. Original-cover and detail-spine upgrades wait
300 ms after decoding the derivatives, and a new selection or close cancels them.
The model keeps its current aspect until the incoming cover decodes. Texture commits
wait for active drag or entrance motion to finish. Replay and frame-timing evidence
live in `docs/verification/bookcase-preview-performance/README.md` and
`docs/verification/bookcase-preview-spikes/README.md`.

The view keys `shelfStyle` (straight, corner, spine, wall, zigzag, ladder) and
`shelfMaterial` (paper, walnut, birch-ply, black-lacquer, steel, copper, concrete)
select a cabinet variant. The server render writes them to `data-shelf-style` and
`data-shelf-material`, and browser repacking reads them back. A material sets the
tone tokens through an inline style in the scene. Each face also gets a finish:
`faceFill` picks a per-tone gradient, `grainFill` adds a grain overlay (veneer
tiles from `quartz/static/bookcase-grain/`, made by
`docs/verification/bookcase-materials/gen-grain.py`; noise patterns for steel,
copper and concrete), laid on each face by the exact affine map of its plane, and
`materialDefs` emits the shared `<defs>`; `materialCss` writes the gradient stop variables per theme. A material
can bring companion parts (a concrete case has black boards, a copper case a black
plinth). Every book spine gets a `bc-spine-sheen` polygon above its art. Books on the corner wing turn 90
degrees and show the back cover, so they get no cover image. Keep filtering,
grouping, entry identities and preview navigation in their existing owners.

Replay scripts and measured limits live in
`docs/verification/bookcase-memory/README.md`.
