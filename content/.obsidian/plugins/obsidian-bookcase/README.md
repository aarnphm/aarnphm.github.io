# Garden Bookcase

A Bases view, `bookcase`, that draws the query result as one isometric bookcase in the anatomy-kit style. Each group is one or more shelves; each entry is a book with its spine to the front. Scrolling moves the camera along the books in order. A book you select slides out of the shelf and shows its front cover. A page panel next to the case shows the selected book's title, byline, properties, description and, optionally, the note body. In a pane, the note can open in an editor beside the view instead (see [Editor pane](#editor-pane)).

The garden renders the same `.base` file at build time. Both sides read the same keys and draw from the same geometry module, `quartz/util/base/bookcase/geometry.ts`, and its styles module, `styles.ts`, which esbuild bundles by relative path. `src/scene-model.ts` is the only file that calls them.

## View keys

| key             | option type | default        | meaning                                                                                                                                                                  |
| --------------- | ----------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `type`          | -           | `bookcase`     | view id                                                                                                                                                                  |
| `image`         | property    | `note.posters` | cover image, as in the cards view                                                                                                                                        |
| `spineTitle`    | property    | `note.title`   | spine label; the spine shows the text before the first `:`                                                                                                               |
| `spineAuthor`   | property    | `note.author`  | small label at the spine foot; a list shows 2 names, then `et al.`                                                                                                       |
| `spineColor`    | property    | none           | CSS colour; if absent or invalid, the colour sampled from the cover                                                                                                      |
| `bookHeight`    | slider      | `220`          | tallest book in world units, 120 to 320                                                                                                                                  |
| `shelfStyle`    | dropdown    | `straight`     | `straight`, `corner`, `spine`, `wall`, `zigzag` or `ladder` for `all` and for a topic with no style of its own; an unknown value draws `straight`                        |
| `shelfMaterial` | dropdown    | `paper`        | `paper`, `walnut`, `birch-ply`, `black-lacquer` or `steel` for every furniture part, with the same fallback; unset keeps the style's own materials. Books do not change. |
| `shelfWidth`    | slider      | `600`          | shelf length in world units; a group wraps onto a new shelf at this width, 320 to 1200                                                                                   |
| `scrollStep`    | slider      | `56`           | px of scroll per book, 24 to 120                                                                                                                                         |
| `shelfRows`     | slider      | `5`            | shelf levels in one bay; more bays run to the right, 1 to 12                                                                                                             |
| `showTags`      | toggle      | `true`         | label plates on the shelves; a topic's `showTags` overrides it                                                                                                           |
| `pageContent`   | toggle      | `true`         | render the note body in the page panel                                                                                                                                   |
| `editorPane`    | toggle      | `true`         | "Edit notes in a side pane": in a wide pane, a selection opens the note in an editor beside the view                                                                     |

`filters`, `sort`, `groupBy`, `limit` and `order` keep their Bases meaning. `order` lists the properties on the page panel.

## Topics

A topic is a value of the view's `groupBy` property or a key of `topicShelves`. A book is in a topic when its `groupBy` value is the topic, or when it carries the topic as a tag: the tag itself or a tag nested under it (`p/nf/essays` is in `p/nf`; a leading `#` is ignored). A book counts once per topic. Topics with one book share `other`. `all` is the view as Bases groups it, in the view's own style. A topic shows only its books, on shelves grouped by `topicGroupBy`, and its groups share runs. When that value is a list, the book goes on the shelf of its first item. Shelves sort by name, and books without a value come last on a shelf with no label.

There is no separate tag filter, as on the garden: a tag that is a `topicShelves` key, such as `p/scifi`, is a topic tab.

The topic is kept per viewer in the plugin's `data.json`, by `.base` path and view name. It is never written to the `.base` file.

Bases options cannot hold a map, so these keys are set in the `.base` file:

```yaml
views:
  - type: bookcase
    name: bookcase
    groupBy:
      property: category
    topicGroupBy: subcategory # default
    showTags: true # default
    topicShelves:
      philosophy:
        style: wall
        material: walnut
      p/nf:
        displayNames: non fiction
        style: ladder
        material: birch-ply
        showTags: false
      p/scifi: # a tag, not a category value: books tagged p/scifi
        displayNames: sci-fi
        style: spine
        material: steel
```

| key            | default       | meaning                                                                                                      |
| -------------- | ------------- | ------------------------------------------------------------------------------------------------------------ |
| `topicGroupBy` | `subcategory` | property that groups a topic's books into shelves                                                            |
| `topicShelves` | see below     | per topic: `displayNames`, `style`, `material`, `showTags`. A key that is no `groupBy` value is a tag topic. |

`displayNames` is the name on the topic's tab. Without one, `p/` is removed and `nf` reads `non-fiction`. An entry whose style is unknown uses the default for that topic. An entry with no material takes the view's `shelfMaterial`; with neither, the style's own materials apply. `showTags` on an entry overrides the view's `showTags` inside that topic. The garden's `topicKeys`, `topicLabel`, `topicShelf`, `hasTopicTag`, `topicShortcuts` and `TOPIC_SHELVES` in `styles.ts` hold these rules. Without an entry, `philosophy` is `wall`/`walnut`, `p/fiction` is `corner`/`paper`, `p/nf` is `ladder`/`birch-ply` and `p/scifi` is `spine`/`steel`. Other topics, `all` and `other` use the view's `shelfStyle` and `shelfMaterial`. There is no `all` entry. The rule is `topicShelves[topic]?.style ?? TOPIC_SHELVES[topic]?.style ?? shelfStyle ?? 'straight'`, and the same for the material. `showTags` is the entry's, else the view's.

## Selection

The reading position and the selection are different. The reading position is the book at the scroll position: the camera follows it and the index's roving tab stop is on it. The selection is the book that is lifted, marked `aria-current` and shown on the page panel or in the editor pane.

- Nothing is selected when the view opens or when the topic changes. The page panel then shows the topic and its count.
- A wheel or touch scroll by the user selects the book at the reading position, and that book slides out. A scroll that does not follow a wheel or touch move within 250 ms (a key, the scrollbar, a resize or the view itself) selects nothing.
- A click on a shelf book, the arrow keys in the index, keyboard focus on an index button and `Enter` select. `Enter`, or a click on the lifted book, opens the note (Mod: a new tab).
- The garden also selects on index scroll and index-row hover. Here the index is a visually hidden list for screen readers and the keyboard (SPEC 9), so there is no row to scroll or hover. Keyboard focus on an index button selects, as above.
- `Esc` (with focus in the view or the pointer over it) and a click on the empty stage clear the selection.
- A pointer that moves onto the lifted book, onto the slot it left on the shelf or onto the way between them, and then leaves all three, puts the book back and closes its preview. So does focus that leaves the view. The selection stays on the page panel or in the editor pane. A book that slides out from under a still pointer, or glides away from the pointer that clicked it, was not hovered. A press on the stage that drops focus to nothing is not a blur.
- A page preview (`hover-link`, source `bases`) opens only when the pointer really moves onto the lifted book, once per lift. Scroll and keyboard never open one. Any change of the lift closes the preview the view opened.

## Editor pane

With `editorPane` on, in a wide view in a main-area leaf, the first selection made by click or keyboard opens the note in an editor next to the view:

1. `workspace.createLeafBySplit(thisLeaf, 'vertical')`, once. The view's leaf is the one that `workspace.iterateAllLeaves` finds around the view; a leaf whose `getRoot()` is `workspace.leftSplit` or `rightSplit` does not count.
2. `leaf.openFile(file, { active: false, state: { mode: 'source' } })` for this and each later selection, with the file from `vault.getFileByPath`. The editor does not take focus.
3. `Enter`, or a click on the lifted book, calls `workspace.setActiveLeaf(leaf, { focus: true })` to move into the editor. Mod still opens a new tab.

A selection made by a wheel or touch scroll moves an open editor after a 300 ms pause, and never opens one. While the editor leaf lives, the page panel is hidden and the stage takes the full width. On `layout-change`, a closed leaf is dropped and the view falls back to the 80/20 page panel. It does not open a new one until the next click or keyboard selection. Clearing the selection leaves the editor on the last book. The view never closes the leaf, which can hold unsaved edits. Obsidian has no public API to size a split, so the leaf keeps Obsidian's default size. Embeds, the narrow layout and sidebars keep the page panel.

## Controls

The controls sit in a column at the top right of the stage, as in the garden:

- Zoom in, zoom out, fit and full screen buttons. Zoom steps by 1.35, from the fitted scale to `max(2, 4 × fitted)`. A button is disabled at its limit. `+` or `=` and `-` zoom while focus is in the view. Zoom keeps the focused book in place; fit returns to the fitted scale with no pan. Full screen shows the view alone; `Esc` or the button leaves it.
- Under them, the column of topic tabs: a collapsed tab shows its shortcut; the chosen topic is filled and opens leftward to "name count". The arrow keys, `Home` and `End` move between tabs and show that topic, and a letter jumps to the next tab whose shortcut starts with it. The column stops 5.5rem above the stage bottom and scrolls on a short stage. In a view narrower than 600 px, the zoom buttons sit in a row.

Drag the stage with a mouse or pen to pan. A press that moves less than 5 px is still a click, on the element where it went down. After a drag, the click is dropped, so no book opens or lifts. The pan holds until the focus changes; then it eases back to zero and the camera follows the scroll again. The zoom stays. A touch scrolls the books, as before. A new topic starts at its first book, from the top, at the fitted scale.

In a wide view the whole case, with every pulled-out book, fits beside the column (see [Layout](#layout)), so the camera only moves up and down. When it does not fit (the narrow layout, an embed, or the 0.55 scale floor), the camera keeps the pulled-out book clear of the column. Near the left wall it keeps the book on the stage, which wins when both cannot hold. A case narrower than the stage can therefore sit off-centre near the first or last book.

## Layout

In a pane, the stage fills the height of the element that scrolls: containerEl or its nearest ancestor with `overflow-y: auto | scroll` and a height of its own. A resize observer follows that element, so a pane that gets its height after the view loads still fills. A view 600 px or wider is wide (SPEC 8 says 760; the 80/20 split holds from 600): a CSS grid gives the stage 80% of the width and the page panel 20%, at least 240 px, which the stage gives up. With the editor pane open, the stage takes the full width. The case fits the stage width (SPEC 4.3, with the garden's fit width, clamped to 0.55–1); a case narrower than the stage is centred. In a wide view the scale also fits the whole case with every pulled-out book beside the control column, as the garden fits its whole box. The case then stays still across: centred, or moved left of the column. A narrower view puts the page under the stage. In an embed, the view scrolls itself in 520 px.

Covers are decoded once, in idle time, into 480 px thumbnails under `cache/covers/` with a sampled spine colour. The index is in `data.json`.

## Film vault

A second Bases view, `vault`, draws the query result as 35 mm film cans on edge in a steel archive rack: the garden's view for `cinematheque.base`. `src/vault-model.ts` is the only file that calls the garden's `quartz/util/base/vault/` modules. It gives resolved entries to `buildVault`, and so the rack, the can eras, the labels, the dividers and the card holders are the garden's. It replaces the scene's theme rules (`.theme-dark`) and grain base (`grain/vault/`, copies of `quartz/static/vault-grain/`), and removes the garden's card dialog.

| key             | option type | default         | meaning                                                                   |
| --------------- | ----------- | --------------- | ------------------------------------------------------------------------- |
| `type`          | -           | `vault`         | view id                                                                   |
| `image`         | property    | `note.posters`  | poster on the page panel                                                  |
| `labelTitle`    | property    | `note.title`    | rim title                                                                 |
| `labelDirector` | property    | `note.director` | `DIR.` surname on the label                                               |
| `labelYear`     | property    | `note.year`     | year; it sets the can era with `eraCuts`                                  |
| `labelRating`   | property    | `note.rating`   | number on the label; salmon dot from 8.5, sage from 7.5                   |
| `labelStatus`   | property    | `note.status`   | `recurring` is camera tape, `finished` is a red `VIEWED` stamp            |
| `labelStudio`   | property    | `note.studio`   | studio on a 2000 ft label when it fits                                    |
| `labelWatched`  | property    | `note.watched`  | date of the `VIEWED` stamp                                                |
| `labelRuntime`  | property    | `note.runtime`  | runtime and estimated reels on the page panel                             |
| `rackMaterial`  | dropdown    | `rack-steel`    | `rack-steel` or `black-lacquer`                                           |
| `bayWidth`      | slider      | `1200`          | shelf length in world units, 600 to 2400                                  |
| `shelfRows`     | slider      | `5`             | shelves per bay, 1 to 8; more bays run to the right                       |
| `dividers`      | slider      | `8`             | cans between dividers, 0 for none; a section change always gets one       |
| `eraCuts`       | -           | `[1980, 2016]`  | set in the `.base` file: tin before the first year, hammertone, then poly |
| `showTags`      | toggle      | `true`          | card holders with the section and its count                               |
| `crank`         | toggle      | `true`          | end panel and handwheel                                                   |
| `pageContent`   | toggle      | `true`          | render the note body in the page panel                                    |
| `editorPane`    | toggle      | `true`          | as the bookcase's                                                         |

Without `groupBy`, a divider and a card holder mark each decade. With `groupBy`, a group change does.

The rack scrolls on both axes in a viewport that fills the pane. The first frame shows the top-left of the first bay at a scale where rim titles are about 9 px. Drag across the rack to pan. Drag a can along its pull, toward the aisle or back into the slot, and it follows; past halfway it stays out, and short of halfway it returns. A press that moves less than 4 px is a click. The zoom, zoom out, fit and full screen buttons are at the top right; `+`, `-` and `0` do the same with focus on the rack. A data change keeps the camera and the pick.

- Nothing is picked when the view opens. A click on a can, or the arrow keys on the rack, pull the can out: it slides toward the aisle and its lid shows the embossed rings and a round label. The page panel (80/20, as the bookcase's) then shows the poster, the title, the garden's label line and record, the view's other `order` properties and, with `pageContent`, the note body.
- With `editorPane`, a click or key pick in a wide main-area pane opens the note in an editor beside the view, as the bookcase does, and the page panel stays hidden.
- A click on the pulled can, or `Enter`, opens the note (Mod: a new tab). A pointer that moves onto the pulled can opens its page preview, once per pull.
- `Esc` and a click on the empty rack return the can and close the page panel.
- The index is visually hidden, for screen readers; a live line reads the picked title and its place.

## Development

```sh
npm ci
npm run build
```

The scoped build checks TypeScript and emits the installed `main.js`. Generated bundles, dependencies, the cover cache and `data.json` are ignored. `npm run dev` starts an esbuild watcher, so inspect running processes before starting one.

The walnut and birch-ply veneer tiles in `grain/` are copies of `quartz/static/bookcase-grain/`, and `grain/vault/` holds copies of `quartz/static/vault-grain/`. The vault cannot reach the garden's `/static`. When the garden regenerates the tiles, copy them again.
