# Garden Bookcase

A Bases view, `bookcase`, that draws the query result as one isometric bookcase in the anatomy-kit style. Each group is one or more shelves; each entry is a book with its spine to the front. Scrolling moves the camera along the books in order. The book at the reading position slides out of the shelf and shows its front cover. A page panel next to the case shows the book's title, byline, properties, description and, optionally, the note body.

The garden renders the same `.base` file at build time. Both sides read the same keys and draw from the same geometry module, `quartz/util/base/bookcase/geometry.ts`, and its styles module, `styles.ts`, which esbuild bundles by relative path. `src/scene-model.ts` is the only file that calls them.

## View keys

| key             | option type | default        | meaning                                                                                                                                          |
| --------------- | ----------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `type`          | -           | `bookcase`     | view id                                                                                                                                          |
| `image`         | property    | `note.posters` | cover image, as in the cards view                                                                                                                |
| `spineTitle`    | property    | `note.title`   | spine label; the spine shows the text before the first `:`                                                                                       |
| `spineAuthor`   | property    | `note.author`  | small label at the spine foot; a list shows 2 names, then `et al.`                                                                               |
| `spineColor`    | property    | none           | CSS colour; if absent or invalid, the colour sampled from the cover                                                                              |
| `bookHeight`    | slider      | `220`          | tallest book in world units, 120 to 320                                                                                                          |
| `shelfStyle`    | dropdown    | `straight`     | `straight`, `corner`, `spine`, `wall`, `zigzag` or `ladder`; an unknown value draws `straight`                                                   |
| `shelfMaterial` | dropdown    | `paper`        | `paper`, `walnut`, `birch-ply`, `black-lacquer` or `steel` for every furniture part; unset keeps the style's own materials. Books do not change. |
| `shelfWidth`    | slider      | `600`          | shelf length in world units; a group wraps onto a new shelf at this width, 320 to 1200                                                           |
| `scrollStep`    | slider      | `56`           | px of scroll per book, 24 to 120                                                                                                                 |
| `showTags`      | toggle      | `true`         | label plates on the shelves; a topic's `showTags` overrides it                                                                                   |
| `pageContent`   | toggle      | `true`         | render the note body in the page panel                                                                                                           |

`filters`, `sort`, `groupBy`, `limit` and `order` keep their Bases meaning. `order` lists the properties on the page panel.

## Topics and tags

A topic is a value of the view's `groupBy` property or a key of `topicShelves`. A book is in a topic when its `groupBy` value is the topic, or when it carries the topic as a tag: the tag itself or a tag nested under it (`p/nf/essays` is in `p/nf`; a leading `#` is ignored). A book counts once per topic. Topics with one book share `other`. `all` is the view as Bases groups it, in the view's own style. A topic shows only its books, on shelves grouped by `topicGroupBy`, and its groups share runs. When that value is a list, the book goes on the shelf of its first item. Shelves sort by name, and books without a value come last on a shelf with no label.

Tags narrow what is on show: a book shows when it is in the topic and carries every pressed tag. The tags listed are those of the books in the topic, most books first. A tag that every one of those books carries is not listed. A new topic releases the pressed tags.

The topic, the pressed tags and the open column (topics or tags) are kept per viewer in the plugin's `data.json`, by `.base` path and view name. They are never written to the `.base` file.

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

`displayNames` is the name on the topic's tab, and on the tag's toggle when the key is also a tag. Without one, `p/` is removed and `nf` reads `non-fiction`. An entry whose style is unknown uses the default for that topic. With a known style and no material, the style's own materials apply. `showTags` on an entry overrides the view's `showTags` inside that topic. The garden's `topicKeys`, `topicLabel`, `topicShelf`, `hasTopicTag`, `topicShortcuts` and `TOPIC_SHELVES` in `styles.ts` hold these rules. Without an entry, `philosophy` is `wall`/`walnut`, `p/fiction` is `corner`/`paper`, `p/nf` is `ladder`/`birch-ply` and `p/scifi` is `spine`/`steel`. Other topics use the view's `shelfStyle` and `shelfMaterial`.

## Controls

The controls sit in a column at the top right of the stage, as in the garden:

- Zoom in, zoom out, fit and full screen buttons. Zoom steps by 1.35, from the fitted scale to `max(2, 4 × fitted)`. A button is disabled at its limit. `+` or `=` and `-` zoom while focus is in the view. Zoom keeps the focused book in place; fit returns to the fitted scale with no pan. Full screen shows the view alone; `Esc` or the button leaves it.
- Under them, a two-part switch picks the column: topics or tags. It is not shown when only one of them has something to list.
- The column: a collapsed button shows its shortcut; the chosen topic and the pressed tags are filled and open leftward to "name count". The arrow keys, `Home` and `End` move between buttons, and a letter jumps to the next button whose shortcut starts with it. On a topic, these keys also show the topic. On a tag, `Enter` or `Space` presses it. The column stops 5.5rem above the stage bottom and scrolls on a short stage. In a view narrower than 600 px, the zoom buttons sit in a row.

Drag the stage with a mouse or pen to pan. A press that moves less than 5 px is still a click. After a drag, the click is dropped, so no book opens or lifts. The pan holds until the focus changes; then it eases back to zero and the camera follows the scroll again. The zoom stays. A touch scrolls the books, as before. A new topic or tag starts at its first book, from the top, at the fitted scale.

The camera keeps the pulled-out book clear of the column. Near the left wall it keeps the book on the stage, which wins when both cannot hold.

## Layout

In a pane, the stage fills the height of the element that scrolls: containerEl or its nearest ancestor with `overflow-y: auto | scroll` and a height of its own. A resize observer follows that element, so a pane that gets its height after the view loads still fills. The stage is as wide as the case at its scale (SPEC 4.3, fitted to the garden's fit width), at most 64% of the view; the page panel takes the rest. In an embed, the view scrolls itself in 520 px.

Covers are decoded once, in idle time, into 480 px thumbnails under `cache/covers/` with a sampled spine colour. The index is in `data.json`.

## Development

```sh
npm ci
npm run build
```

The scoped build checks TypeScript and emits the installed `main.js`. Generated bundles, dependencies, the cover cache and `data.json` are ignored. `npm run dev` starts an esbuild watcher, so inspect running processes before starting one.
