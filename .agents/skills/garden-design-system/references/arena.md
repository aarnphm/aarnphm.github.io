# Arena

Arena mirrors aarnphm's are.na channels: an index of channels, a channel page of blocks, an entry page per block, and a reader at `/arena/feed`. It is the source of the tool-surface recipe.

## Structure

- **Index** (`ArenaIndex`): a centred `width-arena` column, the title block, then `ArenaChannelList`.
- **Channel** (`ChannelContent`): title block, an `ArenaControls` row, then sections (pinned first), each a lowercase mono header over an `ArenaBlockGrid`. Grid and list views share the same blocks.
- **Block view**: opens over the channel with the block's preview (image, embed, PDF, map or text), a metadata sidebar (mono `meta` labels in `darkgray`, values in `dark`), connections and the wikilink trail. It should be a native `<dialog>` that takes focus and returns it on close. The current `div` overlay does neither (debt).
- **Reader** (`ArenaReader`): a full-height column with a header bar, an optional rail and a queue. The rail widens the column to `width-arena-wide`.

## Tokens

`arena-line`, `arena-surface`, `arena-selected`, `control-collection`, `control-reader`, `width-arena`, `width-arena-wide`. The type styles are `arena-channel-title`, `arena-block-title`, `arena-row`, `arena-control` and `arena-search`.

## Rules

- Hide the site's header row, footer and heading anchors on Arena pages. The title block carries the page.
- Thumbnails are square-cornered, 16:10, on a `lightgray` slot while loading.
- Hover never inverts. Rows and blocks take `arena-surface`, and text links underline.
- Arena follows the site theme through `saved-theme`. A block or search field that turns dark when the OS is dark but the site is light is a bug.
- A failed embed shows a mono message and a source link in the block's place.
