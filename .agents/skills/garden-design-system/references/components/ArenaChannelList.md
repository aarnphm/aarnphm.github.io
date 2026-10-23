# ArenaChannelList

A framed list of Arena channels, each with a header row and up to three preview rows.

**Source:** `quartz/components/styles/arena.scss` (`.arena-channels-list`, `.arena-channel-row-header`, `.arena-channel-row-preview-item`).

**Corrections:** the source draws the count and JSON link in `gray` and the focus ring in sage. This rendition uses `darkgray` and `focus`.

## Inputs

- Per channel: name and link, block count, JSON link.
- Up to three preview rows, each with a 64px 16:10 thumbnail (or an empty `lightgray` slot) and a one-line title.

## Rules

- One `1px arena-line` frame. Channels and preview rows separate with `arena-line` rules.
- Header: `arena-surface` ground, padded `space-2 space-3`. Name in `arena-channel-title`; meta in mono at 0.7rem.
- Preview rows: `darkgray` text at 0.85rem, padded `0.35rem space-3`. Hover: `dark` text with an underline. Focus: a `2px` ring in `focus`, offset 2px.
- Width `width-arena`, centred.
