# StreamEntry

One stream entry: timestamped body, tags and word count, grouped by day.

**Source:** `quartz/components/stream/Entry.tsx` · styles: `quartz/components/styles/stream.scss`

**Corrections:** the source draws the date, tags, share buttons, word count and description in `gray`. This rendition uses `darkgray`.

## Preview

A copy of the server-rendered markup from `/stream/on/2025/11/28` (selector `li.stream-entry`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static because the stream script (`stream.inline.ts`) fetches more entries and audio.

## Rules

- A metadata column on the left in `meta-mono`: date and time with the zone, tags and links. The body sits right of it with the entry title, a `url:` line and the transcluded note text.
- Date, tags, share buttons and word count are `gray` in the source. This rendition uses `darkgray`.
- Entries load in pages as the reader scrolls.
