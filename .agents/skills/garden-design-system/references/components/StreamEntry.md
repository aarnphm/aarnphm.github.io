# StreamEntry

One stream entry: timestamped body, tags and word count, grouped by day.

**Source:** `quartz/components/stream/Entry.tsx` · styles: `quartz/components/styles/stream.scss`

## Preview

A copy of the server-rendered markup from `/stream` (selector `li.stream-entry`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is static because the stream script (`stream.inline.ts`) fetches more entries and audio.

## Rules

- Metadata in `meta-mono`. Entries load in pages as the reader scrolls.
